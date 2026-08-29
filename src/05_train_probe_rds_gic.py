"""Stage 5 - train one probe per layer, compute the Regret Dominance Score on
the final layer, split the neurons into RegretD / Non-RegretD / DualD, and
measure probe accuracy after ablating each group and each combination.

Requires a GPU and the outputs of stages 3 and 4.
"""
import argparse
import csv
import datetime
import json
import os
import pickle
import random
import shutil
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lib.probe_models import MLPClassifier

# Root of the experiment tree, holding datasets/ and results/. Override with
# the REGRET_ROOT environment variable; defaults to the repository root.
ROOT = os.environ.get(
    "REGRET_ROOT",
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
)
from sklearn.metrics import mutual_info_score

_parser = argparse.ArgumentParser(description=__doc__)
_parser.add_argument('--model_size', default='7b', choices=['7b', '13b', '70b'],
                     help='Target model scale. Selects the layer count, the hidden '
                          'size and the results/ sub-directory.')
_args, _ = _parser.parse_known_args()


class Variables:
    task_type = _args.model_size  # '7b' '13b' '70b'
    reload_data = False
    answer_number = 2
    if task_type == '7b':
        total_layer_num = 33
        input_dim = 4096
    elif task_type=='13b':
        total_layer_num = 41
        input_dim = 5120
    elif task_type=='70b':
        total_layer_num = 81
        input_dim = 8192
    best_layer = None
    mixdatapath = os.path.join(ROOT, 'results', 'mix-data-pt_' + task_type)
    use_mixed_hidden_states = True # Disregard False for now. 
    offset_1 = 10000 # can not reach 
    offset_2 = 20000 # can not reach 
    offset_3 = 30000 # can not reach 
    extend_offset = 40000
    select_hidden_states = 'layer_outputs_hidden_states' # attention_hidden_states mlp_hidden_states, layer_outputs_hidden_states, layer_outputs_hidden_states
    
    figure_1_S_CDI_in_different_layers = None
    figure_2_front_backward_tokens = None
    figure_3_gyperparameter_threshold_CRDS = None
    figure_4_mutual_information = None
    table_1_probe_classification_performances = None
    Table_2_previous_token_prediction_performances = None
    Table_3_probe_classification_performances_after_interven = None

    csv_writer_first_round = None
    csv_writer_second_round = None
    csv_writer_third_round = None
    
local_variables = Variables()





def classify_neurons(RDS, threshold_std=1.0):
    """
    Classify neurons into three categories based on their RDS values: mean and standard deviation.

    Parameters:
        RDS: np.array, shape (num_neurons,)
            The Regret Dominant Score of each neuron.
        threshold_std: float, default 1.0
            The multiple of standard deviations used for classification.

    Returns:
        dict, with three keys: 'regret_dominant', 'no_regret_dominant', 'mixed'
        Each key corresponds to an array of neuron indices.
    """
    RDS = RDS.astype(np.float32)
    mu = np.mean(RDS)
    sigma = np.std(RDS)

    regret_dominant = np.where(RDS > mu + threshold_std * sigma)[0]
    no_regret_dominant = np.where(RDS < mu - threshold_std * sigma)[0]
    mixed = np.where((RDS >= mu - threshold_std * sigma) & (RDS <= mu + threshold_std * sigma))[0]

    return {
        'regret_dominant': regret_dominant,
        'no_regret_dominant': no_regret_dominant,
        'mixed': mixed
    }

# Function to calculate accuracy, sensitivity, and specificity
def calculate_metrics(preds, labels):
    # Convert predictions to class indices
    preds = torch.argmax(preds, dim=1)
    # Calculate confusion matrix
    TP = ((preds == 1) & (labels == 1)).sum().item()  # True Positives
    TN = ((preds == 0) & (labels == 0)).sum().item()  # True Negatives
    FP = ((preds == 1) & (labels == 0)).sum().item()  # False Positives
    FN = ((preds == 0) & (labels == 1)).sum().item()  # False Negatives
    # Calculate metrics
    accuracy = (TP + TN) / (TP + TN + FP + FN + 1e-10)  # Add small epsilon to avoid division by zero
    sensitivity = TP / (TP + FN + 1e-10)  # Sensitivity (Recall)
    specificity = TN / (TN + FP + 1e-10)  # Specificity
    return accuracy, sensitivity, specificity

class Config:
    def __init__(self):
        self.data = type('', (), {})()
        self.data.num_of_labels = 2

def map_tokenized_words(P, words_list):
    result_indices = []
    i = 0
    
    while i < len(P):
        # Check if current token directly matches any word in words_list
        if P[i] in words_list:
            result_indices.append(i)
            i += 1
            continue
        
        # Try combining with subsequent tokens
        current_token = P[i]
        j = i + 1
        found_match = False
        
        while j < len(P):
            combined_token = current_token + P[j]
            # Check if this combination is in words_list
            if combined_token in words_list:
                # If we find a match, add all indices used to form this word
                result_indices.extend(range(i, j+1))
                i = j + 1
                found_match = True
                break
            
            # Continue combining
            current_token = combined_token
            j += 1
        
        # If no combination worked, move to next token
        if not found_match:
            i += 1
    
    return result_indices

# what's mixed data: 
# answer_number: number of answer stages used; four ID/index sites depend on it.
class CustomDataset(Dataset):
    def __init__(self, indices, type, specific_layer=32):
        self.specific_layer = specific_layer
        self.type = type
        self.indices = indices # train or test

        if local_variables.use_mixed_hidden_states:
            # update self.indices
            self.mix_indices = np.concatenate([
                # self.indices,  
                self.indices + local_variables.offset_1,  # Q_ID + offset_1
                # self.indices + local_variables.offset_2,  # Q_ID + offset_2
                self.indices + local_variables.offset_3  # Q_ID + offset_3
            ])
            self.answer_number = local_variables.answer_number
        else:
            self.answer_number = 1
        
        self.eval_model_name = 'GPT-4-'
        self.jsondata_path = os.path.join(ROOT, 'datasets', local_variables.task_type + '_output_generated_data_with_target_model_answer_label_confident.json')
        self.pre_store_path = os.path.join(ROOT, 'results', 'pre-store_' + local_variables.task_type)
        self.basepath = os.path.join(ROOT, 'results', 'regret_' + local_variables.task_type)
        self.mixdatapath = local_variables.mixdatapath

        if not os.path.exists(self.mixdatapath):
            os.makedirs(self.mixdatapath)
        if not os.path.exists(self.pre_store_path):
            os.makedirs(self.pre_store_path)
        self.all_label_dict, self.all_mask_dict = self.loadjson() # for label & mask. 
        
        key_emo_entity_path = os.path.join(ROOT, 'datasets', 'key_position.json')
        with open(key_emo_entity_path, "r") as f:
            key_emo_entity_lists = json.load(f)
        self.key_emo_entity_mild = key_emo_entity_lists['Mild']
        self.key_emo_entity_moderate = key_emo_entity_lists['Moderate']
        self.key_emo_entity_severe = key_emo_entity_lists['Severe']
        
        self.combined_data_path = os.path.join(self.pre_store_path, 'combined_data_20250121.pt')
        if self.type=='train':
            self.mixed_data_path = os.path.join(self.mixdatapath, str(self.answer_number) + '_combined_data_' + local_variables.select_hidden_states + '.pt')
        elif self.type=='test':
            self.mixed_data_path = os.path.join(self.mixdatapath, str(self.answer_number) + '_combined_data_test_' + local_variables.select_hidden_states + '.pt')
        
        if local_variables.use_mixed_hidden_states:
            if os.path.exists(self.mixed_data_path):
                print("Loading all mix data...")
                self.data = torch.load(self.mixed_data_path)
                self.mix_indices = np.load(os.path.join(self.mixdatapath, self.type + '_mix_indices.npy'))
                with open(os.path.join(self.mixdatapath, self.type + '_all_label_dict.pickle'), 'rb') as file:
                    self.all_label_dict = pickle.load(file)
            else:
                print("Currently pre-reading the scattered files and saving them as a whole mix-data...")
                self.data = self.load_mix_data()
                torch.save(self.data, self.mixed_data_path)
                print(f"The overall mix-data has been saved to {self.mixed_data_path}")            
        else:
            if os.path.exists(self.combined_data_path):
                print("Loading all pre-stored data...")
                self.data = torch.load(self.combined_data_path)
            else:
                print("Currently pre-reading the scattered files and saving them as a whole data...")
                self.data = self.load_data()
                torch.save(self.data, self.combined_data_path)
                print(f"The overall data has been saved to {self.combined_data_path}")        

    def loadjson(self):
        all_label_dict = {} # all data: train + test
        all_emo_mask_dict = {}
        with open(self.jsondata_path, 'r', encoding="utf-8") as file:
            for line in file:
                entry = json.loads(line)
                Q_ID = entry["ID"]
                emotion_mask  = entry["emotion_mask"]
                if local_variables.use_mixed_hidden_states:
                    # Generate new keys using offset
                    label_dict = {
                        Q_ID + local_variables.offset_1: entry[self.eval_model_name + "ground_truth_initial"],       # e.g. Q_ID=1 -> 20001
                        # Q_ID + local_variables.offset_2: entry[self.eval_model_name + "ground_truth_weak"],  # e.g. Q_ID=1 -> 30001
                        Q_ID + local_variables.offset_3: entry[self.eval_model_name + "ground_truth_strong"],          # e.g. Q_ID=1 -> 40001
                    }
                    mask_dict = {
                        Q_ID + local_variables.offset_1: emotion_mask, # Q & initial answer
                        # Q_ID + local_variables.offset_2: emotion_mask, # Q & answer after weak hint 
                        Q_ID + local_variables.offset_3: emotion_mask, # Q & answer after strong hint
                    }

                    all_label_dict.update(label_dict)
                    all_emo_mask_dict.update(mask_dict)
                else:
                    value_dict = {
                        "fake_evidence": entry["fake_evidence"],
                        "question": entry["question"],
                        "ground_truth": entry["ground_truth"],
                        "gpt_4o_mini_initial_answer_with_fake_evidence": entry["gpt-4o-mini-initial_answer_with_fake_evidence"],
                        "gpt_4o_mini_confidence_score_initial": entry["gpt-4o-mini-confidence_score_initial"],
                        "weak_hint_true": entry["weak_hint_true"],
                        "gpt_4o_mini_reflection_answer_weak": entry["gpt-4o-mini-reflection_answer_weak"],
                        "gpt_4o_mini_confidence_score_weak": entry["gpt-4o-mini-confidence_score_weak"],
                        "strong_hint_true": entry["strong_hint_true"],
                        "gpt_4o_mini_reflection_answer_strong": entry["gpt-4o-mini-reflection_answer_strong"],
                        "gpt_4o_mini_confidence_score_strong": entry["gpt-4o-mini-confidence_score_strong"]
                    }

                    label_dict = {
                        # "ground_truth": entry["ground_truth"],
                        "pure_ground_truth": entry["pure_ground_truth"],
                        "fake_evidence_initial_ground_truth": entry["fake_evidence_initial_ground_truth:"],
                        "weak_hint_res_ground_truth": entry["weak_hint_res_ground_truth:"],
                        "strong_hint_res_ground_truth": entry["strong_hint_res_ground_truth:"]
                    }

                    all_label_dict[Q_ID] = label_dict
                # switch to mixed label

        return all_label_dict, all_emo_mask_dict

    def load_mix_data(self):
        data = {}
        all_hidden_states = {}
        for idx in tqdm(self.indices):
            # Missing the original

            input_token_position_mask = self.all_mask_dict[idx + local_variables.offset_1]["input"]
            llm_res_token_position_mask = self.all_mask_dict[idx + local_variables.offset_1]["llm_res"]

            # ============================ consider token position ============================
            # change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            
            # temp 2012
            # try: 
                    # change !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            # except:
            
            #     'input_hidden_states_pure': {
            #         'attention': token_data_pure_by_mask['attention_hidden_states']['layer_-1'],
            #         'mlp': token_data_pure_by_mask['mlp_hidden_states']['layer_-1'],
            #         'layer_output': token_data_pure_by_mask['layer_outputs_hidden_states']['layer_-1']
            #     },
            #     'input_hidden_states_initial': {
            #         'attention': token_data_initial_by_mask['attention_hidden_states']['layer_-1'],
            #         'mlp': token_data_initial_by_mask['mlp_hidden_states']['layer_-1'],
            #         'layer_output': token_data_initial_by_mask['layer_outputs_hidden_states']['layer_-1']
            #     },
            #     'input_hidden_states_with_weak_hint': {
            #         'attention': token_data_with_weak_hint_by_mask['attention_hidden_states']['layer_-1'],
            #         'mlp': token_data_with_weak_hint_by_mask['mlp_hidden_states']['layer_-1'],
            #         'layer_output': token_data_with_weak_hint_by_mask['layer_outputs_hidden_states']['layer_-1']
            #     },
            #     'input_hidden_states_with_strong_hint': {
            #         'attention': token_data_with_strong_hint_by_mask['attention_hidden_states']['layer_-1'],
            #         'mlp': token_data_with_strong_hint_by_mask['mlp_hidden_states']['layer_-1'],
            #         'layer_output': token_data_with_strong_hint_by_mask['layer_outputs_hidden_states']['layer_-1']
            #     }
            # }
            
            # load LLM responses hidden_states
            print('QID:', idx)
            q_responses_path_pure = os.path.join(self.basepath, 'Q_ID_' + str(idx), 'LLM_responses', 'Response_0_token_data_llm_response_hidden_state.pt')
            q_responses_path_initial = os.path.join(self.basepath, 'Q_ID_' + str(idx), 'LLM_responses', 'Response_1_token_data_llm_response_hidden_state.pt')
            q_responses_path_with_weak_hint = os.path.join(self.basepath, 'Q_ID_' + str(idx), 'LLM_responses', 'Response_2_token_data_llm_response_hidden_state.pt')
            q_responses_path_with_strong_hint = os.path.join(self.basepath, 'Q_ID_' + str(idx), 'LLM_responses', 'Response_3_token_data_llm_response_hidden_state.pt')

            llm_token_data_pure = torch.load(q_responses_path_pure)
            llm_token_data_initial = torch.load(q_responses_path_initial)
            llm_token_data_with_weak_hint = torch.load(q_responses_path_with_weak_hint)
            llm_token_data_with_strong_hint = torch.load(q_responses_path_with_strong_hint)

            # add by abner
            
            llm_res_strong_hint_filtered_tokens_test = [token['token_text'] for token in llm_token_data_with_strong_hint]


            #self.key_emo_entity_mild
            #self.key_emo_entity_moderate
            #self.key_emo_entity_severe
            concern_entity = ['regret'] #self.key_emo_entity_mild + self.key_emo_entity_moderate + self.key_emo_entity_severe
            mapping_idx = map_tokenized_words(llm_res_strong_hint_filtered_tokens_test, concern_entity)

            # ============================ consider token position ============================
            llm_token_data_pure_by_mask = llm_token_data_pure[-1]
            llm_token_data_initial_by_mask = llm_token_data_initial[-1]
            all_hidden_states.update({idx + local_variables.offset_1: llm_token_data_initial_by_mask['hidden_states']})

            llm_token_data_with_weak_hint_by_mask = llm_token_data_with_weak_hint[-1]

            if len(mapping_idx) > 0:
                for m in range(len(mapping_idx)):
                    if m==0:
                        all_hidden_states.update({idx + local_variables.offset_3: llm_token_data_with_strong_hint[mapping_idx[m]]['hidden_states']})
                    else:
                        all_hidden_states.update({idx + local_variables.extend_offset*m: llm_token_data_with_strong_hint[mapping_idx[m]]['hidden_states']})
                        # update self.all_label_dict
                        self.all_label_dict.update({idx + local_variables.extend_offset*m: "True"})
                        # update index
                        self.mix_indices = np.append(self.mix_indices, idx + local_variables.extend_offset*m)
            else:
                all_hidden_states.update({idx + local_variables.offset_3: llm_token_data_with_strong_hint[-1]['hidden_states']})
            
            # if local_variables.use_mixed_hidden_states:       
            #     #     # idx: last_token_data_pure['layer_outputs_hidden_states']['layer_-1'],  
            #     #     idx + local_variables.offset_1: token_data_initial_by_mask[local_variables.select_hidden_states]['layer_-1'],  
            #     #     # idx + local_variables.offset_2: token_data_with_weak_hint_by_mask[local_variables.select_hidden_states]['layer_-1'], 
            #     #     idx + local_variables.offset_3: token_data_with_strong_hint_by_mask[local_variables.select_hidden_states]['layer_-1'], 
            #     # }    

            #         # idx: last_token_data_pure['layer_outputs_hidden_states']['layer_-1'],  
            #         idx + local_variables.offset_1: llm_token_data_initial_by_mask['hidden_states'],
            #         # idx + local_variables.offset_2: llm_last_token_data_with_weak_hint[local_variables.select_hidden_states]['layer_-1'],#last_token_data_with_weak_hint[local_variables.select_hidden_states]['layer_-1'], 
            #         idx + local_variables.offset_3: llm_token_data_with_strong_hint_by_mask['hidden_states'],
            #     }    

        # save mix_metrics & all_label_dict
        # Persist the label dictionary as a pickle file.
        with open(os.path.join(self.mixdatapath, self.type + '_all_label_dict.pickle'), 'wb') as f:
            pickle.dump(self.all_label_dict, f)
        np.save(os.path.join(self.mixdatapath, self.type + '_mix_indices.npy'), self.mix_indices)
        if local_variables.use_mixed_hidden_states:        
            return all_hidden_states
    
    def __len__(self):
        return len(self.mix_indices) #len(self.indices) * self.answer_number

    def __getitem__(self, idx):
        if local_variables.use_mixed_hidden_states:
            hidden_states_single = self.data[self.mix_indices[idx]][self.specific_layer].squeeze() # layer 0 ~ layer 32
            if hidden_states_single.shape[0]!=local_variables.input_dim:
                print(hidden_states_single.shape[0])
                exit(0)
            label = self.all_label_dict[self.mix_indices[idx]]
            label = 1 if label == 'True' else 0
        else:
            data = self.data[self.indices[idx]]
        
        # Fix the bug
        if not isinstance(hidden_states_single, torch.Tensor):
            hidden_states_single = torch.tensor(hidden_states_single, dtype=float)
        elif hidden_states_single.dtype != torch.float32:
            # Stage 3 runs the target model in fp16, so the stored activations are
            # half precision while the probe parameters are float32. Widening
            # fp16 -> fp32 is exact (every fp16 value is representable in fp32),
            # so no recorded activation changes; it only makes the dtypes match.
            hidden_states_single = hidden_states_single.float()
        
        if len(hidden_states_single.shape) == 2:
            hidden_states_single = hidden_states_single.mean(dim=0)
        
        return hidden_states_single, label

from sklearn.metrics import confusion_matrix

from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix

def test(model, test_loader, regret_indices=None):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()
            # Deactivate the selected neurons by setting their activation to -1.
            if regret_indices is not None:
                # Indices must be a LongTensor on the same device as x.
                indices = torch.LongTensor(regret_indices).to(x.device)
                x[:, indices] = -1
            
            outputs = model(x)
            _, predicted = torch.max(outputs, 1)
            y_true.extend(y.cpu().numpy())
            y_pred.extend(predicted.cpu().numpy())
    
    # Convert the collected lists to tensors.
    y_true = torch.tensor(y_true)
    y_pred = torch.tensor(y_pred)
    
    # Basic classification metrics.
    accuracy = (y_true == y_pred).float().mean().item()
    precision = precision_score(y_true, y_pred, average='binary')  # binary task
    recall = recall_score(y_true, y_pred, average='binary')  # sensitivity == recall
    f1 = f1_score(y_true, y_pred, average='binary')
    
    # Specificity from the confusion matrix.
    cm = confusion_matrix(y_true, y_pred)
    TN = cm[0, 0]
    FP = cm[0, 1]
    specificity = TN / (TN + FP + 1e-10)  # epsilon avoids division by zero
    
    # Report them.
    print(f"Accuracy: {accuracy:.4f}, Sensitivity/Recall: {recall:.4f}, "
          f"Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    
    # Return every metric.
    return accuracy, recall, specificity, precision, f1

# def test(model, test_loader, regret_indices=None):
#     y_true, y_pred = [], []
#     with torch.no_grad():
#         for x, y in test_loader:
#             x, y = x.cuda(), y.cuda()
#             if regret_indices is not None:
#                 x[:, indices] = -1
            
#             _, predicted = torch.max(outputs, 1)
    
    
    
    
#     if len(torch.unique(y_true)) < 2 or len(torch.unique(y_pred)) < 2:
#         return accuracy, 0.0, 0.0, 0.0, 0.0
    
    
    
#     if cm.shape == (2, 2):
#         TN, FP = cm[0, 0], cm[0, 1]
#         FN, TP = cm[1, 0], cm[1, 1]
#     else:
    
#           f"Specificity: {specificity:.4f}, Precision: {precision:.4f}, F1: {f1:.4f}")
    
#     return accuracy, recall, specificity, precision, f1

# for memory match
def compute_RDS_corrected(activations_regret, activations_no_regret, epsilon=1e-8):
    """
    Regret Dominance Score for unpaired activations (paper, Eq. 1).
    """
    # Mean activation of each neuron in the two states.
    regret_mean = np.mean(activations_regret, axis=0)
    no_regret_mean = np.mean(activations_no_regret, axis=0)
    
    # Regret Dominance Score.
    RDS = regret_mean / (regret_mean + no_regret_mean + epsilon)
    return RDS

def entropy(p):
    """Shannon entropy (bits) of a discrete distribution."""
    p = p[p > 0]  # drop zero-probability events
    return -np.sum(p * np.log2(p))

def compute_mutual(regret_indices, regret_indices1, mix_ind, all_activations):
    # Accurately calculate the mutual information of three mutually exclusive groups
    def calc_group_mi(group_a, group_b):
        if len(group_a) == 0 or len(group_b) == 0:
            return 0.0
        # Activations of each group: (n_samples, group_size).
        act_a = all_activations[:, group_a]  # activations of group A
        act_b = all_activations[:, group_b]  # activations of group B
        
        # Per-sample mean activation of each group.
        mean_a = np.mean(act_a, axis=1)  # (n_samples,)
        mean_b = np.mean(act_b, axis=1)  # (n_samples,)
        
        # Discretise into 20 bins.
        bins = np.linspace(0, 1, 20)
        disc_a = np.digitize(mean_a, bins)
        disc_b = np.digitize(mean_b, bins)
        
        # Mutual information between the two discretised signals.
        mi = mutual_info_score(disc_a, disc_b)
        
        # Normalise by the geometric mean of the entropies.
        entropy_a = entropy(np.bincount(disc_a)/len(disc_a))
        entropy_b = entropy(np.bincount(disc_b)/len(disc_b))
        return mi / np.sqrt(entropy_a * entropy_b)
    
    # Build the 3x3 mutual information matrix.
    groups = [regret_indices, regret_indices1, mix_ind]
    group_names = ['RegretD', 'NonRegretD', 'CrossD']
    mi_matrix = np.zeros((3,3))
    
    for i in range(3):
        for j in range(3):
            mi_matrix[i,j] = calc_group_mi(groups[i], groups[j])

    return mi_matrix, group_names

def save_neuron_indices_and_rds(layer, threshold_value, regret_indices, non_regret_indices, mixed_indices, RDS):
    """
    Save the neuron indices and RDS values under layer/threshold directories.
    
    Parameters:
        layer: index of the layer being analysed.
        threshold_value: the tau used for the split.
        regret_indices: RegretD neuron indices.
        non_regret_indices: Non-RegretD neuron indices.
        mixed_indices: DualD neuron indices.
        RDS: the full per-neuron RDS array.
    """
    # Base directory for the neuron indices.
    neuron_indices_base_dir = os.path.join(ROOT, 'results', f'regret_neuron_indices_{local_variables.task_type}')
    
    layer_threshold_dir = os.path.join(neuron_indices_base_dir, f"layer-{layer}", f"threshold-{threshold_value:.2f}")
    os.makedirs(layer_threshold_dir, exist_ok=True)
    
    # Save the indices of the three neuron groups.
    indices_to_save = {
        'regret_dominant': regret_indices,
        'no_regret_dominant': non_regret_indices,
        'mixed': mixed_indices
    }
    
    for neuron_type, indices in indices_to_save.items():
        save_path = os.path.join(layer_threshold_dir, f"{neuron_type}_indices.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(indices, f)
    
    # Also save summary statistics.
    stats_info = {
        'threshold': threshold_value,
        'layer': layer,
        'regret_count': len(regret_indices),
        'non_regret_count': len(non_regret_indices),
        'mixed_count': len(mixed_indices),
        'total_neurons': local_variables.input_dim,
        'RDS_mean': float(np.mean(RDS)),
        'RDS_std': float(np.std(RDS)),
        'RDS_min': float(np.min(RDS)),
        'RDS_max': float(np.max(RDS))
    }
    
    stats_path = os.path.join(layer_threshold_dir, "neuron_stats.json")
    with open(stats_path, "w") as f:
        json.dump(stats_info, f, indent=2)
    
    # Save the full RDS array.
    rds_path = os.path.join(layer_threshold_dir, "RDS_array.npy")
    np.save(rds_path, RDS)
    
    print(f"    Saved indices for layer {layer}, threshold {threshold_value:.2f}")
    print(f"    Regret: {len(regret_indices)}, Non-regret: {len(non_regret_indices)}, Mixed: {len(mixed_indices)}")

def composional_control(model, train_loader, test_loader, best_acc=None, layer=None):
    model.eval()
    accuracy, sensitivity, specificity, _, _ = test(model, test_loader)
    print(f"[Best Test]Accuracy: {accuracy:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")    
    activations_regret = []
    activations_no_regret = []
    all_neuros_state = []
    
    for hidden_states, labels in train_loader:
        hidden_np = hidden_states.cpu().numpy()
        labels_np = labels.cpu().numpy()
        # Separate the regret and non-regret activations.
        mask_regret = (labels_np == 1)
        mask_no_regret = (labels_np == 0)
        activations_regret.append(hidden_np[mask_regret])
        activations_no_regret.append(hidden_np[mask_no_regret])
        all_neuros_state.append(hidden_np)
    
    activations_regret = np.concatenate(activations_regret, axis=0)
    activations_no_regret = np.concatenate(activations_no_regret, axis=0)
    all_neuros_state = np.concatenate(all_neuros_state, axis=0)
    # Sanity check on the sample counts.
    print(f"Regret samples: {activations_regret.shape[0]}")
    print(f"Non-regret samples: {activations_no_regret.shape[0]}")
    # Compute RDS and split the neurons into groups.
    RDS = compute_RDS_corrected(activations_regret, activations_no_regret)
    results = []

    first_column = np.array(['Regret', 'Non-Regret', 'Dual', 'Random','Regret & Non-Regret', 'Regret & Dual', 'Non-Regret & Dual'])
    data = pd.DataFrame({'Group': first_column})
    for k in range(1, 51):
        threshold_value = 0.01 * k
        classes = classify_neurons(RDS, threshold_std=threshold_value)
        print(f"Threshold {threshold_value:.2f} - RegretD neurons: {len(classes['regret_dominant'])}")
        print(f"Threshold {threshold_value:.2f} - Non-RegretD neurons: {len(classes['no_regret_dominant'])}")
        mix_ind = classes['mixed']
        non_regret_indices = classes['no_regret_dominant']
        regret_indices = classes['regret_dominant']
        
        if layer is not None:
            save_neuron_indices_and_rds(layer, threshold_value, regret_indices, non_regret_indices, mix_ind, RDS)
        # ====================================================================
        
        # compute 
        mi_matrix, group_names = compute_mutual(regret_indices, non_regret_indices, mix_ind, all_neuros_state)
        # Report them.
        print("\nMutual information matrix over the disjoint neuron groups:")
        print(f"           {group_names[0]:<10} {group_names[1]:<10} {group_names[2]:<10}")
        for i, name in enumerate(group_names):
            print(f"{name:<10} {mi_matrix[i,0]:.3f}      {mi_matrix[i,1]:.3f}     {mi_matrix[i,2]:.3f}")
        mutual_f = f"[threshold_std]: {0.01 * k}; [Mutual Value]{group_names[0]} & {group_names[0]}: {mi_matrix[0,0]:.3f};{group_names[1]} & {group_names[1]}: {mi_matrix[1,1]:.3f};{group_names[2]} & {group_names[2]}: {mi_matrix[2,2]:.3f};{group_names[0]} & {group_names[1]}: {mi_matrix[0,1]:.3f}; {group_names[0]} & {group_names[2]}: {mi_matrix[0,2]:.3f}; {group_names[1]} & {group_names[2]}: {mi_matrix[1,2]:.3f}"
        results.extend([mutual_f])
        #
        results.extend([f"regret_indices: {len(regret_indices)}, non_regret_indices: {len(non_regret_indices)}, mix_ind: {len(mix_ind)}"])
        regret_indices_new = np.concatenate((regret_indices, non_regret_indices), axis=0)
        all_neuros = np.concatenate((regret_indices, non_regret_indices, mix_ind), axis=0)
        total_count = len(regret_indices_new)
        print(total_count)
        random_neurons = random.sample(all_neuros.tolist(), total_count)
        accuracy_regret, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=regret_indices)
        results.extend([f"[control: regret_indices to -1], Accuracy: {accuracy_regret:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])
        accuracy_non_regret, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=non_regret_indices)
        results.extend([f"[control: non_regret_indices to -1], Accuracy: {accuracy_non_regret:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])
        accuracy_dual, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=mix_ind)
        results.extend([f"[control: dual_indices to -1], Accuracy: {accuracy_dual:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])

        accuracy_random, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=random_neurons)
        print(f"i: {0.01*k}, random_neurons, Accuracy: {accuracy_random:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")    
        results.extend([f"[control: random neuros (len(regret_indices)+len(non_regret_indices)) to -1], Accuracy: {accuracy_random:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])
        
        accuracy_regret_non_regret, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=regret_indices_new)
        print(f"i: {0.01*k},regret_indices_new, Accuracy: {accuracy_regret_non_regret:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")    
        results.extend([f"[control: regret_indices + non_regret_indices to -1], Accuracy: {accuracy_regret_non_regret:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])

        accuracy_regret_dual, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=np.concatenate((regret_indices, mix_ind), axis=0))
        print(f"i: {0.01*k},regret_indices_new, Accuracy: {accuracy_regret_dual:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")    
        results.extend([f"[control: regret_indices + dual_indices to -1], Accuracy: {accuracy_regret_dual:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])

        accuracy_non_regret_dual, sensitivity, specificity, precision, f1 = test(model, test_loader, regret_indices=np.concatenate((non_regret_indices, mix_ind), axis=0))
        print(f"i: {0.01*k},regret_indices_new, Accuracy: {accuracy_non_regret_dual:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")    
        results.extend([f"[control: non_regret_indices + dual_indices to -1], Accuracy: {accuracy_non_regret_dual:.4f}, sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}, precision: {precision:.4f}, F1: {f1:.4f}"])
        
        # Accuracy drop relative to the unablated probe.
        new_column = np.array([np.abs(best_acc - accuracy_regret), np.abs(best_acc - accuracy_non_regret),np.abs(best_acc - accuracy_dual), np.abs(best_acc - accuracy_random), np.abs(best_acc - accuracy_regret_non_regret),np.abs(best_acc - accuracy_regret_dual),np.abs(best_acc - accuracy_non_regret_dual)])
        # Name the column after the threshold.
        column_name = f'{0.01*k}'
        # Append the column.
        data[column_name] = new_column

    # Write the table to CSV.
    try:
        data.to_csv(local_variables.task_type+ '_hot_image_for_threshold.csv', index=False)
        print("Table written to CSV.")
    except Exception as e:
        print(f"Failed to write the CSV file: {e}")

    return results




def get_time():
    # Timestamp with minute resolution.
    now = datetime.datetime.now()
    formatted_time = now.strftime("%Y%m%d%H%M")
    return formatted_time

# Probing Functions
def probing():
    EPOCH = 100
    batch_size = 256
    training = True
    config = Config()
    cur_time = get_time()
    root_dir = os.path.join(ROOT, 'results', 'regret_' + local_variables.task_type)
    best_model_path = os.path.join(ROOT, 'results', 'best_model_' + local_variables.task_type)
    draw_files_path = os.path.join(ROOT, 'results', 'draw', local_variables.task_type, 'draw_' + cur_time)

    # =============================================== Draw Flow ================================================
    # ======================================== Save image/pdf/data/model =======================================
    # ==========================================================================================================
    # Figure-1: S-CDI in different layers, to sure best layer
    # Figure-2: top - 3, top, top + 3: Accuracy in different layer
    # Figure-4: Mutual Information in different layers
    
    # Layer | Probe-Acc | S-CDI-Llama-2-7b
    local_variables.figure_1_S_CDI_in_different_layers = os.path.join(draw_files_path, 'figure_1_S_CDI_in_different_layers.csv')
    local_variables.figure_2_front_backward_in_different_layers = os.path.join(draw_files_path, 'figure_2_front_backward_in_different_layers.csv') # different sheet at different model
    
    # hot figure
    local_variables.figure_3_gyperparameter_threshold_CRDS = os.path.join(draw_files_path, 'figure_3_gyperparameter_threshold_CRDS.csv') # different sheet at different model
    
    local_variables.figure_4_mutual_information_in_different_layers = os.path.join(draw_files_path, 'figure_4_mutual_information_in_different_layers.csv')


    # Table-1: probe classification best performances, including the ablation experiment results of CRDS
    # Table-2: previous token prediction performances at best layer. Does llm is regret before they speak regret?
    # Table-3: probe classification performances after interven....
    local_variables.table_1_probe_classification_performances = os.path.join(draw_files_path, 'table_1_probe_classification_performances.csv') # different sheet at different model
    local_variables.Table_2_previous_token_prediction_performances = os.path.join(draw_files_path, 'Table_2_previous_token_prediction_performances.csv') # different sheet at different model
    local_variables.Table_3_probe_classification_performances_after_interven = os.path.join(draw_files_path, 'Table_3_probe_classification_performances_after_interven.csv') # different sheet at different model
    
    
    # Table-2-extend: previous token prediction performances at best layer. Does llm is regret before they speak regret?
    # =========================================================================================================== 
    # =========================================================================================================== 
    # =========================================================================================================== 

    # don't run CRDS
    file_config_first_round = {
        'figure-1': local_variables.figure_1_S_CDI_in_different_layers,
    }
    fieldnames_config_first_round = {
        'figure-1': ['Layer', 'Probe-Acc', 'S-CDI'],
    }
    
    # best layer based, run neuron intervention module
    file_config_second_round = {        
        'figure-3': local_variables.figure_3_gyperparameter_threshold_CRDS,
        'figure-4': local_variables.figure_4_mutual_information,
        'table-3': local_variables.Table_3_probe_classification_performances_after_interven,
    }

    fieldnames_config_second_round = {
        'figure-3': ['Composition of CRDS', 'Acc_Difference', 'threshold_of_CRDS'],
        'figure-4': ['threshold_of_CRDS', 'Mutual_Matrix'], 
        'table-3': ['Interven', 'Acc', 'Sen', 'Spe', 'Precision', 'F1'], # best Layer
    }

    # best layer based, other tokens (around 'regret'), not 'regret'
    file_config_thrid_round = {    
    'table-2': local_variables.Table_2_previous_token_prediction_performances,
    }

    fieldnames_config_third_round = {
        'table-2': ['token_id', 'Acc', 'Sen', 'Spe', 'Precision', 'F1'], 
    }
    
    
    if local_variables.reload_data:
        shutil.rmtree(best_model_path)
        os.makedirs(best_model_path)
        shutil.rmtree(local_variables.mixdatapath)
        os.makedirs(local_variables.mixdatapath)
    
    if not os.path.exists(draw_files_path):
        os.makedirs(draw_files_path)
    
    if not os.path.exists(best_model_path):
        os.makedirs(best_model_path)
    
    all_q_lists_length = len(os.listdir(root_dir))
    indices = np.arange(all_q_lists_length)
    print(all_q_lists_length)
    np.random.seed(42)
    np.random.shuffle(indices)
    train_size = int(len(indices) * 0.7)
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    if training: 
        if os.path.exists(local_variables.task_type + '_record_layers_best_model_acc.txt'):
            os.remove(local_variables.task_type + '_record_layers_best_model_acc.txt')
        # for r in range(3):
        for layer in range(local_variables.total_layer_num):
            # if layer > 0:
            #     break

            best_acc = 0
            best_sen = 0
            best_spe = 0
            best_precision = 0
            best_f1 = 0
            all_hidden_states = None
            all_labels = None
            train_dataset = CustomDataset(indices = train_indices, type='train', specific_layer=layer)
            test_dataset = CustomDataset(indices = test_indices, type='test', specific_layer=layer)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
            model = MLPClassifier(input_dim=local_variables.input_dim, num_of_labels=config.data.num_of_labels).cuda()
            optim = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)
            loss_fun = torch.nn.CrossEntropyLoss()
            for epoch in range(EPOCH):    
                model.train()
                for hidden_states, label in train_loader:
                    print('[Epoch]: {}/{}'.format(epoch, EPOCH))
                    hidden_states = hidden_states.cuda()
                    label = label.cuda()

                    hidden_np = hidden_states.cpu().detach().numpy()  
                    labels_np = label.cpu().detach().numpy()

                    if epoch==0:                    
                        if all_hidden_states is None:
                            all_hidden_states = hidden_np
                            all_labels = labels_np
                        else:
                            all_hidden_states = np.concatenate([all_hidden_states, hidden_np], axis=0)
                            all_labels = np.concatenate([all_labels, labels_np], axis=0)

                    optim.zero_grad()
                    out = model(hidden_states)
                    loss = loss_fun(out, label.long())
                    loss.backward()
                    optim.step()
                    # Compute the metrics.
                    accuracy, sensitivity, specificity = calculate_metrics(out, label.long())
                    # Report them.
                    print(f"Train Loss: {loss.item()}")
                    print(f"[Train]Accuracy: {accuracy:.4f}, Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}\n")

                model.eval()
                accuracy, sensitivity, specificity, precision, f1 = test(model, test_loader)

                if sensitivity * specificity > best_sen * best_spe:
                    best_acc = accuracy
                    best_sen = sensitivity
                    best_spe = specificity
                    best_precision = precision
                    best_f1 = f1
                    model_path = os.path.join(best_model_path, f"layer_{layer}_best_model_epoch_acc.pth")
                    torch.save(model.state_dict(), model_path)
                print(f"[Best Test]Accuracy: {best_acc:.4f}, Sensitivity: {best_sen:.4f}, Specificity: {best_spe:.4f}\n")    
            
            # s_cdi, cdi, intra_compact, inter_ortho = compute_supervised_cdi(all_hidden_states, all_labels)
            #         'Layer': layer,
            #         'Probe-Acc': best_acc,
            #         'S-CDI': s_cdi
            # })
            
            # # skip first round
            # if r > 0:
                
            formatted_str = f"Layer: {layer}, [Best Test]Accuracy: {best_acc:.4f}, Sensitivity: {best_sen:.4f}, Specificity: {best_spe:.4f}, Precision: {best_precision:.4f}, F1: {best_f1:.4f}\n"
            with open(local_variables.task_type + '_record_layers_best_model_acc.txt', 'a', encoding='utf-8') as file:
                file.write(formatted_str)
                # Group ablation; `layer` is passed so the indices are saved.
                composional_control_results = composional_control(model, train_loader, test_loader, best_acc=best_acc, layer=layer)
                for k in range(len(composional_control_results)):
                    file.write(composional_control_results[k] + '\n')
                file.write('\n')
            file.close()
    else:
        model = MLPClassifier(input_dim=local_variables.input_dim, num_of_labels=config.data.num_of_labels).cuda()
        train_dataset = CustomDataset(indices = train_indices, type='train')
        test_dataset = CustomDataset(indices = test_indices, type='test')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
        model.load_state_dict(torch.load(os.path.join(best_model_path, f"best_model_epoch_acc.pth")))

if __name__ == "__main__":
    probing()