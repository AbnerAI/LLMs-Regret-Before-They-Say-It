"""Stage 3 - run the four answer stages through the target LLaMA-2 model and
record the per-layer hidden states at the regret-keyword positions.

Requires a GPU. Outputs land under <REGRET_ROOT>/results/regret_<size>/.
"""
import os
import sys
import shutil
import torch
import argparse
import yaml
import json
import random
import numpy as np
import torch.nn as nn
from transformers import AutoTokenizer, LlamaForCausalLM

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
from lib.hidden_state_store import RegretHiddenStates

class RankUpdate(nn.Module):
    # lora_B is initialised to zeros and never trained, so `forward` returns its
    # input unchanged. The module is kept because it sits on the model's forward
    # path in the original code; `dtype` must follow the model's precision or the
    # matmul raises a type error.
    def __init__(self, embed_dim, rank=8, dtype=torch.float16):
        super().__init__()
        self.lora_A = nn.Parameter(torch.randn(embed_dim, rank, dtype=dtype) / rank**0.5)
        self.lora_B = nn.Parameter(torch.zeros(rank, embed_dim, dtype=dtype))
        self.rank = rank
        
    def forward(self, x):
        return x + (x @ self.lora_A @ self.lora_B)

class LlamaWithRankUpdate(LlamaForCausalLM):
    def __init__(self, config, rank=8):
        super().__init__(config)
        self.rank_update = RankUpdate(config.hidden_size, rank, dtype=self.dtype)
        for param in self.parameters():
            param.requires_grad = False
        for param in self.rank_update.parameters():
            param.requires_grad = True
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, rank=8, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, **kwargs)
        model.rank_update = RankUpdate(model.config.hidden_size, rank, dtype=model.dtype)
        # Freeze base parameters
        for param in model.parameters():
            param.requires_grad = False
        # Enable gradient for rank update
        for param in model.rank_update.parameters():
            param.requires_grad = True
        return model

    def forward(self, input_ids=None, inputs_embeds=None, attention_mask=None, **kwargs):
        if inputs_embeds is None and input_ids is not None:
            inputs_embeds = self.get_input_embeddings()(input_ids)
            inputs_embeds = self.rank_update(inputs_embeds)
            if attention_mask is not None:
                # keep the attention mask aligned with the new sequence length
                bsz, seq_len = input_ids.shape
                attention_mask = attention_mask[:, :seq_len]
                
        return super().forward(inputs_embeds=inputs_embeds, attention_mask=attention_mask, **kwargs)

    # Variant that updates only the probe tokens; kept for reference.
    def prepare_inputs_with_probe_just_probe_input(self, input_ids, probe_input_ids):
        original_embeddings = self.get_input_embeddings()(input_ids)
        probe_embeddings = self.get_input_embeddings()(probe_input_ids)
        updated_probe_embeddings = self.rank_update(probe_embeddings)
        return torch.cat([original_embeddings, updated_probe_embeddings], dim=1)
    
    # Variant that applies the low-rank update to the whole input.
    def prepare_inputs_with_probe_split(self, input_ids, probe_input_ids):
        original_embeddings = self.get_input_embeddings()(input_ids)
        probe_embeddings = self.get_input_embeddings()(probe_input_ids) 
        combined_embeddings = torch.cat([original_embeddings, probe_embeddings], dim=1)
        updated_embeddings = self.rank_update(combined_embeddings)
        return updated_embeddings
    
    # Variant actually used: the embeddings are returned unchanged.
    def prepare_inputs_with_probe(self, input_ids):
        input_embeddings = self.get_input_embeddings()(input_ids)
        return input_embeddings#updated_embeddings
    

def use_probing(text, model, max_generate_tokens):
    print('==============Question Start: {} [Question End]===============\n'.format(text))
    inputs = tokenizer(text, return_tensors="pt")
    combined_embeds = model.prepare_inputs_with_probe(inputs.input_ids.cuda())
    outputs = model.generate(
        inputs_embeds=combined_embeds.cuda(),
        max_new_tokens=max_generate_tokens,
        output_hidden_states=True,
        output_attentions=True,
        return_dict_in_generate=True
    )

    return outputs

class obj(object):
    def __init__(self, dict_):
        self.__dict__.update(dict_)

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO = os.path.dirname(_HERE)
sys.path.insert(0, _HERE)
ROOT = os.environ.get("REGRET_ROOT", _REPO)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--config_yaml', type=str,
                    default=os.path.join(_REPO, 'configs', 'llama2-7b.yaml'),
                    help='Model / environment config.')
parser.add_argument('--model_size', default='7b', choices=['7b', '13b', '70b'],
                    help='Target model scale; selects the output directory suffix.')
parser.add_argument('--limit', type=int, default=-1,
                    help='Process only the first N questions (-1 = all).')
parser.add_argument('--dtype', default='float16', choices=['float16', 'float32'],
                    help='Precision the target model runs in. Greedy decoding is '
                         'sensitive to this: fp16 and fp32 diverge after a near-tie.')
parser.add_argument('--save_question_input', action='store_true',
                    help='Also write the prompt-side hidden states. No released '
                         'stage reads them and they are the bulk of the output, '
                         'so this is off by default.')
parser.add_argument('--resume_from', type=int, default=1142,
                    help='With from_scratch disabled, skip question IDs below this.')
parser.add_argument('--fact_idx', type=int, default=5, help='The fact ids involved, -1 for all fact ids.')
parser.add_argument('--root_path', default=ROOT, type=str, help='Project root directory.')
parser.add_argument('--is_probing', type=bool, default=True, help='Whether to perform a probing task.')
parser.add_argument('--is_record_acc', type=bool, default=False, help='Whether the accuracy of the pilot experiment is recorded.')
parser.add_argument('--is_plot_heatmap', type=bool, default=True, help='Whether a heat map is required, for the case where fact_idx!=-1.')
parser.add_argument('--is_record_last_vi', type=bool, default=False, help='Whether to record the vi of the last token for plotting line graphs.')
parser.add_argument('--is_record_all_vi', type=bool, default=False, help='Whether to record the vi of all tokens for comparing entity tokens with non-entity tokens')
parser.add_argument('--num_of_irrelevant_evidence', type=int, default=0, help='Limited to the password task, detects the effect of the amount of irrelevant evidence on vi')
args = parser.parse_args()

from_scratch = True
target_model_size = '_' + args.model_size
target_model_name = 'llama-2-%s-' % args.model_size
data_path = os.path.join(ROOT, 'datasets', 'regret_dataset_with_id.json')
output_targer_model_answer = os.path.join(ROOT, 'datasets', 'generated_data_with_target_model_answer.json')
output_targer_model_answer_with_label_confident = os.path.join(ROOT, 'datasets', 'generated_data_with_target_model_answer_label_confident.json')
max_generate_tokens = 500
with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

# Simply change to deepspeed for multi-GPUs
os.environ['CUDA_VISIBLE_DEVICES']=str(config.environment.cuda_visible_devices[0])
using_probing = True

with open(args.config_yaml) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
config = json.loads(json.dumps(config), object_hook=obj)

tokenizer = AutoTokenizer.from_pretrained(config.plm.model_path, use_fast=False)
model = LlamaWithRankUpdate.from_pretrained(config.plm.model_path, rank=8, torch_dtype=getattr(torch, args.dtype)).cuda()

def mlp_hook(module, input, output):
    mlp_outputs.append(output)

def attention_hook(module, input, output):
    attention_outputs.append(output)

def layer_outputs_hook(module, input, output):
    layer_outputs_outputs.append(output)
    # if len(layer_outputs_outputs[0][0][0]) == 1:

    # if len(layer_outputs_outputs[0][0][0]) > 2:


# def layer_outputs_hook(module, input, output):
#     if len(layer_outputs_by_layer[0]) >= 52:
#     for i, layer in enumerate(model.model.layers):
#         if module == layer:
#             break    

# def mlp_hook(module, input, output):
#     for i, layer in enumerate(model.model.layers):
#         if module == layer.mlp:
#             break

# def attention_hook(module, input, output):
#     for i, layer in enumerate(model.model.layers):
#         if module == layer.self_attn:
#             break



    

# def format_prompt(context, question):
#     """
#     """
#     B_INST, E_INST = "[INST]", "[/INST]"
#     B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
#     BOS, EOS = "<s>", "</s>"
    
#         f"{BOS}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{question} {E_INST}"
#     )
#     return formatted_prompt

def format_prompt(question):
    """
    Wrap a question in the LLaMA-2 chat prompt format.
    """
    B_INST, E_INST = "[INST]", "[/INST]"
    B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
    BOS, EOS = "<s>", "</s>"
    DEFAULT_SYSTEM_PROMPT = "You are a helpful, respectful and honest assistant. "
    
    # Combine the system prompt with the user question.
    formatted_prompt = (
        f"{BOS}{B_INST} {B_SYS}{DEFAULT_SYSTEM_PROMPT}{E_SYS}{question} {E_INST}"
    )
    return formatted_prompt


for i in range(model.config.num_hidden_layers):
    if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
        model.model.layers[i].mlp.register_forward_hook(mlp_hook)
        model.model.layers[i].self_attn.register_forward_hook(attention_hook)
    model.model.layers[i].register_forward_hook(layer_outputs_hook)

if os.path.exists(output_targer_model_answer) and from_scratch:
    os.remove(output_targer_model_answer)

def get_surrounding_positions(lst, window=5):
    # Positions flagged with 1 in the keyword mask.
    one_positions = [i for i, val in enumerate(lst) if val == 1]
    
    # Collect every position we need to keep.
    result_positions = set()
    
    # For each flagged position, also take the `window` tokens on each side.
    for pos in one_positions:
        # the flagged position itself
        result_positions.add(pos)
        
        # preceding positions, clipped at the start
        for i in range(max(0, pos - window), pos):
            result_positions.add(i)
        
        # following positions, clipped at the end
        for i in range(pos + 1, min(len(lst), pos + window + 1)):
            result_positions.add(i)
    
    # Return them in order.
    return sorted(list(result_positions))

processed = 0
with open(data_path, 'r', encoding="utf-8") as file:
    for line in file:
        if args.limit >= 0 and processed >= args.limit:
            break
        entry = json.loads(line)
        question_id = entry["ID"]
        if not from_scratch:
            if question_id < args.resume_from:
                continue
        processed += 1
        fake_evidence = entry["fake_evidence"]
        question = entry["question"]
        ground_truth = entry["ground_truth"]
        weak_hint_true = entry["weak_hint_true"]
        strong_hint_true = entry["strong_hint_true"]
        
        pure_prompt = f"Question: {question}\n"
        pure_prompt = format_prompt(pure_prompt)
        # pure_answer, pure_answer_confident = generate_response(pure_prompt)
        
        fake_evidence_prompt = f"Based on this context: {fake_evidence}\n\n The question is: {question}"
        initial_prompt = format_prompt(fake_evidence_prompt)
        # initial_answer_by_target_model, initial_answer_by_target_model_confident = generate_response(initial_prompt)
        
        weak_prompt = f"""
        You previously answered the following question: {question}
        After considering the following hint: {fake_evidence}
        Your revised answer was: [initial_answer_by_target_model]
        Now, consider this information: {weak_hint_true}
        Which previous answer do you regret? What's the final answer? 
        """
        weak_prompt  = format_prompt(weak_prompt)
        # reflection_answer_weak, reflection_answer_weak_confident = generate_response(weak_prompt)
        
        strong_prompt = f"""
        You previously answered the following question: {question}
        After considering the following hint: {fake_evidence}
        Your initial answer was: [initial_answer_by_target_model]
        After considering the following hint: {weak_hint_true}
        Your revised answer was: [reflection_answer_weak]
        Now, consider this information: {strong_hint_true}
        Which previous answer do you regret? What's the final answer? 
        """
        strong_prompt  = format_prompt(strong_prompt)
        # reflection_answer_strong, reflection_answer_strong_confident = generate_response(strong_prompt)

        prompt_list = [pure_prompt, initial_prompt, weak_prompt, strong_prompt]

        pure_prompt_tokenized = tokenizer.tokenize(pure_prompt)
        initial_prompt_tokenized = tokenizer.tokenize(initial_prompt)
        weak_prompt_tokenized = tokenizer.tokenize(weak_prompt)
        strong_prompt_tokenized = tokenizer.tokenize(strong_prompt)

        question_max_length = max(len(pure_prompt_tokenized), len(initial_prompt_tokenized), len(weak_prompt_tokenized), len(strong_prompt_tokenized))

        if not args.is_record_last_vi and not args.num_of_irrelevant_evidence:
            is_record_attention_and_mlp = True
        else:
            is_record_attention_and_mlp = False

        # [question_length, labels, layers, hidden_size]
        hidden_states_save_basepath = os.path.join(ROOT, 'results', 'regret' + target_model_size)
        tensor_root_llm_res_path = os.path.join(hidden_states_save_basepath, 'Q_ID_' + str(question_id), 'LLM_responses')
        tensor_root_path = os.path.join(hidden_states_save_basepath, 'Q_ID_' + str(question_id), 'Question_input')
        if not os.path.exists(tensor_root_path):
            os.makedirs(tensor_root_path)
            os.makedirs(tensor_root_llm_res_path)
        
        MAX_SEQUENCE_LENGTH = question_max_length + max_generate_tokens
        hidden_states_list = [RegretHiddenStates(config.data.num_of_labels, model.config.num_hidden_layers, model.config.hidden_size, tensor_root_path, is_record_attention_and_mlp) 
                            for step_index in range(MAX_SEQUENCE_LENGTH)] # the answer stages for one question
        max_input_token_length = 0
        all_answer = []
        all_entity_tag_list = dict() # input
        all_entity_tag_llm_res_list = dict() # input
        for label_idx, prompt in enumerate(prompt_list):
            if label_idx == 2:
                prompt =  prompt.replace("[initial_answer_by_target_model]", all_answer[1])
            elif label_idx == 3:
                prompt =  prompt.replace("[initial_answer_by_target_model]", all_answer[1])
                prompt =  prompt.replace("[[reflection_answer_weak]]", all_answer[2])

            # hook
            attention_outputs = []
            mlp_outputs = []
            layer_outputs_outputs = []
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            input_length = len(inputs.input_ids[0])
            # Decoded tokens: 
            # ['<s>', '<s>', '[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful', ',', '▁respect', 'ful', '▁and', '▁honest', '▁assistant', '.', '▁', '<0x0A>', '<', '</', 'SY', 'S', '>>', '<0x0A>', '<0x0A>', 'Question', ':', '▁Are', '▁more', '▁people', '▁today', '▁related', '▁to', '▁G', 'eng', 'his', '▁Khan', '▁than', '▁Julius', '▁Ca', 'esar', '?', '<0x0A>', '▁[', '/', 'INST', ']']
            # ['<s>', '[', 'INST', ']', '▁<<', 'SY', 'S', '>>', '<0x0A>', 'You', '▁are', '▁a', '▁helpful', ',', '▁respect', 'ful', '▁and', '▁honest', '▁assistant', '.', '▁', '<0x0A>', '<', '</', 'SY', 'S', '>>', '<0x0A>', '<0x0A>', 'Question', ':', '▁Are', '▁more', '▁people', '▁today', '▁related', '▁to', '▁G', 'eng', 'his', '▁Khan', '▁than', '▁Julius', '▁Ca', 'esar', '?', '<0x0A>', '▁[', '/', 'INST', ']']
            # 52
            # 51

            # get index by key entity name
            # load key position entity name
            key_emo_entity_path = os.path.join(ROOT, 'datasets', 'key_position.json')
            with open(key_emo_entity_path, "r") as f:
                key_emo_entity_lists = json.load(f)
            key_emo_entity_mild = key_emo_entity_lists['Mild']
            key_emo_entity_moderate = key_emo_entity_lists['Moderate']
            key_emo_entity_severe = key_emo_entity_lists['Severe']
            combine_key_emo_entity = key_emo_entity_mild + key_emo_entity_moderate + key_emo_entity_severe
            # Step 1: Tokenize all entities
            entity_token_sequences = [tokenizer.tokenize(entity) for entity in combine_key_emo_entity]
            # Step 2: Tokenize the prompt
            prompt_tokens = tokenizer.tokenize(prompt)
            # Step 3: Initialize the entity tag list
            entity_tag_list = [0] * len(prompt_tokens)
            # Step 4: Match entity sequences in the prompt tokens
            for entity_sequence in entity_token_sequences:
                sequence_length = len(entity_sequence)
                for i in range(len(prompt_tokens) - sequence_length + 1):
                    # Extract the current window of tokens
                    current_sequence = prompt_tokens[i:i + sequence_length]
                    # Check if the current window matches the entity sequence
                    if current_sequence == entity_sequence:
                        # Mark the matched positions as 1
                        entity_tag_list[i:i + sequence_length] = [1] * sequence_length
            
            # Print the results
            all_entity_tag_list[label_idx] = entity_tag_list

            if False: #using_probing:
                # If is LLM response (Probing type), The recommendation is to go through the decoder before entering the following process detection. 
                text = prompt # or LLM Output Text
                outputs = use_probing(text, model, max_generate_tokens) 
                generated_tokens = outputs.sequences[0]#[0][0]#[input_length:]
            else:  
                print('==============Question Start: {} [Question End]===============\n'.format(prompt))
                inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
                # Start hooks
                # max_new_tokens must be larger
                outputs = model.generate(inputs.input_ids.cuda(), max_new_tokens = max_generate_tokens, output_hidden_states = True, output_attentions = True, return_dict_in_generate = True)
                # End hooks 
                print(label_idx)
                generated_tokens = outputs[0][0][input_length:]
                # Exclude the last token if it's the EOS token
                if generated_tokens[-1] == 2:  # Assuming 2 is your EOS token ID
                    generated_tokens = generated_tokens[:-1]

            model_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True) 
            all_answer.append(model_answer)
            print('======================== [model_answer Start]: {} [model_answer End]========================  \n'.format(model_answer))
            print('************************ [ground_truth Start]: {} [ground_truth End]************************ \n'.format(ground_truth))

            # get llm response mask 
            llm_res_tokens = tokenizer.tokenize(model_answer)
            llm_res_entity_tag_list = [0] * len(llm_res_tokens)
            # Step 4: Match entity sequences in the prompt tokens
            for entity_sequence in entity_token_sequences:
                sequence_length = len(entity_sequence)
                for i in range(len(llm_res_tokens) - sequence_length + 1):
                    # Extract the current window of tokens
                    current_sequence = llm_res_tokens[i:i + sequence_length]
                    # Check if the current window matches the entity sequence
                    if current_sequence == entity_sequence:
                        # Mark the matched positions as 1
                        llm_res_entity_tag_list[i:i + sequence_length] = [1] * sequence_length
            
            # Print the results
            all_entity_tag_llm_res_list[label_idx] = llm_res_entity_tag_list
            # end
            
            # Get the generated token length
            generated_token_length = len(generated_tokens)
            # Total token length 
            cur_input_token_length = input_length# + generated_token_length
            if max_input_token_length < cur_input_token_length:
                max_input_token_length = cur_input_token_length
            
            # record the token length of each answer stage so it can be read back later
            # Need to record the different labels (actually the token lengths of the replies from different periods of time, for subsequent reading)
            if args.is_probing:
                
                # Establish mapping relationship
                llm_token_data = []
                llm_token_surround = dict()
                surround_pos = get_surrounding_positions(llm_res_entity_tag_list, window = 5)
                # ==================================== llm responses ====================================
                for token_index, token_id in enumerate(generated_tokens[:len(generated_tokens) - 1]):
                    token_text = tokenizer.decode(token_id, skip_special_tokens=True)
                    hidden_states = {}
                    # all multi layer
                    layer_hidden_states = outputs.hidden_states[token_index]
                    # after & before tokens
                    if len(surround_pos) > 0 and token_index in surround_pos and llm_res_entity_tag_list[token_index]!=1:
                        llm_token_surround.update({token_index: layer_hidden_states})
                    # keyword tokens plus the final token, not all tokens
                    if llm_res_entity_tag_list[token_index]!=1 and token_index < len(generated_tokens) - 2:
                        continue
                    # layer_hidden_states[f"layer_{-1}"] = hidden_state.tolist()
                    llm_token_data.append({
                        "question_id": question_id, 
                        "index": token_index,  # token index
                        "token_id": token_id.item(),  # token ID
                        "token_text": token_text,  # token surface form
                        "hidden_states": layer_hidden_states
                    })
                
                np.save(os.path.join(tensor_root_llm_res_path, 'Response_' + str(label_idx) + '_llm_response_answer.npy'), np.array(model_answer))
                torch.save(llm_token_data, os.path.join(tensor_root_llm_res_path, 'Response_' + str(label_idx) + '_token_data_llm_response_hidden_state.pt'))
                torch.save(llm_token_surround, os.path.join(tensor_root_llm_res_path, 'Response_' + str(label_idx) + '_token_surround_llm_response_hidden_state.pt'))
                # last layer
                token_data = []
                decoded_text = []
                
                # ======================================== input ========================================= 
                # the hook only stores hidden states for the input span
                input_token_surround = dict()
                input_surround_pos = get_surrounding_positions(entity_tag_list, window = 3)
                for token_index, token_id in enumerate(inputs.input_ids[0]):
                    #     continue
                    token_text = tokenizer.decode(token_id, skip_special_tokens=True)
                    all_attention_hidden_states = []
                    all_layer_outputs_hidden_states = []
                    all_mlp_hidden_states = []
                    for layer_idx in range(model.config.num_hidden_layers):
                        # all multi layer
                        layer_hidden_states_attention_outputs = torch.Tensor(attention_outputs[layer_idx][0][0][-token_index-1])#outputs.hidden_states[-1]
                        hidden_state = layer_hidden_states_attention_outputs.squeeze()
                        all_attention_hidden_states.append(hidden_state.tolist())
                        
                        layer_hidden_states_layer_outputs = torch.Tensor(layer_outputs_outputs[layer_idx][0][0][-token_index-1])
                        hidden_state = layer_hidden_states_layer_outputs.squeeze()
                        all_layer_outputs_hidden_states.append(hidden_state.tolist())
                        # layer_outputs_hidden_states[f"layer_{-1}"] = hidden_state.tolist() 

                        layer_hidden_states_mlp_outputs = torch.Tensor(mlp_outputs[layer_idx][0][-token_index-1])
                        hidden_state = layer_hidden_states_mlp_outputs.squeeze()
                        all_mlp_hidden_states.append(hidden_state.tolist())
                        # mlp_hidden_states[f"layer_{-1}"] = hidden_state.tolist()  
                    
                    # after & before tokens
                    if len(input_surround_pos) > 0 and token_index in input_surround_pos and entity_tag_list[token_index]!=1:
                        input_token_surround.update({token_index: [all_attention_hidden_states,all_layer_outputs_hidden_states,all_mlp_hidden_states]})

                    if entity_tag_list[token_index] !=1 and token_index < len(inputs.input_ids[0]) - 1:
                        continue

                    token_data.append({
                        "question_id": question_id, 
                        "index": token_index,  # token index
                        "token_id": token_id.item(),  # token ID
                        "token_text": token_text,  # token surface form
                        "attention_hidden_states": all_attention_hidden_states,
                        "layer_outputs_hidden_states": all_layer_outputs_hidden_states,
                        "mlp_hidden_states": all_mlp_hidden_states, 
                    })

                np.save(os.path.join(tensor_root_path, 'Response_' + str(label_idx) + '_input_prompt.npy'), np.array(prompt))
                # The prompt-side ("Question_input") tensors are not read by any
                # released stage: stage 5 builds the probe dataset entirely from
                # the LLM_responses tensors. They are ~60% of the extraction
                # output, so writing them is off by default.
                if args.save_question_input:
                    torch.save(token_data, os.path.join(tensor_root_path, 'Response_' + str(label_idx) + '_token_data_input_hidden_state.pt'))
                    torch.save(input_token_surround, os.path.join(tensor_root_path, 'Response_' + str(label_idx) + '_token_data_input_surround_hidden_state.pt'))
        
        all_entity_mask_list = {'input': all_entity_tag_list ,'llm_res': all_entity_tag_llm_res_list}
        updated_entry = {
        "ID": question_id,
        "emotion_mask": all_entity_mask_list,
        "question": question,
        target_model_name + "pure_answer": all_answer[0],
        "fake_evidence": fake_evidence,
        "ground_truth": ground_truth,
        target_model_name + "initial-answer_with_fake_evidence": all_answer[1],
        "weak_hint_true": weak_hint_true,
        target_model_name + "reflection_answer_weak": all_answer[2],
        "strong_hint_true": strong_hint_true,
        target_model_name + "reflection_answer_strong": all_answer[3],
        }
        with open(output_targer_model_answer, "a", encoding="utf-8") as f:
            json.dump(updated_entry, f, ensure_ascii=False)
            f.write("\n") 
