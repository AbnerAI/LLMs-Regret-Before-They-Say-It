"""On-disk container for the per-layer hidden states collected from a target LLM.

One instance holds the hidden states of a single example for every layer and
every answer stage, and appends them to per-layer tensor files so that the
probe training stage can load one layer at a time.
"""

import os

import torch


class RegretHiddenStates:
    """Buffer for the hidden states of one example across all layers.

    Args:
        num_of_labels: number of answer stages stored per example.
        layer_num: number of transformer layers (including the embedding
            output), e.g. 33 for LLaMA-2-7B.
        dimension: hidden size, e.g. 4096 for LLaMA-2-7B.
        tensor_root_path: directory the per-layer tensors are appended to.
        is_record_attention_and_mlp: also store attention and MLP sub-layer
            outputs, not just the layer output.
    """

    def __init__(self, num_of_labels, layer_num, dimension, tensor_root_path,
                 is_record_attention_and_mlp):
        self.mlp_states = torch.zeros((num_of_labels, layer_num, dimension))
        self.attention_states = torch.zeros((num_of_labels, layer_num, dimension))
        self.layer_outputs = torch.zeros((num_of_labels, layer_num, dimension))
        self.num_of_labels = num_of_labels
        self.layer_num = layer_num
        self.tensor_root_path = tensor_root_path
        self.is_record_attention_and_mlp = is_record_attention_and_mlp

    def set_mlp_states(self, mlp_states_per_layer, layer_idx, label_idx):
        self.mlp_states[label_idx][layer_idx] = mlp_states_per_layer

    def set_attention_states(self, attention_states_per_layer, layer_idx, label_idx):
        self.attention_states[label_idx][layer_idx] = attention_states_per_layer

    def set_layer_outputs(self, layer_outputs_per_layer, layer_idx, label_idx):
        self.layer_outputs[label_idx][layer_idx] = layer_outputs_per_layer

    def save_tensors(self, step_index):
        """Append the buffered states of every layer to disk."""
        if self.is_record_attention_and_mlp:
            self.save_file(self.mlp_states, "mlp_states", step_index)
            self.save_file(self.attention_states, "attention_states", step_index)
        self.save_file(self.layer_outputs, "layer_outputs", step_index)

    def save_file(self, states, position, step_index):
        for layer_idx in range(self.layer_num):
            tensor_save_path = os.path.join(
                self.tensor_root_path, "step_%s" % step_index,
                "%s_layer_%s.pt" % (position, layer_idx),
            )
            if not os.path.exists(tensor_save_path):
                torch.save(states[:, layer_idx], tensor_save_path)
            else:
                pre_states = torch.load(tensor_save_path)
                torch.save(torch.cat((pre_states, states[:, layer_idx]), 0), tensor_save_path)

    def save_tensors_last_layer(self, step_index):
        """Append only the final layer, used for the neuron-level analyses."""
        if self.is_record_attention_and_mlp:
            self.save_file_last_layer(self.mlp_states, "mlp_states", step_index)
            self.save_file_last_layer(self.attention_states, "attention_states", step_index)
        self.save_file_last_layer(self.layer_outputs, "layer_outputs", step_index)

    def save_file_last_layer(self, states, position, step_index):
        layer_idx = self.layer_num - 1
        tensor_save_path = os.path.join(
            self.tensor_root_path, "step_%s" % step_index,
            "%s_layer_%s.pt" % (position, layer_idx),
        )
        if not os.path.exists(tensor_save_path):
            torch.save(states[:, layer_idx], tensor_save_path)
        else:
            pre_states = torch.load(tensor_save_path)
            torch.save(torch.cat((pre_states, states[:, layer_idx]), 0), tensor_save_path)
