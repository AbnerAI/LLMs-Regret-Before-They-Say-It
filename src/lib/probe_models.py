"""Probe classifier used for regret detection on LLM hidden states.

`MLPClassifier` is the probe behind every probing result reported in the paper
(Tables 1, 3, 4, 5 and Figures 5, 10). Its layer widths are part of the
experimental setup: changing them changes the reported numbers.

The optimiser settings are *not* defined here on purpose. The probe is trained
by `src/05_train_probe_rds_gic.py`, which owns the training loop and its
hyper-parameters (Adam, lr 1e-4, weight decay 0.01, 100 epochs, batch size 256,
70/30 split with numpy seed 42).
"""

import torch
import torch.nn.functional as F


class MLPClassifier(torch.nn.Module):
    """Four-layer MLP probe: input_dim -> 2048 -> 1024 -> 512 -> num_of_labels.

    Args:
        input_dim: hidden size of the probed layer (4096 / 5120 / 8192 for
            LLaMA-2 7B / 13B / 70B).
        num_of_labels: number of classes; 2 for the regret / non-regret task.
    """

    def __init__(self, input_dim, num_of_labels):
        super().__init__()
        self.layer1 = torch.nn.Linear(input_dim, 2048)
        self.layer2 = torch.nn.Linear(2048, 1024)
        self.layer3 = torch.nn.Linear(1024, 512)
        self.layer4 = torch.nn.Linear(512, num_of_labels)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        # The training loop applies CrossEntropyLoss on top of this softmax,
        # which internally applies log_softmax again. The double normalisation
        # is redundant but is exactly how the reported results were produced,
        # so it is kept deliberately. `dim=1` is explicit here and resolves to
        # the same axis the original implicit call used for 2-D probe outputs.
        return F.softmax(x, dim=1)
