# LLMs Regret Before They Say It

Official code and data for **"LLMs Regret Before They Say It: Early Detection and
Compositional Architecture of Regret in Hidden States"** (EMNLP 2026).

We study how large language models encode *regret* — the explicit acknowledgement
of a previous answer being wrong once contradicting evidence appears. The
repository contains the regret dataset, the probing pipeline, and the
neuron-level analyses (RDS, GIC) behind the reported results.

---

## What is here

```
configs/
  llama2-7b.yaml  llama2-13b.yaml  llama2-70b.yaml
data/
  conflictQA-strategyQA-gpt4.json   source questions with parametric / counter memory
  regret_dataset.json               the regret dataset (1,356 items, four answer stages)
  key_position.json                 97 regret keywords in three severity bands
src/
  01_build_regret_dataset.py        three-stage prompting -> regret dataset
  02_assign_ids.py                  attach a stable ID to every record
  03_extract_hidden_states.py       run the answer stages through the target LLM,
                                    record per-layer hidden states at keyword positions
  04_label_regret_gpt4.py           GPT-4 regret labels (these are the probe labels)
  05_train_probe_rds_gic.py         layer-wise probes, RDS, group ablations, MI
  lib/
    probe_models.py                 the MLP probe
    metrics.py                      RDS, neuron categorisation, GIC, group mutual information
    hidden_state_store.py           per-layer hidden-state container used by stage 3
requirements.txt
```

Model weights and extracted hidden states are **not** distributed: extracting
LLaMA-2-7B over the full dataset produces on the order of 125 GB. Stage 3
regenerates them from the released dataset on a single GPU.

## Data

`regret_dataset.json` is JSON Lines, one object per question, 1,356 items. It is
built from `conflictQA-strategyQA-gpt4.json`, the StrategyQA split of ConflictQA
(Xie et al., 2023).

Each record carries the four answer stages of the workflow in Figure 2(A):

| field | meaning |
| --- | --- |
| `question` | the original StrategyQA question |
| `ground_truth` | `True` / `False`, the correct answer |
| `Pure_answer` | a0, answered with no evidence supplied |
| `fake_evidence` | strengthened misleading evidence written by GPT-4o |
| `gpt-4o-mini-initial_answer_with_fake_evidence` | a1, answered under the fake evidence |
| `weak_hint_true` | an indirect hint that invites reconsideration |
| `gpt-4o-mini-reflection_answer_weak` | a2, answered after the weak hint |
| `strong_hint_true` | the true evidence (parametric memory) |
| `gpt-4o-mini-reflection_answer_strong` | a3, answered after the true evidence |
| `*_confidence_score_*` | the model's self-reported confidence, 1–10 |
| `pure_ground_truth`, `*_ground_truth:` | GPT-4o-mini's judgement of whether that answer expresses regret |

Positive probe samples (label 1) are hidden states at the position of an explicit
regret expression in a2 and a3; negative samples (label 0) are the equivalent
positions in a1, where no regret is expressed.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export REGRET_ROOT=/path/to/experiment/tree   # holds datasets/ and results/
```

Stages 1 and 4 call the OpenAI API and need `OPENAI_API_KEY` in the environment.
No key is stored in this repository.

## Pipeline

**Stage 1 — build the dataset.** Only needed to regenerate the data from
scratch; the released dataset is the output of this step.

```bash
export OPENAI_API_KEY=...
python src/01_build_regret_dataset.py \
    --input data/conflictQA-strategyQA-gpt4.json \
    --output data/regret_dataset.json
```

**Stage 2 — assign IDs.** The ID ties each record to its extracted hidden
states, so it is assigned once and never reshuffled.

```bash
python src/02_assign_ids.py \
    --input data/regret_dataset.json \
    --output data/regret_dataset_with_id.json
```

**Stage 3 — extract hidden states.** Runs the four answer stages through the
target model, locates the regret keywords from `key_position.json` in each
answer, and stores the per-layer hidden states at those positions together with
a +/-5 token window around them (the window is what the early-detection analysis
uses). Requires a GPU; point `configs/llama2-7b.yaml` at your local copy of the
model first.

```bash
cp data/regret_dataset_with_id.json data/key_position.json "$REGRET_ROOT/datasets/"
python src/03_extract_hidden_states.py \
    --config_yaml configs/llama2-7b.yaml --model_size 7b
```

Use `--limit N` to process only the first N questions.

**Stage 4 — regret labels.** GPT-4 judges, for each answer stage, whether the
answer expresses regret. The resulting `GPT-4-ground_truth_*` fields are the
probe labels.

```bash
export OPENAI_API_KEY=...
python src/04_label_regret_gpt4.py --model_size 7b
```

**Stage 5 — probes, RDS and group ablations.** Trains one probe per layer,
computes the Regret Dominance Score on the final layer, splits the neurons into
RegretD / Non-RegretD / DualD, and measures probe accuracy after deactivating
each group and each combination.

```bash
python src/05_train_probe_rds_gic.py --model_size 7b
```

`--model_size` also selects the layer count and hidden size
(33/4096 for 7B, 41/5120 for 13B, 81/8192 for 70B).

## Reproduction settings

These are the settings that produced the reported numbers. Changing them
changes the results.

| setting | value |
| --- | --- |
| probe | MLP, `hidden -> 2048 -> 1024 -> 512 -> 2`, ReLU |
| optimiser | Adam, lr `1e-4`, weight decay `0.01` |
| epochs / batch size | 100 / 256 |
| train / test split | 70 / 30, `numpy` seed 42 |
| probed representation | `layer_outputs_hidden_states` |
| neuron deactivation | activation set to `-1` |
| RDS threshold `tau` | swept over `0.01 … 0.50` |
| hardware | 2x NVIDIA L20 (48 GB), PyTorch 1.12 |

The pipeline was re-verified for this release on a single RTX 4090 with Python
3.8.20 and torch 2.4.1; stages 2, 3 and 5 were run end to end there. Stages 1
and 4 call the OpenAI API and have not been executed since the release was
assembled.

Reproducibility caveats, stated as the code behaves:

* The 70/30 train/test split is seeded (`numpy` seed 42) and is stable across
  runs.
* Probe weights are initialised without a torch seed, so probe training is not
  deterministic; layer accuracies vary slightly between runs.
* The random control groups in the ablation table are drawn with Python's
  `random.sample`, which is never seeded, so those control rows differ between
  runs.

Setting `torch.manual_seed` and `random.seed` would make both deterministic, but
it would also change the numbers away from the ones that were reported, so the
code is left as it was run.

**Metrics.** `src/lib/metrics.py` is a reference implementation of

- **RDS** (Eq. 1) — per-neuron regret dominance, the ratio of the mean regret
  activation to the summed mean activation of both states.
- **Neuron categorisation** (Eq. 2) — `mu ± tau * sigma` over the RDS
  distribution, giving the disjoint RegretD / Non-RegretD / DualD sets.
- **GIC** (Eq. 3) — `Acc(Z - S1) / Acc(Z)` for a single group, and
  `Acc(Z - union(S_i)) / mean_i Acc(Z - S_i)` for a combination. `GIC < 1` means
  the groups act compositionally.
- **Group mutual information** — normalised MI between the mean activations of
  two groups, discretised into 20 bins.

Stage 5 executes its own inline copies of RDS, the categorisation and the mutual
information; they were checked to produce element-wise identical output to the
functions in `metrics.py`. GIC is the exception: stage 5 records the accuracy
after each ablation, and `group_impact_coefficient` turns those numbers into
Eq. 3.

## Intended use

The dataset contains **deliberately fabricated evidence**: for every question,
GPT-4o was asked to write, and then strengthen, misleading evidence supporting
the wrong answer. It exists so that a model can be put into a state where it
later has to retract, and so that the hidden states of that retraction can be
studied.

It is released for interpretability research only. Do not use the fabricated
evidence as training data for generative models, and do not present any of it as
factual. The questions and their true answers come from StrategyQA via
ConflictQA and are unmodified.

## Status

This is the initial release. It covers dataset construction, hidden-state
extraction, GPT-4 regret labelling, and the probing / RDS / GIC analysis.

Planned for a follow-up release:

* anchor-guided gradient attribution and the generation-time neuron intervention
  (Section 4.3, Figure 4);
* the preceding-token early-detection analysis (Table 1, Figure 5);
* the cross-keyword generalisation analysis (Table 2).

The probe-level group ablations behind the RDS and GIC results are already
included, in stage 5.

**On exact reproduction.** The target model runs in fp16 with greedy decoding,
which is sensitive to the torch / transformers / GPU combination. Re-running
stage 3 on a different stack produces different generated text, and therefore
numbers that differ from the reported ones. The code released here is the
implementation that produced the reported results; the reported values
themselves come from the original run described in Appendix B (2x NVIDIA L20,
PyTorch 1.12).

## Extraction output size

Stage 3 records, per question and per answer stage, the hidden states at the
recorded token positions plus a window around them. Measured on 150 questions
with LLaMA-2-7B:

| file | 150 questions | read by stage 5 |
| --- | --- | --- |
| `LLM_responses/*_token_data_llm_response_hidden_state.pt` | 0.6 GB | yes |
| `LLM_responses/*_token_surround_llm_response_hidden_state.pt` | 13.0 GB | no; input for the preceding-token analysis, which is not part of this release |
| `Question_input/*` | 22.1 GB | no |

`Question_input` is therefore not written unless `--save_question_input` is
passed. Measured over the same 150 questions, the default settings produce
92 MB per question, so expect roughly 125 GB for the full 1,356 questions on
7B (about 330 GB if `--save_question_input` is enabled).

## Numerical precision

Stage 3 runs the target model in fp16, so the stored activations are half
precision, while the probe parameters are float32. Stage 5 widens the stored
activations to float32 when loading them. This widening is exact -- every fp16
value is representable in fp32 -- so no recorded activation is altered.

Note that fp16 inference itself is not bit-identical to fp32 inference, so
activations extracted with this pipeline may differ in the last digits from a
full-precision run.

## Notes on this release

- The pipeline was originally built on top of the layer-wise probing code of
  Ju et al. (2024). Only the regret-specific code is distributed here; inherited
  components that no reported result depends on have been removed.
- The output CSV of stage 5 labels its first column `Group`.

## Citation

```bibtex
@inproceedings{cui2026regret,
  title     = {LLMs Regret Before They Say It: Early Detection and Compositional
               Architecture of Regret in Hidden States},
  author    = {Cui, Xiangxiang and Yang, Shu and Huang, Tianjin and Lin, Wanyu
               and Hu, Lijie and Wang, Di},
  booktitle = {Proceedings of EMNLP},
  year      = {2026}
}
```

## License

This code is released under the MIT license, see [LICENSE](LICENSE).

`data/conflictQA-strategyQA-gpt4.json` is redistributed from
[ConflictQA](https://huggingface.co/datasets/osunlp/ConflictQA) (Xie et al.,
2023), which is Apache-2.0 licensed.
