# Calibrated Typed Decisions

Can a small classifier learn **honest confidence** from a larger model's probabilities? When it says 80%, is it
right about 80% of the time, even on data it has never seen?

This folder has the code behind the technical report
[**Calibrated Typed Decisions: What a 7B Teacher's Soft Labels Do, and Don't, Teach a Small Classifier**](https://lagnajitmoharana.web.app/calibrated-typed-decisions-technical-report.pdf).
It builds a dataset of typed decisions with soft labels from a 7B teacher. It then trains ModernBERT students on six
different targets, tests them three ways, and publishes the best two.

| Artifact | Link |
|---|---|
| Technical report (PDF) | [calibrated-typed-decisions-technical-report.pdf](https://lagnajitmoharana.web.app/calibrated-typed-decisions-technical-report.pdf) |
| Dataset (v1.0, v2.0, v2.1; CC BY 4.0) | [m-lagnajit/calibrated-typed-decisions](https://huggingface.co/datasets/m-lagnajit/calibrated-typed-decisions) |
| Student F: for data like the training sources | [m-lagnajit/ctd-student-f](https://huggingface.co/m-lagnajit/ctd-student-f) |
| Student E: for new label sets of the same kind | [m-lagnajit/ctd-student-e](https://huggingface.co/m-lagnajit/ctd-student-e) |
| Everything in one place | [Hugging Face collection](https://huggingface.co/collections/m-lagnajit/calibrated-typed-decision) |
| Blog post, for a general audience | [How Sure Is "Sure"? Calibration, Uncertainty, and Knowledge in LLMs](https://lagnajitmoharana.web.app/how-sure-is-sure-understanding-llm-calibration) |

## The idea in one example

Every example is a **typed decision**: a text, a question and a list of options. It is either a *choice* question
(3 to 8 options) or a *yes/no* question about one candidate label (about 30% of questions).

```text
instructions: "Which banking intent best describes this customer message?"
state:        "I still haven't received my new card, what should I do?"
options:      card arrival · card delivery estimate · lost or stolen card · activate my card
gold:         card arrival
```

The **teacher** is Qwen2.5-7B-Instruct in 4-bit on a free Colab T4. It reads each question once. Instead of
generating an answer, we read the probability it gives each option's letter ("logit-slicing"). Those chances,
adjusted by the teacher's own dials, become the question's **soft label**. For GoEmotions, the soft label comes from
the human raters' votes instead.

The **students** are ModernBERT-base models (about 150M parameters). Each has one shared linear scorer that reads a
`[MASK]` marker placed in front of every option:

```text
[CLS] question [SEP] [MASK] option 1 [MASK] option 2 ... [MASK] option K [SEP] text [SEP]
```

Because the scorer is shared, the options can change from question to question. The student learns to judge how
well each option fits, rather than memorizing a fixed list of labels.

A **dial** is a temperature fitted after training on validation log loss. Yes/no questions also get a shift on the
"yes" score. Dials never change which option the model picks; they only make its confidence honest.

## What we found

1. **On familiar data, hard labels with one dial are as good as any soft target.** The raw soft label (C) is reliably
   the worst, because it copies the teacher's mistakes.
2. **On a new label set (SNIPS), the corrected teacher distribution (E) transfers best.** It beat every other target
   under both checks.
3. **On a new kind of judgment (PubMedQA), teacher-shaped targets make students more honest, not more knowledgeable.**
   The best ensemble (F) only just beats always predicting the label frequencies (log loss 0.899 against 0.937).
4. **A 2.6× larger student (ModernBERT-large) helps near transfer, not far transfer.** Full fine-tuning made F worse
   on PubMedQA. LoRA recovered part of the loss, but not reliably across seeds.

Every comparison uses three seeds and three-seed ensembles. A result counts as established only if it holds both
seed by seed and in a paired bootstrap. From Run F onward, each experiment's question and pass/fail rule were written
down before its results were seen.

### The six targets

`g` is the one-hot human answer, `s` the soft label, and `c` the corrected soft label. To correct a row, the
teacher's top pick swaps values with the right answer; this changes 1,391 of 13,996 teacher-labelled training rows.

| Run | Target | Idea |
|---|---|---|
| A | `g` | human answers only |
| B | `0.9 g + 0.1 uniform` | label smoothing |
| C | `s` | the teacher's (or raters') soft label |
| D | `0.5 g + 0.5 s` | blend of A and C |
| E | `c` | the corrected soft label |
| F | `0.5 g + 0.5 c` | blend of A and E |

### Ensemble log loss on the three tests (lower is better)

Values are for three-seed ensembles of ModernBERT-base students with 2 dials each (report, Appendix B). The best
score on each test is in bold.

| Run | Familiar data (test, 1,797) | New label set (SNIPS, 1,050) | New kind of judgment (PubMedQA, 1,000) |
|---|---|---|---|
| A | **0.221** | 0.143 | 0.964 |
| B | 0.249 | 0.153 | 0.948 |
| C | 0.292 | 0.143 | 0.932 |
| D | 0.231 | 0.128 | 0.928 |
| E | 0.241 | **0.114** | 0.926 |
| F | 0.225 | 0.126 | **0.899** |
| Teacher (Qwen2.5-7B, 4-bit) | 0.448 | not run | 2.325 (no dials) |
| Baseline | | 1.304 (guess evenly) | 0.937 (label frequencies) |

A and F are statistically tied on familiar data.

## Which student to use

| If the student will see… | Use | Why |
|---|---|---|
| data like the six training sources | [`ctd-student-f`](https://huggingface.co/m-lagnajit/ctd-student-f) | nothing was reliably better than F on familiar data |
| a new label set of the same kind | [`ctd-student-e`](https://huggingface.co/m-lagnajit/ctd-student-e) | best SNIPS ensemble; beat every other target under both checks |
| a new kind of judgment | F, but expect little | best PubMedQA log loss, barely below the label-frequency baseline |

Each model repository holds three students trained with seeds 0, 1 and 2, each with its own dials. Their chances are
averaged. Seeds 1 and 2 are retrained copies of the report's students, because only seed 0's original weights were
saved, so each model card's numbers differ slightly from the report.

## Quick start

```python
# pip install "transformers>=4.48" safetensors huggingface_hub torch numpy
import importlib.util
from huggingface_hub import hf_hub_download

path = hf_hub_download("m-lagnajit/ctd-student-f", "ctd_student.py")
spec = importlib.util.spec_from_file_location("ctd_student", path)
ctd_student = importlib.util.module_from_spec(spec); spec.loader.exec_module(ctd_student)

student = ctd_student.CalibratedStudent.from_pretrained("m-lagnajit/ctd-student-f")

# A choice question: options, ideally with a short description of each
student.predict_one(
    instructions="Which banking intent best describes this customer message?",
    text="I still haven't received my new card, what should I do?",
    options=["card arrival", "card delivery estimate", "lost or stolen card", "activate my card"],
    descriptions=["asking where an ordered card is", "asking how long delivery takes",
                  "reporting a lost or stolen card", "asking how to activate a card"])

# A yes/no question: put the candidate label in the question and leave the options empty
student.predict_one(
    instructions="Is the customer's intent in this message: card arrival (asking where an ordered card is)?",
    text="I still haven't received my new card, what should I do?", question_type="yes/no")
```

`predict_one` returns a dictionary of chances, one per option. To answer many questions at once, call
`student.predict(rows)`. Each row is a dict with `instructions`, `state`, `options`, `type` (`"choice"` or
`"yes/no"`), and optionally `option_descriptions`. Rows from the dataset can be passed in as they are.

From inside this folder, you can import the package directly with
`from ctd_student import CalibratedStudent`. [ctd_student.py](ctd_student.py) is the same file that ships with both
models.

## What's in this folder

The report's Appendix F uses working names for the notebooks; the second column maps them to the files here.

| File | Name in the report | What it does | GPU |
|---|---|---|---|
| [SystemOneModels.ipynb](SystemOneModels.ipynb) | – | Tutorial, shown on SmolLM-135M: turn a text generator into a classifier with one forward pass, by keeping only the logits of the option letters. The teacher scores every question the same way. | optional |
| [CalibratedTypedDecisionDatasetBuilder-v1.ipynb](CalibratedTypedDecisionDatasetBuilder-v1.ipynb) | `build_calibrated_typed_decisions.py` | Builds and publishes dataset **v1.0**: five sources, a license gate, teacher scoring, and one dial per source and type. | yes |
| [CalibratedTypedDecisionDatasetBuilder_v2.ipynb](CalibratedTypedDecisionDatasetBuilder_v2.ipynb) | `build_ctd_v2.ipynb`, `update_ctd_v21.ipynb` | First part: builds **v2.0** with GoEmotions rater votes, option descriptions, hard distractors, SciCite and the PubMedQA held-out split. Second part: adds the corrected soft label (**v2.1**) with arithmetic only, no GPU. | yes (v2.0) |
| [TrainACalibratedModel.ipynb](TrainACalibratedModel.ipynb) | `train_calibrated_student_v2.ipynb` | The student notebook. Cells 1–15: data, model and training. 16–22: single-run reports. 23: dial setups. 24: seeds. 25: ensembles. 26–28: SNIPS. 29: figures. 30: ModernBERT-large. 31: LoRA. | yes |
| [inspect_contractnli.ipynb](inspect_contractnli.ipynb), [inspect_cosmos_qa.ipynb](inspect_cosmos_qa.ipynb) | same | Pre-registered inspections of two candidate training sources, both rejected (report, Section 6). | no |
| [update_dataset_card.ipynb](update_dataset_card.ipynb) | `update_card_results.ipynb` | Regenerates the dataset card from the results files: the results section, plus the report, models and citation links. It refuses to publish if a claim on the card stops holding. | no |
| [publish_students.ipynb](publish_students.ipynb) | – | Packages students E and F with three seeds each, as 16-bit safetensors plus their dials. It checks that the package reproduces the training notebook's predictions, measures the published files on all three tests, writes the model cards and uploads them. | yes (recommended) |
| [test_published_students.ipynb](test_published_students.ipynb) | – | Loads both released models from the Hub and answers the example questions. | no |
| [ctd_student.py](ctd_student.py) | – | The small inference package shipped with the released students. | optional |
| [CITATION.cff](CITATION.cff) | – | Citation metadata for this code and the report. | – |
| [LICENSE](LICENSE) | – | Apache 2.0, for the code in this folder. | – |

## The dataset in brief

| Split | Questions | Notes |
|---|---|---|
| train | 16,995 | 3,000 per source (SciCite 2,000) |
| validation | 1,799 | dials are fitted here, never on held-out |
| test | 1,797 | familiar data |
| heldout | 1,000 | PubMedQA (PQA-L), expert-labelled; no student trains on it |

The six training sources are CLINC150, Banking77, MASSIVE (English), GoEmotions, Amazon Polarity and SciCite. Each was
checked against an allow-list of licenses at its original source. SNIPS 2017 was used for evaluation only and is
not distributed with the dataset. Four other candidate sources (WANLI, ContractNLI, COSMOS QA, HWU64) were rejected
by rules fixed before they were inspected.

Key columns: `instructions`, `state`, `options`, `option_descriptions`, `type`, `gold_index`, `teacher_probs`,
`soft_label`, `corrected_soft_label`, `soft_label_corrected`. Older versions stay available:
`load_dataset("m-lagnajit/calibrated-typed-decisions", revision="v1.0")`.

## Reproducing the results

Everything runs on Google Colab; a free T4 is enough. Results and weights are saved to Google Drive as they are
produced, so a disconnect loses little. The dataset builders and the student notebook have a `SMOKE` switch for a
short end-to-end trial: run once with `SMOKE = True`, then set it to `False` for real runs. The notebooks that upload
something (the card and publish notebooks) have a `PUBLISH` switch for a dry run first.

> The settings cell (Cell 4) of `TrainACalibratedModel.ipynb` is committed as it was for the last experiment
> (`SEED = 2`, `MODEL_SIZE = "large"`, `ADAPT = "lora"`). Set it back to the step you want before running.

| # | Run | Produces | GPU |
|---|---|---|---|
| 1 | `CalibratedTypedDecisionDatasetBuilder_v2.ipynb`, first part, with `SMOKE = False` | dataset v2.0 on the Hub | yes (teacher) |
| 2 | the same notebook, second part (v2.1) | dataset v2.1 (corrected labels) | no |
| 3 | `TrainACalibratedModel.ipynb`, Cells 1–15, for `SEED = 0, 1, 2` | the six base runs, three seeds each | yes |
| 4 | Cells 23, 24, 25 | dial setups, seed tables, ensembles | no |
| 5 | Cells 26 and 27; then `RERUN = True` with `SEED = 1` and `2`, Cells 1–13, 26, 14, 15; then Cell 28 | SNIPS for all three seeds | yes |
| 6 | `MODEL_SIZE = "large"`, `SEED = 0`–`2`, Cells 1–13, 26, 14, 15; then Cell 30 | large E and F | yes |
| 7 | also `ADAPT = "lora"`; then Cell 31 | large F with LoRA | yes |
| 8 | Cell 29; then `update_dataset_card.ipynb` | the figures; the dataset card | no |
| 9 | `publish_students.ipynb`, dry run first (`PUBLISH = False`); then `test_published_students.ipynb` | the two model repositories | yes (recommended) |

The training recipe is the same for every run: 3 epochs, batch 16, AdamW (weight decay 0.01), learning rates 3e-5 for
the body and 1e-3 for the scorer, 10% warmup, gradient clipping at 1.0, and mixed precision. The epoch with the lowest
validation log loss is kept. A base run takes about 20 minutes on a T4, and a large run about 40–45 minutes. The
report's Appendix D lists every setting.

## Limitations

- One teacher, one student family, English only, closed-set text decisions, and one far-transfer dataset.
- Dials are fitted on familiar data and can over- or under-correct elsewhere. On PubMedQA, A's dials make it
  underconfident.
- Seed-to-seed variation far from home (up to 7.6 accuracy points on PubMedQA) is larger than most of the differences
  being measured. Treat single-run results with suspicion.

The report's Section 9 discusses threats to validity in full.

## License

- Code in this folder: Apache 2.0 ([LICENSE](LICENSE)).
- Dataset: CC BY 4.0. Each source keeps its own license; the dataset card lists them.
- Models: Apache 2.0, the same as ModernBERT.

## Citation

If you use this work, please cite the technical report ([CITATION.cff](CITATION.cff) has the full metadata):

```bibtex
@techreport{moharana2026ctdreport,
  author      = {Moharana, Lagnajit},
  title       = {Calibrated Typed Decisions: What a 7B Teacher's Soft Labels Do, and Don't, Teach a Small Classifier},
  institution = {Independent},
  type        = {Technical report},
  year        = {2026},
  month       = nov,
  url         = {https://lagnajitmoharana.web.app/calibrated-typed-decisions-technical-report.pdf}
}
```
