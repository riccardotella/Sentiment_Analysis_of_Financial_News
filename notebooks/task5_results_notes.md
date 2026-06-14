# Task 5 — Results notes (DistilBERT fine-tuning)

Model: distilbert-base-uncased, fine-tuned on RAW text (not preprocessed — transformer
uses its own subword tokenizer; cleaning would hurt because DistilBERT was pretrained on
natural English).

Setup: 80/20 stratified split (random_state=42, same as Task 3 for fair comparison),
max_length=128 (covers ~all sentences; EDA max length ~100 tokens), 3 epochs,
batch size 16, learning rate 2e-5 (small LR preserves pretrained knowledge), seed 42.
Ran on Google Colab (free T4 GPU); data pulled from raw GitHub URL.

Label mapping: 0=negative, 1=neutral, 2=positive (LabelEncoder, alphabetical).

## MULTICLASS RESULTS

              precision    recall  f1-score   support
    negative       0.76      0.86      0.81       121
     neutral       0.89      0.83      0.86       576
    positive       0.74      0.79      0.77       273
    accuracy                           0.82       970
    macro avg      0.80      0.83      0.81       970

Confusion matrix (rows=true, cols=pred; order neg/neut/pos):
    [[104  11   6]
     [ 26 479  71]
     [  6  50 217]]

## COMPARISON TO TASK 3 (multiclass)  -- THE REPORT CLIMAX

| Model | Accuracy | Macro-F1 | neg recall | pos recall |
|---|---|---|---|---|
| NB + BoW | 0.72 | 0.66 | 0.53 | 0.55 |
| NN + BoW | 0.72 | 0.67 | 0.57 | 0.62 |
| DistilBERT | 0.82 | 0.81 | 0.86 | 0.79 |

Massive jump: macro-F1 0.67 -> 0.81 (+0.14). The gain is concentrated EXACTLY where the
classical models failed:
- negative recall 0.53 -> 0.86 (+0.33): the rare class the classical models ignored.
- positive recall 0.55 -> 0.79 (+0.24): the class that kept collapsing into neutral.
- positive->neutral confusion (worst NB cell = 109) drops to 50.

## WHY (ties Task 3 error analysis to Task 5) — report narrative spine

DistilBERT fixes exactly the Task 3 failure categories:
| Task 3 failure | Why the transformer fixes it |
|---|---|
| negation ("acquitted ... conspiracy") | reads WORD ORDER -> "acquitted" reverses the negatives |
| domain knowledge ("won a contract") | PRETRAINED world knowledge -> knows business events are good |
| implicit/connotative ("new era") | CONTEXTUAL embeddings capture implied meaning |
| sentiment in numbers | sees the full sentence in context |

Report sentence:
"The transformer's +0.14 macro-F1 and dramatic recovery of minority-class recall
(negative 0.53->0.86) directly address the failure modes identified in the Task 3 error
analysis — negation, implicit sentiment, and domain knowledge — confirming that
contextual, pretrained representations capture sentiment that bag-of-words counting cannot."

## TODO
- Binary version (drop neutral, num_labels=2) + compare to Task 3 binary.
- Computational cost note for report: classical models train in seconds on CPU;
  DistilBERT needs a GPU (Colab T4) and ~minutes/epoch -> accuracy vs cost trade-off.
