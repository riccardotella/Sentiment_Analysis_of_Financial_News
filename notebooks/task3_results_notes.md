# Task 3 — Results notes (scratchpad)

Plots saved in: results/task3_outputs/

## Results — 3 Naive Bayes experiments

| Experiment | Accuracy | Macro-F1 | negative recall |
|---|---|---|---|
| NB + BoW (default) | 0.72 | 0.66 | 0.53 |
| NB + TF-IDF | 0.68 | 0.45 | 0.06 💥 |
| NB + BoW (uniform prior) | 0.68 | 0.64 | 0.63 ↑ |

## The story (and the uniform-prior prediction, now answered)

- **TF-IDF hurt** — collapsed minority classes (negative recall 0.06). Best avoided with NB.
- **Uniform prior worked as intended** — exactly the predicted trade-off:
  - negative recall rose 0.53 → 0.63 (model is braver about rare classes)
  - accuracy dropped 0.72 → 0.68 (it sacrifices some neutral wins)
  - per-class performance got more balanced (all three F1s now ~0.57–0.77)
  - Macro-F1 barely moved (0.66 → 0.64) because the gain on negatives offset the loss on neutral.
  - That balance is the point — the model stopped ignoring the minority class.

## Why TF-IDF failed with NB (mechanism, for the report)
TF-IDF down-weighting flattened the word-likelihood signal that Multinomial Naive Bayes
relies on, so predictions defaulted to the high-prior neutral class — improving neutral
recall but collapsing minority-class recall and lowering macro-F1 from 0.66 to 0.45.
Key point: "better" representation =/= better results; it depends on the classifier.
Accuracy (0.68) barely moved but macro-F1 cratered (0.45) -> macro-F1 is the honest metric.

## Confusion matrix takeaway (default BoW)
Errors are overwhelmingly confusions with the neutral majority class rather than polarity
flips -> the difficulty is separating weak sentiment from neutral reporting, not telling
positive from negative.

## Neural Network (MLPClassifier, hidden=(100,)) vs Naive Bayes

| Model | Accuracy | Macro-F1 | neg recall | pos recall |
|---|---|---|---|---|
| NB + BoW (default) | 0.72 | 0.66 | 0.53 | 0.55 |
| NB + BoW (uniform prior) | 0.68 | 0.64 | 0.63 | 0.58 |
| NN + BoW | 0.72 | 0.67 | 0.57 | 0.62 |

### The real finding
A neural network with ~876,000 parameters barely beat a model that just counts words
(macro-F1 0.67 vs 0.66 — essentially a tie). Why this matters:

- The NN's extra power (learning word combinations) bought almost nothing -> the task is
  largely linearly separable by word presence. For short financial headlines, knowing
  WHICH words appear ("loss", "profit", "growth") gets you most of the way; modeling their
  interactions adds little.
- The NN is slightly more balanced: it lifted the weaker classes (pos recall 0.55->0.62,
  neg 0.53->0.57) while holding accuracy at 0.72 — without the accuracy sacrifice the
  uniform-prior NB made.
- Overfitting risk (small data 3876 + huge features 8733 + ~876K params) did not blow up,
  but should be verified by comparing TRAIN vs TEST accuracy (big gap = overfitting). TODO.

### Report takeaway sentence
Despite ~876K parameters and the capacity to model word interactions, the feed-forward
network only matched Naive Bayes (macro-F1 0.67 vs 0.66), indicating that for short
financial headlines word presence alone is nearly sufficient and the added model
complexity yields little benefit while raising overfitting and compute costs.
