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

## ERROR ANALYSIS (NN + BoW, 267 errors / 970)

Focused on the worst confusion-matrix cell: positive -> predicted neutral.
Read the actual misclassified sentences. They share one thing: the "positive"
sentences contain almost NO positive WORDS. The positivity lives in events, numbers,
connotation, or negation — none of which bag-of-words can see.

### Error categories (with examples)

| Category | Example (cleaned text) | Why BoW/NN fails |
|---|---|---|
| Sentiment in numbers/events | "subscriptions increase cargotec s share capital 36 780 euros"; "finnish cargotec awarded significant order total 292 hiab loader cranes" | the signal is the financial EVENT (capital increase, contract won), not a sentiment word |
| Implicit / connotative | "nordea moving new headquarters signifies beginning new era"; "staff recruited japan complement network close 50 service locations" | positive by implication/expansion; no positive keywords |
| Negation / polarity reversal | "stora enso ... ACQUITTED charges participated paper price fixing CONSPIRACY" | sentence is full of negative words ("charges","conspiracy"); "acquitted" reverses them, but BoW ignores word order |
| Domain knowledge required | "company featured ethibel pioneer investment register ... sustainable corporations" | being listed = an honor; needs world knowledge |

Best single example for the report: the "acquitted ... conspiracy" sentence — a textbook
case of negation defeating a bag-of-words model.

### WHY THIS MOTIVATES TASK 5 (narrative spine of the whole report)
Every one of these failure modes is something a transformer (DistilBERT, Task 5) fixes:
- reads word ORDER -> handles the "acquitted ... conspiracy" negation
- has pretrained WORLD KNOWLEDGE -> "knows" winning a contract / capital increase is good
- uses CONTEXT -> understands "new era" connotation
So Task 3's errors are literally the justification for Task 5. In the report we can show
the transformer correctly classifies the exact sentences BoW failed on. Tasks 3 and 5
are the same author -> write it as one coherent arc.

## FULL 2x2 GRID: representation x classifier (KEY RESULT)

macro-F1:

|              | BoW  | TF-IDF |
|--------------|------|--------|
| Naive Bayes  | 0.66 | 0.45   |  <- TF-IDF COLLAPSED NB
| Neural Net   | 0.67 | 0.65   |  <- TF-IDF barely touched NN

Negative recall (where NB's collapse was worst):
- NB  + TF-IDF: 0.06  (catastrophic)
- NN  + TF-IDF: 0.54  (totally fine)

### The insight (sophisticated report point)
TF-IDF is NOT inherently a "bad" representation. Its failure was SPECIFIC TO NAIVE BAYES.
- TF-IDF flattens word-likelihood signals -> NB's class prior dominates -> minority recall
  collapses (neg recall 0.06).
- The NN has NO prior and learns weights directly by gradient descent -> the same
  representation barely affects it (neg recall 0.54, macro-F1 0.65 vs 0.67).
- => Representation effects are NOT absolute; they INTERACT WITH THE CLASSIFIER.

A naive reading "TF-IDF is worse than BoW" is WRONG — it's "TF-IDF is worse for NB
specifically." Running the full 2x2 grid is what reveals this; running only NB+TF-IDF
would give the wrong conclusion. This justifies why we ran all four combinations.

Report sentence:
"TF-IDF degraded Naive Bayes (macro-F1 0.66->0.45, negative recall ->0.06) but left the
neural network essentially unaffected (0.67->0.65), showing that representation effects
interact with the classifier rather than being absolute."
