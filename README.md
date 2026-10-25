# BT4012 Kaggle Competition: Illicit Bitcoin Transaction Detection

Predict the probability that a Bitcoin transaction is illicit, for transactions in **future** time
steps (36–49), using labelled transactions from steps 1–35, 165 anonymised features per transaction,
and the transaction graph. Metric: ROC-AUC.

## Final submissions selected

| Output file | Model | Public LB |
|---|---|---|
| `submissions/final_sub23_stack_ensemble8.csv` | Rank-average of 8 graph-stacked models | **0.95985** |
| `submissions/final_sub14_graphstack.csv` | Single graph-stacked LightGBM model | 0.95903 |

**Approach:** Two families of graph features are built from the edge list: degree and
neighbour means of the local features, and *labelled-neighbour* features (how many neighbours are
labelled vs unlabelled, and feature means over labelled neighbours only). In the test period an
unlabelled transaction has no feature row, so "is this neighbour labelled?" means the same thing in
train and test. Additionally, **graph stacking:** a first-stage LightGBM scores every labelled
transaction (out-of-fold, grouped by time step, for training rows). A second-stage model then uses
each transaction's own features plus the predicted scores of its labelled neighbours. The top
submission averages the ranks of 8 such stacked models that differ in learner (LightGBM / XGBoost),
stage-1 scores and stage-2 base features.

## Reproduce

1. **Python 3.12 or 3.13:** Tested on 3.13 (Linux) and 3.12 (Windows). Avoid 3.15: pandas 3.0.5
   has no prebuilt wheel for it on Windows and pip falls back to a source build that needs Visual
   Studio build tools. Create a virtual environment and install:
   ```
   py -3.12 -m venv .venv           # Windows   (macOS/Linux: python3 -m venv .venv)
   .venv\Scripts\activate          # Windows   (macOS/Linux: source .venv/bin/activate)
   pip install -r requirements.txt
   ```
2. **Data:** Download the competition files from Kaggle and place them in a folder named `data/`
   next to the notebook (the dataset is not in this repo):
   ```
   data/train.csv
   data/test.csv
   data/txs_edgelist.csv
   data/sample_submission.csv
   ```
3. **Run** `e1121405.ipynb` top to bottom. The two final submission files are written to
   `submissions/` (created automatically).
4. *(Optional)* **Validation metrics in the report.** `python metrics_f3.py` recomputes the
   threshold metrics table (ROC-AUC, average precision, accuracy, precision, recall, F1 at 0.5)
   on fold F3 (train steps 1–28, validate 29–35). About 6 minutes; writes `metrics_f3.csv`.

**Runtime:** About 2 hours on a 2-core CPU, faster with more cores. Almost all of it is Section 7: each of
the 8 stacked models trains 7 models per seed, over 3 seeds. To regenerate only the submission files,
set `RUN_ABLATION = False` and `RUN_STACK_VALIDATION = False` in the first code cell. This skips the
validation experiments and leaves the final predictions unchanged.

**Determinism:** Every model uses a fixed seed. On the original machine (Linux, Python 3.13),
re-running the notebook reproduced the submitted CSVs to floating-point precision (max absolute
difference ≈ 1e-16). On a different platform (Windows, Python 3.12), multithreaded tree libraries
give slightly different floating-point results: `final_sub14_graphstack.csv` differed on 5 of 15,329
rows (max absolute difference 0.0018), and `final_sub23_stack_ensemble8.csv`, a rank average,
matched the submitted file with Spearman correlation 0.9998.

## Repo layout
```
e1121405.ipynb     # EDA → graph features → temporal validation → graph stacking → submissions
metrics_f3.py      # threshold metrics on validation fold F3 (report, Section 3)
requirements.txt
README.md
data/              # (git-ignored) competition files — place them here
cache/             # (git-ignored) parquet copies + engineered features, auto-created
submissions/       # (git-ignored) generated submission CSVs
```

## Notebook outline
1. Setup & data loading
2. EDA: label imbalance over time, and how the test set was constructed
3. Graph feature engineering (v1: degree + neighbour means; v2: labelled-neighbour features)
4. Validation design: time-ordered folds, and how far they track the leaderboard
5. Feature-set ablation
6. Single-stage models (Model A, Model B)
7. Graph stacking: validation, sub14, rich variants, and the 8-model ensemble (sub23)
8. Post-step-43 analysis (the dark-market shutdown)
9. Experiment log of all Kaggle submissions, including negative results
