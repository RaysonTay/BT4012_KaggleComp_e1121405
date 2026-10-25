"""To reproduce the metrics table in Section 3 of the report (Section 3): ROC-AUC, average precision, and
accuracy / precision / recall / F1 / confusion counts at a 0.5 threshold, for the raw baseline,
Model B, Model A, graph stacking (the sub14 pipeline) and rich graph stacking. Seed 1.

Uses the same feature engineering, model settings and stacking code as the notebook.
Run from the repo root after placing the Kaggle files in data/:

    python metrics_f3.py

Takes about 6 minutes on a 2-core CPU. Writes metrics_f3.csv next to this script.
"""
from pathlib import Path
import time

import numpy as np
import pandas as pd
import lightgbm as lgb
import scipy.sparse as sp
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score,
                             precision_score, recall_score, f1_score, confusion_matrix)

SEED = 42
DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

LGB_PARAMS = dict(
    n_estimators=500, learning_rate=0.03, num_leaves=31,
    subsample=0.8, subsample_freq=1, colsample_bytree=0.5,
    random_state=SEED, verbose=-1,
)

def load_csv_cached(name):
    pq = CACHE_DIR / f"{name}.parquet"
    if pq.exists():
        return pd.read_parquet(pq)
    df = pd.read_csv(DATA_DIR / f"{name}.csv")
    df.to_parquet(pq)
    return df


train = load_csv_cached("train")
test = load_csv_cached("test")
edges = pd.read_csv(DATA_DIR / "txs_edgelist.csv")
FEATS = [c for c in train.columns if c.startswith("feat_")]
LOCAL = FEATS[:93]
lab = train[train.label.notna()].copy()
lab["label"] = lab["label"].astype(int)


# graph features (Section 3)
def build_graph_features_v1(train, test, edges, agg_cols):
    allx = pd.concat([train[["txId"] + agg_cols], test[["txId"] + agg_cols]]).set_index("txId")
    out = pd.DataFrame(index=allx.index)
    out["in_deg"] = edges.txId2.value_counts().reindex(out.index).fillna(0)
    out["out_deg"] = edges.txId1.value_counts().reindex(out.index).fillna(0)
    has1, has2 = edges.txId1.isin(allx.index), edges.txId2.isin(allx.index)
    e = edges[has1 & has2]
    in_mean = allx.loc[e.txId1].set_axis(e.txId2.values).groupby(level=0).mean()
    out_mean = allx.loc[e.txId2].set_axis(e.txId1.values).groupby(level=0).mean()
    return out.join(in_mean.add_prefix("nbr_in_mean_")).join(out_mean.add_prefix("nbr_out_mean_"))


def build_graph_features_v2(lab, test, edges, feats, agg_cols):
    labelled = pd.concat([lab[["txId"] + feats], test[["txId"] + feats]]).set_index("txId")
    nodes = pd.unique(edges[["txId1", "txId2"]].values.ravel())
    status = pd.Series("U", index=nodes)
    status[status.index.isin(labelled.index)] = "L"
    e = edges.assign(s1=status.reindex(edges.txId1).values, s2=status.reindex(edges.txId2).values)
    G = pd.DataFrame(index=labelled.index)
    for nm, (tgt, src, ss) in {"in": ("txId2", "txId1", "s1"), "out": ("txId1", "txId2", "s2")}.items():
        for s in "LU":
            G[f"n{s}_{nm}"] = e[e[ss] == s][tgt].value_counts().reindex(G.index).fillna(0)
        el = e[(e[ss] == "L") & e[tgt].isin(G.index)]
        nb = labelled.loc[el[src].values, agg_cols].set_axis(el[tgt].values)
        G = G.join(nb.groupby(level=0).mean().add_prefix(f"Lmean_{nm}_"))
    G["nL"] = G.nL_in + G.nL_out
    G["nU"] = G.nU_in + G.nU_out
    G["fracL"] = G.nL / (G.nL + G.nU).replace(0, np.nan)
    return G


def cached(name, build):
    p = CACHE_DIR / name
    if p.exists():
        return pd.read_parquet(p)
    g = build()
    g.to_parquet(p)
    return g


g1 = cached("graph_feats_v1.parquet", lambda: build_graph_features_v1(train, test, edges, LOCAL))
g2 = cached("graph_feats_v2.parquet", lambda: build_graph_features_v2(lab, test, edges, FEATS, LOCAL))
lab = lab.join(g1, on="txId").join(g2, on="txId").copy()

DEG = ["in_deg", "out_deg"]
NBR_V1 = [c for c in g1.columns if c.startswith("nbr_")]
CNT = ["nL_in", "nL_out", "nU_in", "nU_out", "nL", "nU", "fracL"]
LMEAN = [c for c in g2.columns if c.startswith("Lmean_")]
COLS_A = FEATS + CNT + LMEAN          # Model A
COLS_B = FEATS + DEG + NBR_V1         # Model B


# models
def fit_lgb(df, cols, seed, n_estimators=1000):
    m = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": n_estimators, "random_state": seed})
    return m.fit(df[cols], df.label)


def oof_and_apply(tr, ap, cols, seed, k=5):
    oof = np.zeros(len(tr))
    for fit_idx, pred_idx in GroupKFold(k).split(tr, groups=tr.time_step):
        m = fit_lgb(tr.iloc[fit_idx], cols, seed)
        oof[pred_idx] = m.predict_proba(tr.iloc[pred_idx][cols])[:, 1]
    return oof, fit_lgb(tr, cols, seed).predict_proba(ap[cols])[:, 1]


def neighbour_score_features(ids, p, rich=False):
    s = pd.Series(p, index=ids)
    e = edges[edges.txId1.isin(s.index) & edges.txId2.isin(s.index)]
    out = pd.DataFrame(index=ids)
    for nm, (tgt, src) in {"in": ("txId2", "txId1"), "out": ("txId1", "txId2")}.items():
        g = pd.Series(s.reindex(e[src]).values, index=e[tgt].values).groupby(level=0)
        out[f"pn_{nm}_mean"], out[f"pn_{nm}_max"] = g.mean(), g.max()
    all_p = pd.concat([pd.Series(s.reindex(e.txId1).values, index=e.txId2.values),
                       pd.Series(s.reindex(e.txId2).values, index=e.txId1.values)])
    g = all_p.groupby(level=0)
    out["pn_mean"], out["pn_max"], out["pn_sum"] = g.mean(), g.max(), g.sum()
    if rich:
        out["pn_std"] = g.std()
        out["pn_n_hi"] = (all_p > 0.5).groupby(level=0).sum()
        pos = pd.Series(np.arange(len(s)), index=s.index)
        a, b, n = pos[e.txId1].values, pos[e.txId2].values, len(s)
        S = sp.coo_matrix((np.ones(len(a)), (a, b)), shape=(n, n)).tocsr(); S = S + S.T
        S2 = (S @ S).tocsr(); S2.setdiag(0); S2.eliminate_zeros()
        d2 = np.asarray(S2.sum(1)).ravel()
        out["pn_2hop_mean"] = np.where(d2 > 0, (S2 @ s.values) / np.maximum(d2, 1), np.nan)
    return out


def stacked_predict(tr, ap, stage1_cols, stage2_base_cols, seed, rich=False):
    oof, p_ap = oof_and_apply(tr, ap, stage1_cols, seed)
    F = neighbour_score_features(np.r_[tr.txId.values, ap.txId.values], np.r_[oof, p_ap], rich=rich)
    tr2, ap2 = tr.join(F, on="txId"), ap.join(F, on="txId")
    cols2 = stage2_base_cols + list(F.columns)
    return fit_lgb(tr2, cols2, seed).predict_proba(ap2[cols2])[:, 1]


# fold F3 metrics
tr, va = lab[lab.time_step <= 28], lab[lab.time_step.between(29, 35)]
y = va.label.values
S = 1  # seed
MODELS = {
    "Raw 165 features":       lambda: fit_lgb(tr, FEATS, S).predict_proba(va[FEATS])[:, 1],
    "Model B (raw + v1)":     lambda: fit_lgb(tr, COLS_B, S).predict_proba(va[COLS_B])[:, 1],
    "Model A (raw + v2)":     lambda: fit_lgb(tr, COLS_A, S).predict_proba(va[COLS_A])[:, 1],
    "Graph stacking (sub14)": lambda: stacked_predict(tr, va, COLS_A, COLS_A, S),
    "Rich graph stacking":    lambda: stacked_predict(tr, va, COLS_A, COLS_A, S, rich=True),
}

print(f"F3 validation: {len(y)} transactions, {int(y.sum())} illicit; "
      f"all-licit accuracy = {1 - y.mean():.4f}")
rows = []
for name, predict in MODELS.items():
    t0 = time.time()
    p = predict()
    b = (p >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, b).ravel()
    rows.append(dict(model=name, roc_auc=roc_auc_score(y, p), avg_precision=average_precision_score(y, p),
                     accuracy=accuracy_score(y, b), precision=precision_score(y, b),
                     recall=recall_score(y, b), f1=f1_score(y, b),
                     FN=int(fn), FP=int(fp), TP=int(tp), TN=int(tn)))
    print(f"{name}: {time.time() - t0:.0f}s")

res = pd.DataFrame(rows).set_index("model")
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(res.round(4))
res.to_csv("metrics_f3.csv")
