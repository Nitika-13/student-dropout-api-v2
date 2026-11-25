"""
train_model.py (v2) - trains the FINAL model (Logistic Regression L2) for checkpoints A, B and C
and saves them in ONE file: model_bundle.joblib

v2 uses the old 19 features PLUS age, gender, special needs, international flag and the parents'
occupation group (9 broad groups each).  Regularisation strength C is re-tuned on the training rows
(3-fold inner CV, scoring macro-F1) because the feature set changed.

Run once on your laptop:   python train_model.py --data data.csv --split split_indices.json
Trains on the 3539 TRAINING rows only (the locked 885-row test set is never used).
The Dropout threshold (for 80% recall) is chosen from out-of-fold predictions on training rows.
"""
import argparse, hashlib, json, sys, warnings
import joblib, numpy as np, pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_recall_curve
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import features as F

warnings.filterwarnings("ignore")
SEED, FP = 42, "598f3aadcc7f5293ce4bb2b883582dcf"
LABELS = ["Dropout", "Enrolled", "Graduate"]
CP = {"A": F.NEW_A, "B": F.NEW_B, "C": F.NEW_C}
C_GRID = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 63.5]


def load(path):
    first = open(path, encoding="utf-8-sig").readline().strip()
    if first.startswith('"') and first.endswith('"') and ";" in first:
        names = [c.strip().strip('"').replace("\t", "") for c in first[1:-1].replace('""', '"').split(";")]
        df = pd.read_csv(path, sep=";", skiprows=1, header=None, names=names, encoding="utf-8-sig")
    else:
        df = pd.read_csv(path, sep=";" if first.count(";") > first.count(",") else ",", encoding="utf-8-sig")
    df.columns = [c.strip().replace("﻿", "") for c in df.columns]
    return df


def make_pipe(C=1.0):
    # mean imputation: for 0/1 columns the mean is the training share, so a missing value is not
    # silently treated as "female" or "no special needs".  (Training rows have no gaps.)
    return Pipeline([("imputer", SimpleImputer(strategy="mean")), ("scaler", StandardScaler()),
                     ("model", LogisticRegression(C=C, class_weight="balanced", max_iter=5000,
                                                  random_state=SEED))])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data.csv")
    ap.add_argument("--split", default="split_indices.json")
    a = ap.parse_args()
    df = load(a.data)
    assert len(df) == 4424, f"expected 4424 rows, got {len(df)}"
    y_all = df["Target"].map({n: i for i, n in enumerate(LABELS)}).to_numpy()
    X_all = F.build_features(df)
    s = json.load(open(a.split))
    tr = np.array(s.get("train_idx", s.get("train")), dtype="int64")
    fp = hashlib.md5(tr.tobytes()).hexdigest()
    print("split fingerprint:", fp)
    if fp != FP:
        sys.exit("Split fingerprint mismatch - use the team's split_indices.json")
    X, y = X_all.iloc[tr].reset_index(drop=True), y_all[tr]
    bundle = {"labels": LABELS, "version": "v2", "models": {}}
    for cp, cols in CP.items():
        search = GridSearchCV(make_pipe(), {"model__C": C_GRID}, scoring="f1_macro",
                              cv=StratifiedKFold(3, shuffle=True, random_state=SEED)).fit(X[cols], y)
        bestC = search.best_params_["model__C"]
        pipe = make_pipe(bestC)
        oof = cross_val_predict(pipe, X[cols], y, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                                method="predict_proba")
        _, rec, thr = precision_recall_curve((y == 0).astype(int), oof[:, 0])
        ok = np.where(rec[:-1] >= 0.80)[0]
        t = float(thr[ok[-1]]) if len(ok) else float(thr[0])
        pipe.fit(X[cols], y)
        bundle["models"][cp] = {"pipeline": pipe, "columns": cols, "threshold": round(t, 4), "C": bestC,
                                "train_macro_f1_oof": round(float(f1_score(y, oof.argmax(1), average="macro")), 4)}
        print(f"checkpoint {cp}: {len(cols)} features | C={bestC} | out-of-fold macro-F1 "
              f"{bundle['models'][cp]['train_macro_f1_oof']} | Dropout threshold for 80% recall {t:.3f}")
    joblib.dump(bundle, "model_bundle.joblib")
    print("Saved model_bundle.joblib")


if __name__ == "__main__":
    main()
