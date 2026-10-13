"""Train baseline (Logistic Regression) vs. XGBoost, pick a cost-based threshold,
evaluate on a held-out test set, and generate SHAP explanations.

Run:  python -m src.train
"""
import json
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import shap
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from src.data import TARGET, feature_columns, load_data

SEED = 42
# Business assumptions (edit these to match your story):
OFFER_COST = 40      # cost of a retention offer per contacted customer ($)
CUSTOMER_VALUE = 300  # value of a saved customer ($)
OFFER_SUCCESS = 0.4   # share of at-risk customers an offer actually retains


def net_benefit(y_true, proba, thr):
    """Expected profit of contacting everyone with proba >= thr."""
    contact = proba >= thr
    saved = (contact & (y_true == 1)).sum() * OFFER_SUCCESS * CUSTOMER_VALUE
    cost = contact.sum() * OFFER_COST
    return saved - cost


def best_threshold(y_true, proba):
    grid = np.arange(0.05, 0.95, 0.01)
    scores = [net_benefit(y_true, proba, t) for t in grid]
    return float(grid[int(np.argmax(scores))])


def metrics(y_true, proba, thr=0.5):
    pred = (proba >= thr).astype(int)
    return {
        "roc_auc": round(roc_auc_score(y_true, proba), 4),
        "pr_auc": round(average_precision_score(y_true, proba), 4),
        "precision": round(precision_score(y_true, pred), 4),
        "recall": round(recall_score(y_true, pred), 4),
        "f1": round(f1_score(y_true, pred), 4),
        "threshold": round(float(thr), 2),
    }


def main():
    df = load_data()
    num, cat = feature_columns(df)
    X, y = df[num + cat], df[TARGET]

    # 60 / 20 / 20 stratified split: train / validation (threshold) / test (final report)
    X_tmp, X_test, y_tmp, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_tmp, y_tmp, test_size=0.25, stratify=y_tmp, random_state=SEED)

    pre = ColumnTransformer([
        ("num", StandardScaler(), num),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat),
    ])
    Xtr = pre.fit_transform(X_train)
    Xva, Xte = pre.transform(X_val), pre.transform(X_test)
    names = list(pre.get_feature_names_out())

    # ---- Baseline ----
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED)
    lr.fit(Xtr, y_train)
    lr_val, lr_test = lr.predict_proba(Xva)[:, 1], lr.predict_proba(Xte)[:, 1]

    # ---- XGBoost ----
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    xgb = XGBClassifier(
        n_estimators=400, learning_rate=0.03, max_depth=3, subsample=0.8,
        colsample_bytree=0.8, min_child_weight=3, reg_lambda=2.0,
        scale_pos_weight=spw, eval_metric="auc", early_stopping_rounds=40,
        random_state=SEED, n_jobs=-1)
    xgb.fit(Xtr, y_train, eval_set=[(Xva, y_val)], verbose=False)
    xgb_val, xgb_test = xgb.predict_proba(Xva)[:, 1], xgb.predict_proba(Xte)[:, 1]

    # ---- Threshold chosen on VALIDATION set only (no test leakage) ----
    thr_lr, thr_xgb = best_threshold(y_val.values, lr_val), best_threshold(y_val.values, xgb_val)

    results = {
        "logistic_regression_baseline": {
            "default_0.5": metrics(y_test, lr_test, 0.5),
            "cost_optimised": metrics(y_test, lr_test, thr_lr),
            "net_benefit_$": round(net_benefit(y_test.values, lr_test, thr_lr), 0)},
        "xgboost": {
            "default_0.5": metrics(y_test, xgb_test, 0.5),
            "cost_optimised": metrics(y_test, xgb_test, thr_xgb),
            "net_benefit_$": round(net_benefit(y_test.values, xgb_test, thr_xgb), 0)},
        "no_model_contact_everyone_net_benefit_$": round(net_benefit(y_test.values, np.ones(len(y_test)), 0.0), 0),
        "assumptions": {"offer_cost": OFFER_COST, "customer_value": CUSTOMER_VALUE, "offer_success_rate": OFFER_SUCCESS},
        "test_size": int(len(y_test)), "test_churn_rate": round(float(y_test.mean()), 4),
        "best_iteration": int(xgb.best_iteration),
    }
    cm = confusion_matrix(y_test, (xgb_test >= thr_xgb).astype(int))
    results["xgboost_confusion_matrix [[TN,FP],[FN,TP]]"] = cm.tolist()

    # ---- SHAP ----
    explainer = shap.TreeExplainer(xgb)
    sample = Xte[:600]
    sv = explainer.shap_values(sample)
    plt.figure(); shap.summary_plot(sv, sample, feature_names=names, show=False, max_display=15)
    plt.tight_layout(); plt.savefig("reports/shap_summary.png", dpi=150, bbox_inches="tight"); plt.close()
    plt.figure(); shap.summary_plot(sv, sample, feature_names=names, plot_type="bar", show=False, max_display=15)
    plt.tight_layout(); plt.savefig("reports/shap_importance.png", dpi=150, bbox_inches="tight"); plt.close()

    # ---- Threshold / profit curve ----
    grid = np.arange(0.05, 0.95, 0.01)
    plt.figure(figsize=(6, 4))
    plt.plot(grid, [net_benefit(y_val.values, xgb_val, t) for t in grid])
    plt.axvline(thr_xgb, color="r", ls="--", label=f"best threshold = {thr_xgb:.2f}")
    plt.xlabel("Decision threshold"); plt.ylabel("Net benefit ($, validation set)")
    plt.title("Cost-based threshold selection"); plt.legend(); plt.tight_layout()
    plt.savefig("reports/threshold_curve.png", dpi=150); plt.close()

    # ---- Save artifacts ----
    joblib.dump({"preprocessor": pre, "model": xgb, "threshold": thr_xgb,
                 "feature_names": names, "numeric": num, "categorical": cat,
                 "categories": {c: list(X[c].unique()) for c in cat}}, "models/churn_model.joblib")
    json.dump(results, open("reports/metrics.json", "w"), indent=2)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
