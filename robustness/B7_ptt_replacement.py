# -*- coding: utf-8 -*-
"""
B7 · PTT 保留的经验证据：PT / INR 替换敏感性分析
- 同一五变量结构，仅替换凝血指标：PTT / PT / Baseline INR
- 重复分层 5 折 CV（10 次，seed 42），fold 内中位数插补 + 标准化，无泄漏
- 比较 ROC-AUC / PR-AUC，并给出配对差值的分布
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

BASE4 = ["Baseline TBIL", "Sodium", "Platelets", "Lactate"]
MARKERS = ["PTT", "PT", "Baseline INR"]

df = pd.read_csv("origindata.csv")
y = df["Outcome"].astype(int).values
rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

records = []
for marker in MARKERS:
    X = df[BASE4 + [marker]].apply(pd.to_numeric, errors="coerce").values
    rocs, prs, coefs = [], [], []
    for tr, te in rskf.split(X, y):
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("sc", StandardScaler()),
            ("lr", LogisticRegression(C=1.0, penalty="l2",
                                      solver="lbfgs", max_iter=2000)),
        ])
        pipe.fit(X[tr], y[tr])
        p = pipe.predict_proba(X[te])[:, 1]
        rocs.append(roc_auc_score(y[te], p))
        prs.append(average_precision_score(y[te], p))
        coefs.append(pipe.named_steps["lr"].coef_[0][-1])
    records.append({
        "Coagulation marker": marker,
        "ROC-AUC (mean)": np.mean(rocs),
        "ROC-AUC 2.5-97.5%": "%.3f-%.3f" % (np.percentile(rocs, 2.5),
                                            np.percentile(rocs, 97.5)),
        "PR-AUC (mean)": np.mean(prs),
        "PR-AUC 2.5-97.5%": "%.3f-%.3f" % (np.percentile(prs, 2.5),
                                           np.percentile(prs, 97.5)),
        "Std coef (mean)": np.mean(coefs),
    })

res = pd.DataFrame(records)
pd.set_option("display.width", 200)
print(res.to_string(index=False, float_format=lambda v: "%.3f" % v))

# 配对差值（同样的 fold，PTT 相对 PT/INR）
Xbase = df[BASE4].apply(pd.to_numeric, errors="coerce").values
folds = list(rskf.split(np.zeros(len(df)), y))


def cv_auc(col):
    X = np.column_stack([Xbase, pd.to_numeric(df[col], errors="coerce").values])
    out = []
    for tr, te in folds:
        pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                         ("sc", StandardScaler()),
                         ("lr", LogisticRegression(max_iter=2000))])
        pipe.fit(X[tr], y[tr])
        out.append(roc_auc_score(y[te], pipe.predict_proba(X[te])[:, 1]))
    return np.array(out)


a_ptt, a_pt, a_inr = cv_auc("PTT"), cv_auc("PT"), cv_auc("Baseline INR")
for name, d in [("PTT - PT", a_ptt - a_pt), ("PTT - INR", a_ptt - a_inr)]:
    print("\n配对 ROC-AUC 差值 %s: mean %+.4f, 2.5-97.5%% [%+.4f, %+.4f]"
          % (name, d.mean(), np.percentile(d, 2.5), np.percentile(d, 97.5)))

res.to_csv("B7_PT_INR替换敏感性.csv", index=False, encoding="utf-8-sig")
print("\n已保存 B7_PT_INR替换敏感性.csv")
