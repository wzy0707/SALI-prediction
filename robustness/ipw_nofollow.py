# -*- coding: utf-8 -*-
"""
IPW 评估“无 7d TBIL 随访”选择偏倚（审稿意见 三.1）
依赖：先在 pgAdmin 导出  SELECT * FROM mimic_work.sali_ml_dataset_full;
      保存为 sali_ml_dataset_full.csv，与本脚本同目录。
输出：倾向评分/权重分布、加权前后事件率、协变量均衡（SMD）。
"""
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

BASE = os.path.dirname(os.path.abspath(__file__))


def resolve_path(name):
    for cand in (os.path.join(BASE, name), name):
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError("未找到 %s，请确认已导出并与脚本同目录" % name)


PREDICTORS = ["age", "charlson", "sofa", "oasis", "sapsii", "apsiii", "lods",
              "tb0_max", "vasopressor", "mech_vent", "crrt", "steroid_24h",
              "sedative_24h", "positive_culture"]

df = pd.read_csv(resolve_path("sali_ml_dataset_full.csv"))
missing_cols = [c for c in ["has_follow_7d", "label"] + PREDICTORS if c not in df.columns]
if missing_cols:
    raise ValueError("CSV 缺少列：%s" % missing_cols)

X = df[PREDICTORS].apply(pd.to_numeric, errors="coerce")
X = X.fillna(X.median())
y = df["has_follow_7d"].astype(int).values

scaler = StandardScaler()
Xs = scaler.fit_transform(X)
prop_model = LogisticRegression(max_iter=2000, C=1.0)
prop_model.fit(Xs, y)
e = prop_model.predict_proba(Xs)[:, 1]
e = np.clip(e, 0.01, 0.99)
p1 = y.mean()
sw = np.where(y == 1, p1 / e, (1 - p1) / (1 - e))
df["propensity"] = e
df["stab_weight"] = sw

print("样本数 %d | 入组 %d | 无随访 %d" % (len(df), y.sum(), (1 - y).sum()))
print("倾向评分: min %.3f / median %.3f / max %.3f" % (e.min(), np.median(e), e.max()))
print("稳定权重: min %.3f / median %.3f / max %.3f" % (sw.min(), np.median(sw), sw.max()))

# 事件率：仅入组可观察 label；IPW 估计全人群事件率
follow = df[df.has_follow_7d == 1]
crude = follow.label.mean()
weighted = np.average(follow.label.values, weights=follow.stab_weight.values)
print("\n观察事件率（仅入组，n=%d）: %.2f%%" % (len(follow), crude * 100))
print("IPW 加权事件率（全人群口径）: %.2f%%" % (weighted * 100))
print("差值（加权-观察）: %.2f 个百分点（>0 提示排除无随访低估事件率）"
      % ((weighted - crude) * 100))


def smd(group_a, group_b, col, wa=None, wb=None):
    if wa is None:
        va, vb = group_a[col].values, group_b[col]
        ma, mb = np.mean(va), np.mean(vb)
        sa, sb = np.var(va, ddof=1), np.var(vb, ddof=1)
    else:
        ma = np.average(group_a[col], weights=wa)
        mb = np.average(group_b[col], weights=wb)
        sa = np.average((group_a[col] - ma) ** 2, weights=wa)
        sb = np.average((group_b[col] - mb) ** 2, weights=wb)
    pooled = np.sqrt((sa + sb) / 2)
    return (ma - mb) / pooled if pooled > 0 else 0.0


g1 = df[df.has_follow_7d == 1]
g0 = df[df.has_follow_7d == 0]
print("\n协变量均衡（|SMD|，>0.1 视为不均衡）")
print("%-14s %8s %8s" % ("变量", "加权前", "加权后"))
for col in PREDICTORS:
    before = abs(smd(g1, g0, col))
    after = abs(smd(g1, g0, col, g1.stab_weight, g0.stab_weight))
    print("%-14s %8.3f %8.3f" % (col, before, after))

df[["stay_id", "has_follow_7d", "propensity", "stab_weight"]].to_csv(
    os.path.join(BASE, "ipw_weights.csv"), index=False)
print("\n权重已保存: ipw_weights.csv")
