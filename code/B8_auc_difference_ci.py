# -*- coding: utf-8 -*-
"""
B8 · CatBoost 与五变量评分 ROC-AUC 差值的 bootstrap 置信区间
- 在冻结测试集上，对成对样本做分层 bootstrap（默认 2000 次）
- 输出两个模型各自的 AUC 及差值（CatBoost - 五变量）的百分位 CI
- 若差值 CI 跨 0，则“差异不显著”的结论更硬

用法：
  1) 把 FILE 指向含【真实标签 + CatBoost 概率 + 五变量概率】的测试集 CSV；
  2) 若自动识别列名失败，直接在下方手动改 LABEL_COL / CAT_COL / FIVE_COL；
  3) 运行：python B8_auc_difference_ci.py
"""
import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

# ===== 可编辑配置 =====
FILE = ""                       # 留空则自动在脚本目录找含概率的 CSV
N_BOOT = 2000
SEED = 42
LABEL_COL = None                # 例："y" / "label" / "Outcome"
CAT_COL = None                  # 例："prob_catboost"
FIVE_COL = None                 # 例："prob_five" / "prob_frozen"


def find_file():
    here = os.path.dirname(os.path.abspath(__file__))
    cands = glob.glob(os.path.join(here, "*.csv"))
    for f in cands:
        cols = [c.lower() for c in pd.read_csv(f, nrows=1).columns]
        if any("cat" in c for c in cols) and any(
                k in " ".join(cols) for k in ["five", "frozen", "simpl", "score"]):
            return f
    return cands[0] if cands else None


def pick(cols, keys, manual):
    if manual and manual in cols:
        return manual
    for k in keys:
        for c in cols:
            if k in c.lower():
                return c
    return None


path = FILE or find_file()
assert path, "未找到 CSV，请手动设置 FILE"
df = pd.read_csv(path)
cols = list(df.columns)

LABEL_COL = pick(cols, ["label", "outcome", "y_true", " y"], LABEL_COL)
CAT_COL = pick(cols, ["catboost", "cat_boost", "cat"], CAT_COL)
FIVE_COL = pick(cols, ["five", "frozen", "simpl", "score", "logistic"], FIVE_COL)
assert LABEL_COL and CAT_COL and FIVE_COL, (
    "列识别失败，当前列：%s，请手动设置三个列名" % cols)

y = df[LABEL_COL].astype(int).values
p_cat = df[CAT_COL].astype(float).values
p_five = df[FIVE_COL].astype(float).values
n = len(y)
print("文件:", os.path.basename(path))
print("标签列=%s | CatBoost列=%s | 五变量列=%s | n=%d"
      % (LABEL_COL, CAT_COL, FIVE_COL, n))

rng = np.random.default_rng(SEED)
auc_cat, auc_five, diff = [], [], []
for _ in range(N_BOOT):
    idx = rng.integers(0, n, n)         # 成对重采样（同一批样本）
    if len(np.unique(y[idx])) < 2:
        continue
    a = roc_auc_score(y[idx], p_cat[idx])
    b = roc_auc_score(y[idx], p_five[idx])
    auc_cat.append(a)
    auc_five.append(b)
    diff.append(a - b)


def ci(x):
    return np.mean(x), np.percentile(x, 2.5), np.percentile(x, 97.5)


print("\nCatBoost  AUC = %.3f (95%% CI %.3f\u2013%.3f)" % ci(auc_cat))
print("Five-var  AUC = %.3f (95%% CI %.3f\u2013%.3f)" % ci(auc_five))
m, lo, hi = ci(diff)
print("差值 (CatBoost - Five) = %+.4f (95%% CI %+.4f to %+.4f)" % (m, lo, hi))
print("结论:", "CI 跨 0，差异不显著" if lo < 0 < hi else "CI 不跨 0，存在显著差异")
