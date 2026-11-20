# -*- coding: utf-8 -*-
# =====================================================================
# B1 · eICU 外部验证：cluster bootstrap（按 hospitalid 聚类）+ 混淆矩阵/PPV
# 对应审稿意见：M1（医院聚类，113 家）、M2（PPV/完整混淆矩阵）
#
# 数据（与脚本同目录，缺列会报错提示）：
#   eicu_sali_validation.csv  含 hospitalid / 五变量 / outcome
#   eicu_predictions.csv      含 y / prob_frozen（用于核对，可选）
#   五变量OR表.csv            精确原始尺度系数（可选，没有则用内置 3 位系数）
#
# 运行：  C:/python310/python.exe B1_eicu_cluster_bootstrap.py
# 依赖：  pandas numpy scikit-learn scipy
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             brier_score_loss, confusion_matrix)

SEED = 42
N_BOOT = 2000
# 五变量模型的固定 cut-off（开发端冻结，不在 eICU 重选）。
# 精确值见本机 python old liver 输出/表/五变量测试集指标.csv 的 threshold；
# 若留空，脚本用 eICU 自身 Youden 仅作参考并明确标注。
CUTOFF_PROB = 0.062145  # 开发端冻结阈值（训练集 Youden，原始尺度），不在 eICU 重选

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else name


five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]

# ---- 冻结系数（部署版 3 位系数，与 eicu_external_validation.py 完全一致）----
# 这是外验冻结模型的唯一口径；6 位 OR 表是开发端原始拟合，不用于外验。
beta = {"Intercept": 2.937, "Baseline TBIL": 1.888, "PTT": 0.002,
        "Sodium": -0.055, "Platelets": -0.003, "Lactate": 0.139}
print("使用冻结部署系数（3 位）。")

# ---- 读取并复刻外验预处理 ----
d = pd.read_csv(resolve_path("eicu_sali_validation.csv"))
rename = {"ptt": "PTT", "sodium": "Sodium", "platelets": "Platelets",
          "lactate": "Lactate", "outcome": "Outcome"}
d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
d.loc[d["PTT"] < 15, "PTT"] = np.nan
d.loc[d["Sodium"] < 100, "Sodium"] = np.nan
for v in five:
    if d[v].isna().any():
        d[v] = d[v].fillna(d[v].median())
d = d.reset_index(drop=True)
y = d["Outcome"].astype(int).values
hosp = d["hospitalid"].values


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


lp = np.full(len(d), beta["Intercept"])
for v in five:
    lp += beta[v] * d[v].astype(float).values
p = sigmoid(lp)

# 与现有 eicu_predictions.csv 核对
pred_path = resolve_path("eicu_predictions.csv")
if os.path.exists(pred_path):
    pr = pd.read_csv(pred_path)
    mad = np.abs(pr["prob_frozen"].values - p).max()
    print("冻结概率与 eicu_predictions.csv 最大绝对差 = %.2e %s"
          % (mad, "（一致）" if mad < 1e-2 else "（请检查系数口径）"))

print("\n外部队列: n=%d | 事件=%d | 事件率=%.2f%% | 医院数=%d"
      % (len(y), y.sum(), 100 * y.mean(), len(np.unique(hosp))))


def metric_set(yy, pp):
    return {
        "ROC_AUC": roc_auc_score(yy, pp),
        "PR_AUC": average_precision_score(yy, pp),
        "Brier": brier_score_loss(yy, pp),
    }


point = metric_set(y, p)
print("\n点估计:", {k: round(v, 4) for k, v in point.items()})

# ---- 1) cluster bootstrap：按医院整群重采样 ----
rng = np.random.default_rng(SEED)
unique_hosp = np.unique(hosp)
by_hosp = {h: np.where(hosp == h)[0] for h in unique_hosp}
cluster_rows = {k: [] for k in point}
for b in range(N_BOOT):
    picked = rng.choice(unique_hosp, size=len(unique_hosp), replace=True)
    idx = np.concatenate([by_hosp[h] for h in picked])
    yy, pp = y[idx], p[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        continue
    m = metric_set(yy, pp)
    for k in cluster_rows:
        cluster_rows[k].append(m[k])

print("\n=== M1 cluster bootstrap 95%% CI（%d 次有效重采样）===" % N_BOOT)
ci_table = {}
for k in point:
    lo, hi = np.percentile(cluster_rows[k], [2.5, 97.5])
    ci_table[k] = (point[k], lo, hi)
    print("%-8s %.3f (95%% CI %.3f–%.3f)" % (k, point[k], lo, hi))

# ---- 2) 普通（患者级）bootstrap 对照，展示聚类修正幅度 ----
plain_rows = {k: [] for k in point}
for b in range(N_BOOT):
    idx = rng.integers(0, len(y), len(y))
    yy, pp = y[idx], p[idx]
    if yy.sum() == 0 or yy.sum() == len(yy):
        continue
    m = metric_set(yy, pp)
    for k in point:
        plain_rows[k].append(m[k])
print("\n=== 普通 bootstrap 95%% CI（对照，偏乐观）===")
for k in point:
    lo, hi = np.percentile(plain_rows[k], [2.5, 97.5])
    print("%-8s %.3f (95%% CI %.3f–%.3f)" % (k, point[k], lo, hi))

# ---- 3) 固定 cut-off 下完整混淆矩阵 + PPV ----
if CUTOFF_PROB is None:
    # 仅作参考：eICU 自身 Youden（不是冻结口径，正文勿当主结果）
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    CUTOFF_PROB = float(thr[np.argmax(tpr - fpr)])
    print("\n[提示] 未指定冻结 cut-off，以下为 eICU 自身 Youden=%.4f，仅参考；"
          "请把开发端阈值填入脚本顶部 CUTOFF_PROB 后重跑。" % CUTOFF_PROB)

pred_cls = (p >= CUTOFF_PROB).astype(int)
tn, fp, fn, tp = confusion_matrix(y, pred_cls, labels=[0, 1]).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
ppv = tp / (tp + fp) if (tp + fp) else np.nan
npv = tn / (tn + fn) if (tn + fn) else np.nan
print("\n=== 混淆矩阵（cut-off %.4f）===" % CUTOFF_PROB)
print("TP=%d FP=%d FN=%d TN=%d" % (tp, fp, fn, tn))
print("Sensitivity %.3f | Specificity %.3f | PPV %.3f | NPV %.3f"
      % (sens, spec, ppv, npv))

# ---- 保存结果 ----
out = pd.DataFrame({
    "metric": list(point.keys()),
    "point": [point[k] for k in point],
    "cluster_ci_low": [ci_table[k][1] for k in point],
    "cluster_ci_high": [ci_table[k][2] for k in point],
    "plain_ci_low": [np.percentile(plain_rows[k], 2.5) for k in point],
    "plain_ci_high": [np.percentile(plain_rows[k], 97.5) for k in point],
})
out_path = os.path.join(SCRIPT_DIR, "B1_eicu_cluster_bootstrap结果.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
cm = pd.DataFrame([{"cutoff": CUTOFF_PROB, "TP": tp, "FP": fp, "FN": fn,
                    "TN": tn, "Sensitivity": sens, "Specificity": spec,
                    "PPV": ppv, "NPV": npv}])
cm_path = os.path.join(SCRIPT_DIR, "B1_eicu_混淆矩阵.csv")
cm.to_csv(cm_path, index=False, encoding="utf-8-sig")
print("\n已保存:", out_path)
print("已保存:", cm_path)
