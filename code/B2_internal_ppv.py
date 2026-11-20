# -*- coding: utf-8 -*-
# =====================================================================
# B2 · MIMIC 内部测试集：六模型 + 五变量的 PPV 与完整混淆矩阵
# 对应审稿意见：M2（方法承诺 PPV 却只报 NPV）
#
# 自动兼容两种文件格式，并优先选择当前版本（测试集 1,069 例 / 61 事件）：
#   长表（当前，SALI_建模流程）：predictions_oof.csv(model,y,prob,threshold)、
#                               predictions_test.csv(model,y,prob)
#   宽表（旧版，会被跳过）：列 y + 六模型名
#   五变量：simple_score_predictions.csv(y,prob) + 五变量测试集指标.csv(threshold)
# 运行：  C:/python310/python.exe B2_internal_ppv.py
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, confusion_matrix

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
FIVE_CUTOFF = 0.062145  # 五变量开发端冻结阈值（训练集 Youden，原始尺度）


def find_all(name):
    # 搜索脚本目录，并向上覆盖父目录、祖父目录（兼容在 eicu 子文件夹里运行）
    roots = [SCRIPT_DIR]
    up = SCRIPT_DIR
    for _ in range(2):
        up = os.path.dirname(up)
        if up and up not in roots:
            roots.append(up)
    hits = []
    seen = set()
    for base in roots:
        for root, _, files in os.walk(base):
            if name in files:
                p = os.path.join(root, name)
                if p not in seen:
                    seen.add(p)
                    hits.append(p)
    return hits


def load_best(name):
    """在所有同名文件里选当前版本（测试/oof 事件数与论文一致），返回 (长表, 路径)。"""
    cands = []
    for p in find_all(name):
        d = pd.read_csv(p)
        if "model" in d.columns:  # 长表
            uniq = d[d["model"] == d["model"].iloc[0]]
            cands.append((len(uniq), int(uniq["y"].sum()), p, d))
        else:  # 宽表 → 转长表
            wide = d
            long = wide.melt(id_vars="y", var_name="model", value_name="prob")
            cands.append((len(wide), int(wide["y"].sum()), p, long))
    if not cands:
        return None, None
    target_events = 61 if "test" in name else 144
    for n, ev, p, d in cands:
        if ev == target_events:
            print("选用当前版本: %s (n=%d, events=%d)" % (p, n, ev))
            return d, p
    n, ev, p, d = sorted(cands, key=lambda x: x[0], reverse=True)[0]
    print("[警告] 未找到事件数=%d 的版本，回退: %s (events=%d)，请核对。"
          % (target_events, p, ev))
    return d, p


def youden(yy, pp):
    fpr, tpr, thr = roc_curve(yy, pp)
    return float(thr[np.argmax(tpr - fpr)])


def confusion_row(yy, pp, cut):
    pred = (pp >= cut).astype(int)
    tn, fp, fn, tp = confusion_matrix(yy, pred, labels=[0, 1]).ravel()
    return {"cutoff": cut, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "Sensitivity": tp / (tp + fn), "Specificity": tn / (tn + fp),
            "PPV": tp / (tp + fp) if (tp + fp) else np.nan,
            "NPV": tn / (tn + fn) if (tn + fn) else np.nan}


oof, oof_path = load_best("predictions_oof.csv")
test, test_path = load_best("predictions_test.csv")
assert oof is not None and test is not None, "缺少 predictions_oof/test.csv"

rows = []
for m in [c for c in test["model"].unique()]:
    o = oof[oof["model"] == m]
    t = test[test["model"] == m]
    # 长表若带 threshold 列直接用，否则用 OOF Youden 固定
    if "threshold" in oof.columns and o["threshold"].nunique() == 1:
        cut = float(o["threshold"].iloc[0])
    else:
        cut = youden(o["y"].astype(int), o["prob"])
    r = confusion_row(t["y"].astype(int), t["prob"], cut)
    r["Model"] = m
    rows.append(r)

# 五变量：优先与六模型测试文件同目录的配对文件，避免跨批次错配
simple_paths = find_all("simple_score_predictions.csv")
if simple_paths:
    test_dir = os.path.dirname(test_path)
    sp = None
    same_dir = [p for p in simple_paths if os.path.dirname(p) == test_dir]
    for p in same_dir + simple_paths:
        d = pd.read_csv(p)
        if len(d) == 1069:
            sp, sp_path = d, p
            break
    if sp is None:
        sp_path = sorted(simple_paths, key=lambda x: -os.path.getsize(x))[0]
        sp = pd.read_csv(sp_path)
    print("五变量配对文件: %s" % sp_path)
    r5 = confusion_row(sp["y"].astype(int), sp["prob"], FIVE_CUTOFF)
    r5["Model"] = "Five-variable score"
    rows.append(r5)
else:
    print("[提示] 未找到 simple_score_predictions.csv，跳过五变量行。")

out = pd.DataFrame(rows)[["Model", "cutoff", "TP", "FP", "FN", "TN",
                          "Sensitivity", "Specificity", "PPV", "NPV"]]
print("\n=== 测试集固定 cut-off 混淆矩阵与 PPV ===")
print(out.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

out_path = os.path.join(SCRIPT_DIR, "B2_内部测试集PPV与混淆矩阵.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n已保存:", out_path)
