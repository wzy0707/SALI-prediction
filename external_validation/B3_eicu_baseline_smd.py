# -*- coding: utf-8 -*-
# =====================================================================
# B3 · eICU 外部队列 vs MIMIC 开发队列：基线特征表 + 标准化均值差（SMD）
# 对应审稿意见：M3（缺外部队列基线表，无法判断两库可比性）
#
# 数据（与脚本同目录，文件名可在下方修改）：
#   MIMIC：origindata.csv（3,567 例，含五变量 + Age/Sex）
#   eICU ：eicu_sali_validation.csv（745 例）
# 运行：  C:/python310/python.exe B3_eicu_baseline_smd.py
# 判读：  |SMD| <0.10 视为差异可忽略；0.10–0.20 小差异；>0.20 明显差异
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MIMIC_FILE = "origindata.csv"
EICU_FILE = "eicu_sali_validation.csv"


def resolve_path(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else name


# ---- 读取 ----
mim = pd.read_csv(resolve_path(MIMIC_FILE))
eicu = pd.read_csv(resolve_path(EICU_FILE))
eicu = eicu.rename(columns={"ptt": "PTT", "sodium": "Sodium",
                            "platelets": "Platelets", "lactate": "Lactate",
                            "outcome": "Outcome", "gender": "Sex",
                            "age_num": "Age"})

# eICU 复刻外验预处理（不可能值→中位数）
eicu.loc[eicu["PTT"] < 15, "PTT"] = np.nan
eicu.loc[eicu["Sodium"] < 100, "Sodium"] = np.nan
five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]
for v in five:
    if eicu[v].isna().any():
        eicu[v] = eicu[v].fillna(eicu[v].median())

print("MIMIC n=%d (事件率 %.2f%%) | eICU n=%d (事件率 %.2f%%)"
      % (len(mim), 100 * mim["Outcome"].mean(), len(eicu),
         100 * eicu["Outcome"].mean()))


def smd_continuous(x1, x2):
    x1, x2 = np.asarray(x1, float), np.asarray(x2, float)
    v1, v2 = np.nanvar(x1, ddof=1), np.nanvar(x2, ddof=1)
    pooled = np.sqrt((v1 + v2) / 2)
    return (np.nanmean(x1) - np.nanmean(x2)) / pooled if pooled > 0 else np.nan


def smd_binary(p1, p2):
    p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
    pooled = np.sqrt((p1 * (1 - p1) + p2 * (1 - p2)) / 2)
    return (p1 - p2) / pooled if pooled > 0 else np.nan


def fmt_med_iqr(x):
    x = np.asarray(x, float)
    q1, q2, q3 = np.nanpercentile(x, [25, 50, 75])
    return "%.2f [%.2f–%.2f]" % (q2, q1, q3)


rows = []
# 连续变量（含五变量 + 年龄）
cont_map = [("Age", "Age (years)"), ("Baseline TBIL", "Baseline TBIL (mg/dL)"),
            ("PTT", "PTT (s)"), ("Sodium", "Sodium (mEq/L)"),
            ("Platelets", "Platelets (×10⁹/L)"), ("Lactate", "Lactate (mmol/L)")]
for col, label in cont_map:
    smd = smd_continuous(mim[col], eicu[col])
    rows.append({"Variable": label, "Type": "continuous",
                 "MIMIC (n=%d)" % len(mim): fmt_med_iqr(mim[col]),
                 "eICU (n=%d)" % len(eicu): fmt_med_iqr(eicu[col]),
                 "SMD": smd})

# 分类：性别（统一为 Male 比例）
def male_prop(s):
    s = pd.Series(s)
    # MIMIC 为数字编码（1=男,0=女）；eICU 为字符串
    if pd.api.types.is_numeric_dtype(s):
        return (s.astype(int) == 1).mean()
    s = s.astype(str).str.strip().str.lower()
    return (s == "male").mean()


m_male, e_male = male_prop(mim["Sex"]), male_prop(eicu["Sex"])
rows.append({"Variable": "Male sex, n (%)", "Type": "categorical",
             "MIMIC (n=%d)" % len(mim): "%d (%.1f%%)" % (round(m_male * len(mim)), 100 * m_male),
             "eICU (n=%d)" % len(eicu): "%d (%.1f%%)" % (round(e_male * len(eicu)), 100 * e_male),
             "SMD": smd_binary(m_male, e_male)})

out = pd.DataFrame(rows)
pd.set_option("display.width", 200)
print("\n=== eICU vs MIMIC 基线特征与 SMD ===")
print(out.to_string(index=False, float_format=lambda x: f"{x: .3f}"))

# 汇总提示
big = out[np.abs(out["SMD"]) > 0.10]
print("\n|SMD| > 0.10 的变量:", list(big["Variable"]) if len(big) else "无")

out_path = os.path.join(SCRIPT_DIR, "B3_eicu基线表_SMD.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print("已保存:", out_path)
