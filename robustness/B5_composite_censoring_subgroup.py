# -*- coding: utf-8 -*-
# =====================================================================
# B5 · 复合结局敏感性 + 删失边界 + 分层性能（M5 / M6 / M11）
# 数据（与脚本同目录，缺哪个跳哪个，不影响其他模块）：
#   origindata.csv               MIMIC 3,567 例（含 ALT/AST/Sex/Age/Race）
#   eicu_sali_validation.csv     eICU 745 例（含 gender/age_num/hospitalid）
#   sali_ml_dataset_full.csv     6,398 超集（M6 用，本机有）
#   五变量OR表.csv               精确系数（可选）
# 运行：  C:/python310/python.exe B5_composite_censoring_subgroup.py
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else (name if os.path.exists(name) else None)


five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]
# 冻结系数：与论文 Table 6 / 外部验证一致（在 eICU 上 Brier=0.0893 精确复现）。
# 注意：不要用同名的旧版 五变量OR表.csv（其 Brier=0.1009，系另一次拟合），故此处不读取外部文件。
beta = {"Intercept": 2.937, "Baseline TBIL": 1.888, "PTT": 0.002,
        "Sodium": -0.055, "Platelets": -0.003, "Lactate": 0.139}


def preprocess(df):
    df = df.copy()
    if "PTT" in df:
        df.loc[df["PTT"] < 15, "PTT"] = np.nan
    if "Sodium" in df:
        df.loc[df["Sodium"] < 100, "Sodium"] = np.nan
    for v in five:
        if v in df and df[v].isna().any():
            df[v] = df[v].fillna(df[v].median())
    return df


def frozen_prob(df):
    lp = np.full(len(df), beta["Intercept"])
    for v in five:
        lp += beta[v] * df[v].astype(float).values
    return 1 / (1 + np.exp(-np.clip(lp, -30, 30)))


def auc_safe(y, p):
    if y.sum() < 5 or y.sum() > len(y) - 5:
        return np.nan
    return roc_auc_score(y, p)


# =====================================================================
# M5 · 复合结局敏感性（TBIL>2 且 ALT/AST 升高）
# =====================================================================
print("=" * 70, "\nM5 · 复合结局敏感性")
op_path = resolve("origindata.csv")
if op_path:
    m = preprocess(pd.read_csv(op_path))
    ALT_CUT, AST_CUT = 120, 120  # 约 3×ULN(40)，可按口径调整
    alt = m["ALT"].fillna(m["ALT"].median())
    ast = m["AST"].fillna(m["AST"].median())
    y_orig = m["Outcome"].astype(int)
    enzyme = ((alt > ALT_CUT) | (ast > AST_CUT)).astype(int)
    y_comp = (y_orig & enzyme).astype(int)
    p = frozen_prob(m)
    print("原结局事件: %d | 复合结局事件(TBIL>2 且 ALT/AST>%d): %d (%.1f%%)"
          % (y_orig.sum(), ALT_CUT, y_comp.sum(), 100 * y_comp.sum() / len(m)))
    print("胆红素-only(无酶升高)事件: %d" % (y_orig.sum() - y_comp.sum()))
    for nm, yy in [("原 bilirubin 结局", y_orig), ("复合结局", y_comp)]:
        print("  %-16s ROC %.3f | PR %.3f" % (nm, auc_safe(yy, p),
              average_precision_score(yy, p) if yy.sum() else np.nan))
    pd.DataFrame({"original_outcome": y_orig, "composite_outcome": y_comp,
                  "prob": p}).to_csv(os.path.join(SCRIPT_DIR, "B5_复合结局结果.csv"),
                                     index=False, encoding="utf-8-sig")
else:
    print("缺 origindata.csv，跳过。")

# =====================================================================
# M6 · 无随访信息性删失：worst-case 边界
# =====================================================================
print("\n" + "=" * 70, "\nM6 · 删失边界分析")
full_path = resolve("sali_ml_dataset_full.csv")
if full_path:
    full = pd.read_csv(full_path)
    N = len(full)
    followed = full[full["has_follow_7d"] == 1]
    untested = full[full["has_follow_7d"] == 1 - 1]
    n_follow, n_untested = len(followed), len(untested)
    obs_events = int(followed["label"].sum())
    obs_rate = followed["label"].mean()
    # 死亡列：优先 no_follow_death7d（无随访 2831 中的 294 例竞争死亡）
    if "no_follow_death7d" in full.columns:
        death_col = "no_follow_death7d"
    else:
        death_col = next((c for c in full.columns if any(k in c.lower()
                        for k in ["death", "expire", "mortal"])), None)
    n_death = int((untested[death_col] == 1).sum()) if death_col else 294
    n_alive_untest = n_untested - n_death
    print("总 N=%d | 有随访 %d (事件 %d, 率 %.2f%%) | 无随访 %d (死亡 %d + 存活未测 %d)"
          % (N, n_follow, obs_events, 100 * obs_rate, n_untested, n_death, n_alive_untest))

    def overall_rate(events_untested):
        return 100 * (obs_events + events_untested) / N

    scenarios = [
        ("下界：存活未测 0 事件", 0),
        ("存活未测率=观察率(%.2f%%)" % (100 * obs_rate), n_alive_untest * obs_rate),
        ("存活未测率=2×观察率", n_alive_untest * min(2 * obs_rate, 1)),
        ("绝对上界：存活未测全事件", n_alive_untest),
    ]
    print("\n全人群事件率边界（死亡 %d 例不计为 SALI）：" % n_death)
    for nm, ev in scenarios:
        print("  %-26s %.2f%%" % (nm, overall_rate(ev)))
    print("IPW 点估计（参考）：4.59%")
else:
    print("缺 sali_ml_dataset_full.csv，跳过（本机应有，确认文件名）。")

# =====================================================================
# M11 · 分层性能（eICU 性别/年龄；MIMIC 性别/年龄/种族）
# =====================================================================
print("\n" + "=" * 70, "\nM11 · 分层性能")


def age_band(a):
    if a < 75:
        return "65-74"
    if a < 85:
        return "75-84"
    return "≥85"


# eICU
ev_path = resolve("eicu_sali_validation.csv")
if ev_path:
    e = pd.read_csv(ev_path).rename(columns={"ptt": "PTT", "sodium": "Sodium",
                                            "platelets": "Platelets",
                                            "lactate": "Lactate", "outcome": "Outcome",
                                            "gender": "Sex", "age_num": "Age"})
    e = preprocess(e)
    e["prob"] = frozen_prob(e)
    e["age_band"] = e["Age"].apply(age_band)
    rows = []
    for axis, key in [("Sex", "Sex"), ("Age band", "age_band")]:
        for g, sub in e.groupby(key):
            yy = sub["Outcome"].astype(int)
            rows.append({"Cohort": "eICU", "Axis": axis, "Subgroup": str(g),
                         "n": len(sub), "events": int(yy.sum()),
                         "event_rate": yy.mean(), "ROC_AUC": auc_safe(yy, sub["prob"])})
    e_sub = pd.DataFrame(rows)
    print("--- eICU 五变量模型分层 ---")
    print(e_sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
else:
    e_sub = None
    print("缺 eicu_sali_validation.csv，跳过 eICU 分层。")

# MIMIC（五变量 frozen，全队列）
if op_path:
    m = preprocess(pd.read_csv(op_path))
    m["prob"] = frozen_prob(m)
    m["age_band"] = m["Age"].apply(age_band)
    rows = []
    # 性别（1=男）
    for g, sub in m.groupby("Sex"):
        yy = sub["Outcome"].astype(int)
        rows.append({"Cohort": "MIMIC", "Axis": "Sex",
                     "Subgroup": "Male" if g == 1 else "Female", "n": len(sub),
                     "events": int(yy.sum()), "event_rate": yy.mean(),
                     "ROC_AUC": auc_safe(yy, sub["prob"])})
    for g, sub in m.groupby("age_band"):
        yy = sub["Outcome"].astype(int)
        rows.append({"Cohort": "MIMIC", "Axis": "Age band", "Subgroup": g,
                     "n": len(sub), "events": int(yy.sum()),
                     "event_rate": yy.mean(), "ROC_AUC": auc_safe(yy, sub["prob"])})
    # 种族：取前 4 大类别，其余合并
    race_map = {0: "White", 1: "Black", 2: "Other"}
    mm = m.copy()
    mm["Race"] = mm["Race"].map(lambda x: race_map.get(x, "Other"))
    for g, sub in mm.groupby("Race"):
        yy = sub["Outcome"].astype(int)
        rows.append({"Cohort": "MIMIC", "Axis": "Race", "Subgroup": str(g),
                     "n": len(sub), "events": int(yy.sum()),
                     "event_rate": yy.mean(), "ROC_AUC": auc_safe(yy, sub["prob"])})
    m_sub = pd.DataFrame(rows)
    print("\n--- MIMIC 五变量模型分层 ---")
    print(m_sub.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    allsub = pd.concat([x for x in [e_sub, m_sub] if x is not None], ignore_index=True)
    allsub.to_csv(os.path.join(SCRIPT_DIR, "B5_分层性能.csv"), index=False,
                  encoding="utf-8-sig")
    print("\n已保存 B5_分层性能.csv")
else:
    print("缺 origindata.csv，跳过 MIMIC 分层。")

print("\nB5 完成。")
