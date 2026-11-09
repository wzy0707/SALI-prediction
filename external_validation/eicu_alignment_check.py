# =====================================================================
# eICU 外部验证 —— 对齐检查（只读，不修改原文件）
# 作用：在跑 eicu_external_validation.py 前，确认 eICU 数据与 MIMIC 的
#       变量名、单位尺度、时间窗、结局口径一致，并标出不可能值。
# 运行：把本脚本与 eicu_sali_validation.csv 放同一目录，
#       C:/python310/python.exe eicu_alignment_check.py
# =====================================================================
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
path = os.path.join(SCRIPT_DIR, "eicu_sali_validation.csv")
if not os.path.exists(path):
    path = "eicu_sali_validation.csv"

d = pd.read_csv(path)

# 1) 列名统一：pgAdmin 把未加引号列名折叠为小写，这里规范回标准名
rename = {"ptt": "PTT", "sodium": "Sodium", "platelets": "Platelets",
          "lactate": "Lactate", "outcome": "Outcome"}
d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
print("当前列名:", list(d.columns), "\n")

five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]

# 2) MIMIC 参考范围（origindata，原始单位），用于判断单位尺度是否一致
#    单位：TBIL mg/dL；PTT 秒；Sodium mEq/L；Platelets 10^9/L；Lactate mmol/L
ref = {
    "Baseline TBIL": (0.1, 0.60, 2.0),
    "PTT":           (18.5, 35.40, 150.0),
    "Sodium":         (117.0, 140.0, 179.0),
    "Platelets":      (5.0, 175.0, 1050.0),
    "Lactate":        (0.0, 2.30, 22.0),
}

# 3) 必含列检查
need = five + ["Outcome"]
missing_cols = [c for c in need if c not in d.columns]
print("===== 检查 1：必含变量与结局 =====")
print("PASS：五个变量 + Outcome 齐全" if not missing_cols
      else "FAIL：缺少 %s" % missing_cols)

# 4) 取值只能是 0/1
print("\n===== 检查 2：结局口径 =====")
ov = set(d["Outcome"].dropna().unique())
print("Outcome 取值:", ov,
      "| n=%d 事件=%d 事件率=%.2f%%"
      % (len(d), d["Outcome"].sum(), 100 * d["Outcome"].mean()))
print("PASS：结局为 0/1（1=24h-7d 新发 TBIL>2）" if ov <= {0, 1}
      else "FAIL：结局出现非 0/1 值")

# 5) 单位尺度对照 + 不可能值
print("\n===== 检查 3：单位尺度与分布对照（MIMIC 参考）=====")
print("%-13s %-28s %-28s %-8s" % ("变量", "eICU(min/中位/max)", "MIMIC(min/中位/max)", "缺失"))
for c in five:
    em = (d[c].min(), d[c].median(), d[c].max())
    rm = ref[c]
    print("%-13s %-28s %-28s %-8d"
          % (c,
             "%.2f / %.2f / %.2f" % em,
             "%.2f / %.2f / %.2f" % rm,
             d[c].isna().sum()))

print("\n===== 检查 4：生理不可能值（需置缺失后中位数插补）=====")
bad_ptt = d[d["PTT"] < 15]
bad_na = d["Sodium"] < 100
print("PTT<15：%d 例（多为 0.7-2，站点上报错误）" % len(bad_ptt))
print("Sodium<100：%d 例（9-99，不可能）" % int(bad_na.sum()))
print("说明：验证脚本会自动把这些置为缺失，并用 eICU 自身中位数插补。")

# 6) 单位尺度判断：若 eICU 中位数与 MIMIC 中位数差一个数量级以上，疑似单位不一致
print("\n===== 检查 5：单位是否同一量级 =====")
ok = True
for c in five:
    ratio = d[c].median() / ref[c][1] if ref[c][1] != 0 else 1
    if not (0.2 < ratio < 5):
        ok = False
        print("WARNING：%s 中位数与 MIMIC 差异过大（%.2f 倍），疑似单位/口径不一致，"
              "先核对单位再验证！" % (c, ratio))
print("PASS：五个变量与 MIMIC 同一量级，单位尺度一致（无需标准化）。" if ok
      else "请先处理上述单位问题。")

print("\n===== 结论 =====")
if not missing_cols and ov <= {0, 1} and ok:
    print("对齐检查通过。下一步运行：C:/python310/python.exe eicu_external_validation.py")
else:
    print("存在需先处理的问题，见上方 FAIL/WARNING。")
