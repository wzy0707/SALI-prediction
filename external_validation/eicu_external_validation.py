# =====================================================================
# eICU-CRD 外部验证 —— 套用 MIMIC 五变量模型（接 eicu_external_validation.sql）
# 数据：eicu_sali_validation.csv（含 Baseline TBIL/PTT/Sodium/Platelets/Lactate/Outcome）
# 方法：
#   1) 冻结模型：直接套用 MIMIC 原始尺度系数，报告 ROC/PR/Brier/校准/HL
#   2) 仅截距重校准（recalibration-in-the-large，保持斜率）
#   3) 逻辑重校准（重估截距+整体斜率）
#   区分度（ROC/PR）在三种口径下相同，差异体现在校准；这是标准外部验证做法。
# 依赖：pandas numpy scikit-learn scipy
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss

SEED = 42
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else name


five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]

# MIMIC 本机权威原始尺度系数（五变量OR表，与 R glm 一致）；若找到原表则读取精确值
beta = {"Intercept": 2.937, "Baseline TBIL": 1.888, "PTT": 0.002,
        "Sodium": -0.055, "Platelets": -0.003, "Lactate": 0.139}
or_path = resolve_path("五变量OR表.csv")
if os.path.exists(or_path):
    ort = pd.read_csv(or_path)
    for _, r in ort.iterrows():
        if r["Variable"] in beta:
            beta[r["Variable"]] = float(r["Beta"])
    print("已从五变量OR表读取精确系数。")
else:
    print("未找到五变量OR表，使用内置（保留 3 位小数）系数。")

d = pd.read_csv(resolve_path("eicu_sali_validation.csv"))
# pgAdmin 导出会把未加引号的列名折叠为小写，统一规范回标准列名
rename = {"ptt": "PTT", "sodium": "Sodium", "platelets": "Platelets",
          "lactate": "Lactate", "outcome": "Outcome"}
d = d.rename(columns={k: v for k, v in rename.items() if k in d.columns})
need = five + ["Outcome"]
miss = [c for c in need if c not in d.columns]
if miss:
    raise ValueError("eICU 数据缺少列：%s" % miss)

# eICU 原始数据含生理不可能值：PTT<15（35 例，多为 0.7-2，站点上报错误）、
# Sodium<100（18 例，9-99）。先置为缺失，再用 eICU 自身中位数插补（与开发端
# 中位数插补口径一致，不引入 MIMIC 信息）。
n_bad_ptt = int((d["PTT"] < 15).sum())
n_bad_na = int((d["Sodium"] < 100).sum())
d.loc[d["PTT"] < 15, "PTT"] = np.nan
d.loc[d["Sodium"] < 100, "Sodium"] = np.nan
n_miss = int(d[five].isna().sum().sum())
for v in five:
    if d[v].isna().any():
        d[v] = d[v].fillna(d[v].median())
print("不可能值置缺失: PTT<15 共 %d 例, Sodium<100 共 %d 例；"
      "连同原始缺失共 %d 个，已用 eICU 中位数插补。" % (n_bad_ptt, n_bad_na, n_miss))

d = d.reset_index(drop=True)
y = d["Outcome"].astype(int).values
print("eICU 外部验证: n=%d | 事件=%d | 事件率=%.2f%%"
      % (len(y), y.sum(), 100 * y.mean()))


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -30, 30)))


def linear_pred(data):
    lp = np.full(len(data), beta["Intercept"])
    for v in five:
        lp += beta[v] * data[v].astype(float).values
    return lp


def hosmer_lemeshow(yy, p, g=10):
    order = np.argsort(p)
    ys, ps = yy[order], p[order]
    chi = 0.0
    for idx in np.array_split(np.arange(len(ps)), g):
        o = ys[idx].sum(); e = ps[idx].sum(); nn = len(idx)
        if e > 0 and (nn - e) > 0:
            chi += (o - e) ** 2 / e + ((nn - o) - (nn - e)) ** 2 / (nn - e)
    return chi, 1 - stats.chi2.cdf(chi, df=g - 2)


def cal_slope_intercept(yy, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    X = np.column_stack([np.ones(len(logit)), logit])
    b = np.zeros(2)
    for _ in range(100):
        eta = X @ b
        pr = sigmoid(eta)
        W = pr * (1 - pr)
        hess = (X.T * W) @ X
        step = np.linalg.solve(hess, X.T @ (pr - yy))
        b -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return b[1], b[0]


# 1) 冻结模型
lp0 = linear_pred(d)
p0 = sigmoid(lp0)

# 2) 仅截距重校准：新截距使平均预测 = 观察患病率
prev = y.mean()
alpha_new = beta["Intercept"] + np.log(prev / (1 - prev)) - np.log(p0.mean() /
                                                                   (1 - p0.mean()))
lp1 = lp0 - beta["Intercept"] + alpha_new
p1 = sigmoid(lp1)

# 3) 逻辑重校准：y ~ 冻结线性预测
Xr = np.column_stack([np.ones(len(y)), lp0])
br = np.zeros(2)
for _ in range(100):
    eta = Xr @ br
    pr = sigmoid(eta)
    W = pr * (1 - pr)
    hess = (Xr.T * W) @ Xr
    step = np.linalg.solve(hess, Xr.T @ (pr - y))
    br -= step
    if np.max(np.abs(step)) < 1e-9:
        break
p2 = sigmoid(Xr @ br)

rows = []
for nm, p in [("冻结模型", p0), ("仅截距重校准", p1), ("逻辑重校准", p2)]:
    sl, ic = cal_slope_intercept(y, p)
    chi, hlp = hosmer_lemeshow(y, p)
    rows.append({"Method": nm, "ROC_AUC": roc_auc_score(y, p),
                 "PR_AUC": average_precision_score(y, p),
                 "Brier": brier_score_loss(y, p), "Slope": sl,
                 "Intercept": ic, "HL_p": hlp})
    print("[%s] ROC %.3f PR %.3f Brier %.4f 斜率 %.3f 截距 %.3f HL p=%.3f"
          % (nm, roc_auc_score(y, p), average_precision_score(y, p),
             brier_score_loss(y, p), sl, ic, hlp))

out = pd.DataFrame(rows)
out_path = os.path.join(SCRIPT_DIR, "eicu外部验证结果.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
pd.DataFrame({"y": y, "prob_frozen": p0, "prob_intercept": p1,
              "prob_logistic": p2}).to_csv(
    os.path.join(SCRIPT_DIR, "eicu_predictions.csv"), index=False,
    encoding="utf-8-sig")
print("\n结果已保存:", out_path)
