# =====================================================================
# 老年 ICU 脓毒症患者 SALI 预测 —— Python 侧六模型评估（全真矢量 PDF）
# 覆盖：ROC / PR / SHAP 柱状 / SHAP 蜂群 / CatBoost 平滑校准 / 决策曲线 DCA
# 读取 R 脚本导出的 “训练集_插补未缩放.csv / 测试集_插补未缩放.csv”（与论文同一次划分）
# 全部输出到脚本同目录的 “矢量PDF输出” 文件夹，均为矢量（可无限缩放、文字可选中）
# 运行顺序：先 Source 运行 SALI_全部图片_R矢量PDF.R（会生成划分文件），再在 PowerShell 运行：
#   C:/python310/python.exe SALI_模型评估_Python矢量PDF.py
# 依赖：pandas numpy scikit-learn matplotlib scipy shap xgboost lightgbm catboost
# =====================================================================

import os
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             roc_curve, precision_recall_curve)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import shap

# ---------------- 0. 全局设置 ----------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
np.random.seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(SCRIPT_DIR, "矢量PDF输出")
os.makedirs(OUT, exist_ok=True)

catVars = ["Sex", "Race", "Infection source", "Positive culture",
           "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
           "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer"]
contVars = ["Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
            "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
            "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
            "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
            "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
            "NE dose", "OASIS", "SAPS II", "APS III", "LODS"]

# ---------------- 1. 读取 R 导出的划分文件 ----------------
train = pd.read_csv(os.path.join(SCRIPT_DIR, "训练集_插补未缩放.csv"), encoding="utf-8-sig")
test = pd.read_csv(os.path.join(SCRIPT_DIR, "测试集_插补未缩放.csv"), encoding="utf-8-sig")
assert len(train.columns) == 51 and len(test.columns) == 51
y_train = train["Outcome"].astype(int).values
y_test = test["Outcome"].astype(int).values
Xtr_raw = train.drop(columns=["Outcome"]).copy()
Xte_raw = test.drop(columns=["Outcome"]).copy()
print("训练", Xtr_raw.shape, "事件", y_train.sum(), "| 测试", Xte_raw.shape, "事件", y_test.sum())

# ---------------- 2. 无泄漏预处理（one-hot + 标准化） ----------------
for v in catVars:
    Xtr_raw[v] = Xtr_raw[v].astype("category")
    Xte_raw[v] = Xte_raw[v].astype("category")
Xtr_cat = pd.get_dummies(Xtr_raw[catVars], prefix_sep="_")
Xte_cat = pd.get_dummies(Xte_raw[catVars], prefix_sep="_")
Xte_cat = Xte_cat.reindex(columns=Xtr_cat.columns, fill_value=0)

scaler = StandardScaler()
Xtr_cont = pd.DataFrame(scaler.fit_transform(Xtr_raw[contVars]),
                        columns=contVars, index=Xtr_raw.index)
Xte_cont = pd.DataFrame(scaler.transform(Xte_raw[contVars]),
                        columns=contVars, index=Xte_raw.index)

X_train = pd.concat([Xtr_cont, Xtr_cat], axis=1).astype(float)
X_test = pd.concat([Xte_cont, Xte_cat], axis=1).astype(float)
print("one-hot 后特征数:", X_train.shape[1])

# ---------------- 3. 六模型（锁定最优参数） ----------------
models = {
    "LR": LogisticRegression(C=0.01, penalty="l2", max_iter=2000, random_state=SEED),
    "XGBoost": XGBClassifier(learning_rate=0.03, max_depth=3, n_estimators=100,
                             eval_metric="logloss", random_state=SEED, n_jobs=1, verbosity=0),
    "LightGBM": LGBMClassifier(learning_rate=0.03, max_depth=5, n_estimators=100,
                               verbosity=-1, random_state=SEED, n_jobs=1),
    "MLP": MLPClassifier(alpha=0.001, hidden_layer_sizes=(100,), max_iter=500,
                         random_state=SEED),
    "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
    "CatBoost": CatBoostClassifier(learning_rate=0.03, depth=6, iterations=200,
                                   verbose=False, random_state=SEED,
                                   allow_writing_files=False),
}
fitted, probs = {}, {}
for name, m in models.items():
    m.fit(X_train, y_train); fitted[name] = m
    probs[name] = m.predict_proba(X_test)[:, 1]
    print(f"{name:9s} ROC={roc_auc_score(y_test, probs[name]):.4f} "
          f"PR={average_precision_score(y_test, probs[name]):.4f} "
          f"Brier={brier_score_loss(y_test, probs[name]):.4f}")

# ---------------- 4. ROC（矢量 PDF） ----------------
plt.figure(figsize=(8, 6))
for name, p in probs.items():
    fpr, tpr, _ = roc_curve(y_test, p)
    plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_test, p):.3f})")
plt.plot([0, 1], [0, 1], "--", color="gray")
plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
plt.title("ROC curves - test set"); plt.legend(loc="lower right"); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(OUT, "ROC_test.pdf")); plt.close()

# ---------------- 5. PR（矢量 PDF） ----------------
plt.figure(figsize=(8, 6))
for name, p in probs.items():
    prec, rec, _ = precision_recall_curve(y_test, p)
    plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y_test, p):.3f})")
plt.xlabel("Recall"); plt.ylabel("Precision")
plt.title("Precision-Recall curves - test set"); plt.legend(loc="upper right")
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "PR_test.pdf")); plt.close()

# ---------------- 6. 决策曲线 DCA（矢量 PDF） ----------------
def dca(y, p):
    base = np.linspace(0.01, 0.6, 100); prev = y.mean(); n = len(y)
    nb, nb_all = [], []
    for t in base:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        nb.append(tp / n - fp / n * (t / (1 - t)))
        nb_all.append(prev - (1 - prev) * (t / (1 - t)))
    return base, np.array(nb), np.array(nb_all)

plt.figure(figsize=(9, 7))
for name, p in probs.items():
    base, nb, _ = dca(y_test, p)
    plt.plot(base, nb, label=name)
base, _, nb_all = dca(y_test, probs["CatBoost"])
plt.plot(base, nb_all, "--", color="red", label="Treat all")
plt.axhline(0, linestyle="--", color="green", label="Treat none")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Decision curve analysis - test set"); plt.legend(fontsize=8)
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(OUT, "DCA_test.pdf")); plt.close()

# ---------------- 7. SHAP（CatBoost，one-hot 聚合回原始变量） ----------------
def origin_of(col):
    if col in catVars: return col
    for v in catVars:
        if col.startswith(v + "_"): return v
    return col

explainer = shap.TreeExplainer(fitted["CatBoost"])
sv = np.asarray(explainer.shap_values(X_test))
if sv.ndim == 3: sv = sv[:, :, 1]

orig_names = list(catVars) + list(contVars)
enc_cols = list(X_test.columns)
sv_agg = np.zeros((sv.shape[0], len(orig_names)))
X_agg = np.zeros((sv.shape[0], len(orig_names)))
for j, v in enumerate(orig_names):
    idx = [k for k, c in enumerate(enc_cols) if origin_of(c) == v]
    sv_agg[:, j] = sv[:, idx].sum(axis=1)
    X_agg[:, j] = Xte_raw[v].astype(float).values
X_agg_df = pd.DataFrame(X_agg, columns=orig_names)

# 自定义全真矢量蜂群（shap 自带 beeswarm 会把点栅格化，这里用 scatter 显式 rasterized=False）
def vector_beeswarm(svmat, Xmat, names, max_display=15, fname="SHAP_beeswarm.pdf"):
    mean_abs = np.abs(svmat).mean(axis=0)
    order = np.argsort(mean_abs)[::-1][:max_display]
    fig, ax = plt.subplots(figsize=(10, 8))
    rng = np.random.default_rng(SEED)
    for yi, j in enumerate(order):
        x = svmat[:, j]
        rk = pd.Series(Xmat[:, j]).rank(pct=True).values   # 变量内秩归一化，颜色铺满
        # 固定行带内随机抖动，所有点不越出本行（±0.36），保证各行分明
        yj = yi + rng.uniform(-0.36, 0.36, size=len(x))
        ax.scatter(x, yj, c=rk, cmap="bwr", s=13, edgecolors="none",
                   alpha=.9, rasterized=False)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([names[j] for j in order])
    ax.invert_yaxis()
    ax.axvline(0, color="gray", lw=.7, alpha=.5)
    ax.set_xlabel("SHAP value (impact on model output)")
    # 用离散矢量矩形拼色条，避免连续 colorbar 被栅格化
    fig.subplots_adjust(right=0.88)
    cax = fig.add_axes([0.905, 0.12, 0.022, 0.76])
    nseg = 100
    for k in range(nseg):
        cax.barh(k, 1, height=1, color=plt.cm.bwr(k / (nseg - 1)), edgecolor="none")
    cax.set_xlim(0, 1); cax.set_xticks([])
    cax.set_yticks([0, nseg - 1]); cax.set_yticklabels(["Low", "High"])
    cax.set_ylabel("Feature value")
    plt.savefig(os.path.join(OUT, fname)); plt.close()

vector_beeswarm(sv_agg, X_agg, orig_names)

plt.figure(figsize=(10, 8))
shap.summary_plot(sv_agg, X_agg_df, feature_names=orig_names, max_display=15,
                  show=False, plot_type="bar")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "SHAP_bar.pdf"),
                                bbox_inches="tight"); plt.close()

# ---------------- 8. CatBoost 平滑校准（分位数分箱 + 等渗 + 样条 + 自举 CI） ----------------
from scipy.interpolate import make_interp_spline
p_cb = probs["CatBoost"]
N_BINS = 10

def binned_curve(yy, pp):
    order = np.argsort(pp)
    yy_s, pp_s = yy[order], pp[order]
    bx, by = [], []
    for idx in np.array_split(np.arange(len(pp_s)), N_BINS):
        bx.append(pp_s[idx].mean()); by.append(yy_s[idx].mean())
    bx, by = np.array(bx), np.array(by)
    yi = IsotonicRegression(out_of_bounds="clip").fit_transform(bx, by)
    return bx, by, yi

def smooth_monotone(bxb, yib, g):
    kk = 2 if len(bxb) > 2 else len(bxb) - 1
    s = make_interp_spline(bxb, yib, k=kk)(g)
    return IsotonicRegression(out_of_bounds="clip").fit_transform(g, np.clip(s, 0, 1))

bx, by, yi = binned_curve(y_test, p_cb)
grid = np.linspace(bx.min(), bx.max(), 200)   # 仅在真实分箱范围内，不外推
smooth = smooth_monotone(bx, yi, grid)

rng = np.random.default_rng(SEED)
boots = np.zeros((1000, len(grid)))
for b in range(1000):
    idx = rng.integers(0, len(y_test), len(y_test))
    bxb, _, yib = binned_curve(y_test[idx], p_cb[idx])
    g_b = np.linspace(bxb.min(), bxb.max(), len(grid))
    boots[b] = smooth_monotone(bxb, yib, g_b)
lo, hi = np.percentile(boots, 2.5, axis=0), np.percentile(boots, 97.5, axis=0)

ymax = max(by) * 1.15
fig, ax = plt.subplots(figsize=(7.5, 6.5))
ax.fill_between(grid, np.clip(lo, 0, ymax), np.clip(hi, 0, ymax),
                color="#9ecae1", alpha=.5, label="95% CI")
ax.plot(grid, smooth, color="#08519c", lw=2.2, label="Smoothed calibration")
ax.plot(bx, by, "o", color="#08519c", ms=6, alpha=.8)
ax.plot([0, bx.max() * 1.05], [0, bx.max() * 1.05], "--", color="gray",
        lw=1.5, label="Ideal")
ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed probability")
ax.set_title("Calibration curve - CatBoost (test set)")
ax.set_xlim(0, bx.max() * 1.08); ax.set_ylim(0, ymax)
ax.legend(loc="upper left", frameon=False); ax.grid(alpha=.25)
ax2 = ax.inset_axes([0, -0.16, 1, 0.13])
ax2.hist(p_cb, bins=40, color="#bdbdbd", edgecolor="white")
ax2.set_yticks([]); ax2.set_xlim(ax.get_xlim()); ax2.axis("off")
plt.tight_layout(); plt.savefig(os.path.join(OUT, "CatBoost校准.pdf")); plt.close()

print("\n全部完成，矢量 PDF 位于：", os.path.abspath(OUT))
