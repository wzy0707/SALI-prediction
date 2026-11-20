# =====================================================================
# 老年 ICU 脓毒症患者 SALI 机器学习预测 —— Python 建模与评估（接 R old liver.R 之后）
# 数据：R 导出的“训练集_插补未缩放.csv / 测试集_插补未缩放.csv”（Outcome 第 1 列）
# 口径：6 模型 + 嵌套交叉验证（外 10 折 / 内 5 折）；PR-AUC 主指标；无 SMOTE；
#       标准化与 one-hot 仅在训练集 fit、测试集 transform（无泄漏）；
#       SHAP 解释最佳模型；五变量简化 Logistic；校准 + HL + DCA。
# 说明：R(caret) 与 Python(sklearn) 划分引擎不同，逐例结果与云主机主流程可能略有差异，
#       但方法学一致；正文精确指标以云主机主流程为准，本脚本用于本机复现与出图。
# 首次运行需要的库（缺哪个装哪个，无需 statsmodels）：
#   pip install pandas numpy scikit-learn matplotlib scipy shap xgboost lightgbm catboost
# =====================================================================

import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (average_precision_score, roc_auc_score, brier_score_loss,
                             roc_curve, precision_recall_curve, confusion_matrix)
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("提示：未检测到 catboost，本次跳过 CatBoost。"
          "如需该模型，运行  python -m pip install catboost  后重跑。")
import shap

# ----------------------------- 0. 全局设置 -----------------------------
plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["axes.unicode_minus"] = False
SEED = 42
np.random.seed(SEED)

# 以脚本自身所在目录为基准，避免终端当前目录不同导致找不到文件
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def resolve_path(name):
    """优先脚本所在目录，找不到再退回当前工作目录"""
    p_script = os.path.join(SCRIPT_DIR, name)
    return p_script if os.path.exists(p_script) else name

# 输入文件（如改名，改这里即可）
TRAIN_FILE = resolve_path("训练集_插补未缩放.csv")
TEST_FILE = resolve_path("测试集_插补未缩放.csv")

# 输出目录（放在脚本所在目录下）
OUT_DIR = os.path.join(SCRIPT_DIR, "python old liver 输出")
FIG_DIR = os.path.join(OUT_DIR, "图")
TAB_DIR = os.path.join(OUT_DIR, "表")
for d in (OUT_DIR, FIG_DIR, TAB_DIR):
    os.makedirs(d, exist_ok=True)

# 与 R 一致的变量清单
catVars = ["Sex", "Race", "Infection source", "Positive culture",
           "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
           "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer"]
contVars = ["Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
            "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
            "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
            "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
            "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
            "NE dose", "OASIS", "SAPS II", "APS III", "LODS"]

# ----------------------------- 1. 读取并核对 -----------------------------
train = pd.read_csv(TRAIN_FILE, encoding="utf-8-sig")
test = pd.read_csv(TEST_FILE, encoding="utf-8-sig")
assert "Outcome" in train.columns
assert len(train.columns) == 51 and len(test.columns) == 51, "应为 51 列（Outcome+50 变量）"
assert set(catVars + contVars + ["Outcome"]) <= set(train.columns)
print("训练集:", train.shape, "| 测试集:", test.shape)
print("训练集事件:", int(train.Outcome.sum()), "| 测试集事件:", int(test.Outcome.sum()))

y_train = train["Outcome"].astype(int).values
y_test = test["Outcome"].astype(int).values
Xtr_raw = train.drop(columns=["Outcome"]).copy()
Xte_raw = test.drop(columns=["Outcome"]).copy()

# ----------------------------- 2. 无泄漏预处理 -----------------------------
# 2.1 分类变量 one-hot（先转为 category，再 get_dummies；训练/测试列对齐）
for v in catVars:
    Xtr_raw[v] = Xtr_raw[v].astype("category")
    Xte_raw[v] = Xte_raw[v].astype("category")
Xtr_cat = pd.get_dummies(Xtr_raw[catVars], prefix_sep="_", dummy_na=False)
Xte_cat = pd.get_dummies(Xte_raw[catVars], prefix_sep="_", dummy_na=False)
Xte_cat = Xte_cat.reindex(columns=Xtr_cat.columns, fill_value=0)

# 2.2 连续变量标准化（scaler 只在训练集 fit）
scaler = StandardScaler()
Xtr_cont = pd.DataFrame(scaler.fit_transform(Xtr_raw[contVars]),
                        columns=contVars, index=Xtr_raw.index)
Xte_cont = pd.DataFrame(scaler.transform(Xte_raw[contVars]),
                        columns=contVars, index=Xte_raw.index)

# 2.3 合并为最终建模矩阵（连续在前、one-hot 在后）
X_train = pd.concat([Xtr_cont, Xtr_cat], axis=1).astype(float)
X_test = pd.concat([Xte_cont, Xte_cat], axis=1).astype(float)
print("one-hot 后特征数:", X_train.shape[1])

# ----------------------------- 3. 模型与参数网格 -----------------------------
models = {
    "LR": (LogisticRegression(max_iter=2000, random_state=SEED),
           {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}),
    "XGBoost": (XGBClassifier(eval_metric="logloss", random_state=SEED,
                              n_jobs=1, verbosity=0),
                {"max_depth": [3, 6], "n_estimators": [100, 200],
                 "learning_rate": [0.03, 0.1]}),
    "LightGBM": (LGBMClassifier(verbosity=-1, random_state=SEED, n_jobs=1),
                 {"max_depth": [-1, 5], "n_estimators": [100, 200],
                  "learning_rate": [0.03, 0.1]}),
    "MLP": (MLPClassifier(max_iter=500, random_state=SEED),
            {"hidden_layer_sizes": [(50,), (100,), (50, 50)],
             "alpha": [0.0001, 0.001]}),
    "SVM": (SVC(probability=True, random_state=SEED),
            {"C": [0.5, 1, 10], "kernel": ["rbf", "linear"]}),
}
if HAS_CATBOOST:
    models["CatBoost"] = (CatBoostClassifier(verbose=False, random_state=SEED,
                                             allow_writing_files=False),
                          {"depth": [4, 6], "iterations": [100, 200],
                           "learning_rate": [0.03, 0.1]})

# ----------------------------- 4. 工具函数 -----------------------------
def logistic_mle(X, y, add_intercept=True, maxit=100, tol=1e-9):
    """牛顿法求 Logistic 极大似然，返回 beta、标准误（基于 Hessian）与 p 值。
    X 为 n x k 的 DataFrame/ndarray；beta 顺序含截距（若 add_intercept）。"""
    X = np.asarray(X, float)
    if add_intercept:
        X = np.column_stack([np.ones(len(X)), X])
    y = np.asarray(y, float)
    beta = np.zeros(X.shape[1])
    for _ in range(maxit):
        eta = X @ beta
        p = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        grad = X.T @ (p - y)
        W = p * (1 - p)
        hess = (X.T * W) @ X
        step = np.linalg.solve(hess, grad)
        beta = beta - step
        if np.max(np.abs(step)) < tol:
            break
    cov = np.linalg.inv(hess)
    se = np.sqrt(np.diag(cov))
    z = beta / se
    pval = 2 * (1 - stats.norm.cdf(np.abs(z)))
    return beta, se, pval

def cal_slope_intercept(y, p):
    """以 logit(p) 为自变量做无惩罚 Logistic，返回校准斜率/截距"""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))
    beta, _, _ = logistic_mle(logit.reshape(-1, 1), y, add_intercept=True)
    return beta[1], beta[0]

def hosmer_lemeshow(y, p, g=10):
    p = np.asarray(p, float)
    order = np.argsort(p)
    y_s, p_s = y[order], p[order]
    cuts = np.array_split(np.arange(len(p_s)), g)
    obs1 = exp1 = n_g = 0.0
    chi = 0.0
    for idx in cuts:
        if len(idx) == 0:
            continue
        o = y_s[idx].sum(); e = p_s[idx].sum(); nn = len(idx)
        if e > 0 and (nn - e) > 0:
            chi += (o - e) ** 2 / e + ((nn - o) - (nn - e)) ** 2 / (nn - e)
    pval = 1 - stats.chi2.cdf(chi, df=g - 2)
    return chi, pval

def youden_threshold(y, p):
    fpr, tpr, thr = roc_curve(y, p)
    j = np.argmax(tpr - fpr)
    return float(thr[j])

def all_metrics(y, p, thr):
    p = np.asarray(p, float)
    slope, intercept = cal_slope_intercept(y, p)
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if (tp + fn) else np.nan
    spec = tn / (tn + fp) if (tn + fp) else np.nan
    ppv = tp / (tp + fp) if (tp + fp) else np.nan
    npv = tn / (tn + fn) if (tn + fn) else np.nan
    return {"PR-AUC": average_precision_score(y, p),
            "ROC-AUC": roc_auc_score(y, p),
            "Brier": brier_score_loss(y, p),
            "Cal_slope": slope, "Cal_intercept": intercept,
            "Sensitivity": sens, "Specificity": spec, "PPV": ppv, "NPV": npv}

# ----------------------------- 5. 嵌套交叉验证（外10 / 内5） -----------------------------
outer = StratifiedKFold(n_splits=10, shuffle=True, random_state=SEED)
inner = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

oof_probs = {name: np.zeros(len(y_train)) for name in models}
cv_summary, best_params_final = {}, {}

for name, (base, grid) in models.items():
    fold_pr, fold_roc, fold_brier = [], [], []
    for tr_idx, va_idx in outer.split(X_train, y_train):
        gs = GridSearchCV(base, grid, scoring="average_precision", cv=inner,
                          n_jobs=-1 if name not in ("CatBoost",) else 1)
        gs.fit(X_train.iloc[tr_idx], y_train[tr_idx])
        prob = gs.best_estimator_.predict_proba(X_train.iloc[va_idx])[:, 1]
        oof_probs[name][va_idx] = prob
        fold_pr.append(average_precision_score(y_train[va_idx], prob))
        fold_roc.append(roc_auc_score(y_train[va_idx], prob))
        fold_brier.append(brier_score_loss(y_train[va_idx], prob))
    # 全训练集上再做一次内 5 折，确定最终参数
    gs_final = GridSearchCV(base, grid, scoring="average_precision", cv=inner,
                            n_jobs=-1 if name not in ("CatBoost",) else 1)
    gs_final.fit(X_train, y_train)
    best_params_final[name] = gs_final.best_params_
    cv_summary[name] = {"PR-AUC": np.mean(fold_pr), "ROC-AUC": np.mean(fold_roc),
                        "Brier": np.mean(fold_brier)}
    print(f"[嵌套CV] {name}: PR={np.mean(fold_pr):.4f} ROC={np.mean(fold_roc):.4f} "
          f"Brier={np.mean(fold_brier):.4f} | 最优参数={gs_final.best_params_}")

pd.DataFrame(cv_summary).T.to_csv(os.path.join(TAB_DIR, "嵌套CV指标.csv"),
                                  encoding="utf-8-sig")
pd.DataFrame(best_params_final).T.to_csv(os.path.join(TAB_DIR, "各模型最优参数.csv"),
                                         encoding="utf-8-sig")

# OOF 预测（长表）
oof_long = pd.DataFrame({"y": y_train})
for name, prob in oof_probs.items():
    oof_long[name] = prob
oof_long.to_csv(os.path.join(TAB_DIR, "predictions_oof.csv"), index=False,
                encoding="utf-8-sig")

# ----------------------------- 6. 全训练集重拟合 + 测试集评估 -----------------------------
fitted, test_probs = {}, {}
test_rows = []
for name, (base, grid) in models.items():
    est = base.set_params(**best_params_final[name])
    est.fit(X_train, y_train)
    fitted[name] = est
    p = est.predict_proba(X_test)[:, 1]
    test_probs[name] = p
    # 阈值由该模型 OOF 概率按 Youden 确定后固定
    thr = youden_threshold(y_train, oof_probs[name])
    m = all_metrics(y_test, p, thr)
    m["threshold"] = thr
    test_rows.append({"Model": name, **m})
    print(f"[测试集] {name}: PR={m['PR-AUC']:.4f} ROC={m['ROC-AUC']:.4f} "
          f"Brier={m['Brier']:.4f}")

test_df = pd.DataFrame(test_rows).set_index("Model")
test_df.to_csv(os.path.join(TAB_DIR, "测试集指标.csv"), encoding="utf-8-sig")

pred_test_long = pd.DataFrame({"y": y_test})
for name, p in test_probs.items():
    pred_test_long[name] = p
pred_test_long.to_csv(os.path.join(TAB_DIR, "predictions_test.csv"), index=False,
                      encoding="utf-8-sig")

# HL 检验（以最佳模型为例）
best_name = max(test_probs, key=lambda k: average_precision_score(y_test, test_probs[k]))
print("\n最佳模型:", best_name)
chi_te, p_te = hosmer_lemeshow(y_test, test_probs[best_name])
chi_oof, p_oof = hosmer_lemeshow(y_train, oof_probs[best_name])
hl_df = pd.DataFrame({"数据集": ["训练集(OOF)", "测试集"],
                      "Chi2": [chi_oof, chi_te], "df": [8, 8],
                      "p值": [p_oof, p_te]})
hl_df.to_csv(os.path.join(TAB_DIR, "HL检验.csv"), index=False, encoding="utf-8-sig")
print(hl_df.to_string(index=False))

# ----------------------------- 7. 出图：ROC / PR / 校准 / DCA -----------------------------
def plot_roc(probs, y, title_text, fname):
    plt.figure(figsize=(8, 6))
    for name, p in probs.items():
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y, p):.3f})")
    plt.plot([0, 1], [0, 1], "--", color="gray")
    plt.xlabel("1 - Specificity"); plt.ylabel("Sensitivity")
    plt.title(title_text); plt.legend(loc="lower right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=300); plt.close()

def plot_pr(probs, y, title_text, fname):
    plt.figure(figsize=(8, 6))
    for name, p in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        plt.plot(rec, prec, label=f"{name} (AP={average_precision_score(y, p):.3f})")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title(title_text); plt.legend(loc="upper right"); plt.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, fname), dpi=300); plt.close()

plot_roc(test_probs, y_test, "ROC curves - test set", "ROC_test.pdf")
plot_pr(test_probs, y_test, "Precision-Recall curves - test set", "PR_test.pdf")
plot_roc(oof_probs, y_train, "ROC curves - out-of-fold", "ROC_oof.pdf")
plot_pr(oof_probs, y_train, "Precision-Recall curves - out-of-fold", "PR_oof.pdf")

# 校准曲线（最佳模型）
plt.figure(figsize=(7, 6))
from sklearn.calibration import calibration_curve
for label, yy, pp in [("Test", y_test, test_probs[best_name]),
                      ("OOF", y_train, oof_probs[best_name])]:
    bt, bp = calibration_curve(yy, pp, n_bins=10)
    plt.plot(bp, bt, marker="o", label=label)
plt.plot([0, 1], [0, 1], "--", color="gray", label="Ideal")
plt.xlabel("Predicted probability"); plt.ylabel("Observed probability")
plt.title(f"Calibration - {best_name}"); plt.legend(); plt.grid(alpha=.3)
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "Calibration.pdf"), dpi=300)
plt.close()

# 决策曲线分析（DCA）
def dca(y, p, name):
    base = np.linspace(0.01, 0.6, 100)
    nb, nb_all = [], []
    prev = y.mean()
    for t in base:
        pred = p >= t
        tp = ((pred) & (y == 1)).sum(); fp = ((pred) & (y == 0)).sum()
        n = len(y)
        nb.append(tp / n - fp / n * (t / (1 - t)))
        nb_all.append(prev - (1 - prev) * (t / (1 - t)))
    return pd.DataFrame({"threshold": base, f"NB_{name}": nb,
                         "Treat_all": nb_all, "Treat_none": 0})

dca_frames = []
for name, p in test_probs.items():
    d = dca(y_test, p, name)
    dca_frames.append(d.rename(columns={f"NB_{name}": "NB"}).assign(Model=name))
dca_all = pd.concat(dca_frames, ignore_index=True)
dca_all.to_csv(os.path.join(TAB_DIR, "DCA_test.csv"), index=False, encoding="utf-8-sig")

plt.figure(figsize=(9, 7))
ref = dca_all[dca_all.Model == best_name]
for name, sub in dca_all.groupby("Model"):
    plt.plot(sub.threshold, sub.NB, label=name)
plt.plot(ref.threshold, ref.Treat_all, "--", color="red", label="Treat all")
plt.plot(ref.threshold, ref.Treat_none, "--", color="green", label="Treat none")
plt.xlabel("Threshold probability"); plt.ylabel("Net benefit")
plt.title("Decision curve analysis - test set"); plt.legend(fontsize=8)
plt.grid(alpha=.3); plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "DCA_test.pdf"), dpi=300); plt.close()

# ----------------------------- 8. SHAP（最佳模型，one-hot 聚合回原始变量） -----------------------------
best_model = fitted[best_name]
if best_name in ("CatBoost", "XGBoost", "LightGBM"):
    explainer = shap.TreeExplainer(best_model)
    sv = explainer.shap_values(X_test)
    sv = np.asarray(sv)
    if sv.ndim == 3:
        sv = sv[:, :, 1]
else:
    background = shap.sample(X_train, 100, random_state=SEED)
    explainer = shap.KernelExplainer(best_model.predict_proba, background)
    sv = explainer.shap_values(X_test)
    sv = np.asarray(sv)[:, :, 1]

# 聚合：one-hot 列名（如 Infection source_2）映射回原始变量
def origin_of(col):
    if col in catVars:
        return col
    for v in catVars:
        if col.startswith(v + "_"):
            return v
    return col

shap_df = pd.DataFrame(sv, columns=X_test.columns)
agg = {}
for col in shap_df.columns:
    o = origin_of(col)
    agg[o] = agg.get(o, 0) + shap_df[col].abs().mean()
shap_imp = (pd.Series(agg).sort_values(ascending=False)
            .rename("shap_importance").reset_index()
            .rename(columns={"index": "Variable"}))
shap_imp.to_csv(os.path.join(TAB_DIR, "shap_variable_importance.csv"),
                index=False, encoding="utf-8-sig")
print("\nSHAP 前 10:\n", shap_imp.head(10).to_string(index=False))

# 聚合到原始变量层面再出图（避免 one-hot 列名/下划线）
orig_names = list(catVars) + list(contVars)
enc_cols = list(X_test.columns)
sv_agg = np.zeros((sv.shape[0], len(orig_names)))
X_agg = np.zeros((sv.shape[0], len(orig_names)))
for j, v in enumerate(orig_names):
    idx = [k for k, c in enumerate(enc_cols) if origin_of(c) == v]
    sv_agg[:, j] = sv[:, idx].sum(axis=1)
    X_agg[:, j] = Xte_raw[v].astype(float).values
X_agg_df = pd.DataFrame(X_agg, columns=orig_names)

plt.figure(figsize=(10, 8))
shap.summary_plot(sv_agg, X_agg_df, feature_names=orig_names, max_display=15,
                  show=False, plot_type="dot")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "SHAP_beeswarm.pdf"),
                                dpi=300, bbox_inches="tight"); plt.close()
plt.figure(figsize=(10, 8))
shap.summary_plot(sv_agg, X_agg_df, feature_names=orig_names, max_display=15,
                  show=False, plot_type="bar")
plt.tight_layout(); plt.savefig(os.path.join(FIG_DIR, "SHAP_bar.pdf"),
                                dpi=300, bbox_inches="tight"); plt.close()

# ----------------------------- 9. 五变量简化 Logistic -----------------------------
five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]
# 标准化后的五变量（自写牛顿法求系数）
X5_tr = X_train[five]; X5_te = X_test[five]
b5_std, se5_std, _ = logistic_mle(X5_tr, y_train, add_intercept=True)
names5 = ["Intercept"] + five
coef_tbl = pd.DataFrame({"变量": names5, "标准化系数": b5_std})
coef_tbl.to_csv(os.path.join(TAB_DIR, "五变量标准化系数.csv"),
                index=False, encoding="utf-8-sig")
# 训练/测试概率（含截距）
X5_tr_i = np.column_stack([np.ones(len(X5_tr)), X5_tr.values])
X5_te_i = np.column_stack([np.ones(len(X5_te)), X5_te.values])
p5_tr = 1 / (1 + np.exp(-np.clip(X5_tr_i @ b5_std, -30, 30)))
p5 = 1 / (1 + np.exp(-np.clip(X5_te_i @ b5_std, -30, 30)))
thr5 = youden_threshold(y_train, p5_tr)
m5 = all_metrics(y_test, p5, thr5)
print("\n五变量简化模型测试集:", {k: round(v, 4) for k, v in m5.items()})
pd.DataFrame([{"Model": "Five-variable score", **m5, "threshold": thr5}]).to_csv(
    os.path.join(TAB_DIR, "五变量测试集指标.csv"), index=False, encoding="utf-8-sig")
pd.DataFrame({"y": y_test, "prob": p5}).to_csv(
    os.path.join(TAB_DIR, "simple_score_predictions.csv"), index=False,
    encoding="utf-8-sig")

# 原始尺度 OR（在未缩放、未标准化的训练数据上重拟合）
raw5 = Xtr_raw[five].astype(float)
b_raw, se_raw, p_raw = logistic_mle(raw5, y_train, add_intercept=True)
raw_names = ["Intercept"] + five
or_tbl = pd.DataFrame({
    "Variable": raw_names,
    "Beta": b_raw, "OR": np.exp(b_raw),
    "Lower_CI": np.exp(b_raw - 1.96 * se_raw),
    "Upper_CI": np.exp(b_raw + 1.96 * se_raw),
    "p_value": p_raw})
or_tbl.to_csv(os.path.join(TAB_DIR, "五变量OR表.csv"), index=False,
              encoding="utf-8-sig")
print("\n五变量原始尺度 OR:\n", or_tbl.to_string(index=False))

print("\n全部完成。结果目录:", os.path.abspath(OUT_DIR))
