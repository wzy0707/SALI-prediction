# =====================================================================
# SALI 退修 —— 敏感性分析（接 R old liver.R 与 python old liver.py 之后）
# 数据：R 导出的“训练集_插补未缩放.csv / 测试集_插补未缩放.csv”（Outcome 第 1 列）
# 内容：
#   B. 单用基线 TBIL 模型 vs 全模型 DeLong 增量
#   C. 五变量 logistic vs CatBoost DeLong 比较
#   A. 低基线 TBIL(<=1.5) 人群：冻结主分析最优参数，10 折 CV 重跑
#   D. CatBoost 交叉验证内 Platt / isotonic 重校准（内层 OOF 拟合校准器）
#   E. 持续事件（可选；读取 sali_event_labels.csv，需 R 导出含 stay_id）
# 首次运行需要的库：pandas numpy scikit-learn scipy xgboost lightgbm catboost
# =====================================================================
import os
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from scipy import stats

from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    print("提示：未检测到 catboost，涉及 CatBoost 的部分将跳过。pip install catboost")

SEED = 42
np.random.seed(SEED)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def resolve_path(name):
    p = os.path.join(SCRIPT_DIR, name)
    return p if os.path.exists(p) else name


OUT_DIR = os.path.join(SCRIPT_DIR, "敏感性分析输出")
os.makedirs(OUT_DIR, exist_ok=True)

catVars = ["Sex", "Race", "Infection source", "Positive culture",
           "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
           "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer"]
contVars = ["Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
            "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
            "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
            "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
            "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
            "NE dose", "OASIS", "SAPS II", "APS III", "LODS"]


def preprocess(train, test):
    ytr = train["Outcome"].astype(int).values
    yte = test["Outcome"].astype(int).values
    Rtr = train.drop(columns=["Outcome"]).copy()
    Rte = test.drop(columns=["Outcome"]).copy()
    for v in catVars:
        Rtr[v] = Rtr[v].astype("category")
        Rte[v] = Rte[v].astype("category")
    Ctr = pd.get_dummies(Rtr[catVars], prefix_sep="_", dummy_na=False)
    Cte = pd.get_dummies(Rte[catVars], prefix_sep="_", dummy_na=False)
    Cte = Cte.reindex(columns=Ctr.columns, fill_value=0)
    sc = StandardScaler()
    Str = pd.DataFrame(sc.fit_transform(Rtr[contVars]), columns=contVars,
                       index=Rtr.index)
    Ste = pd.DataFrame(sc.transform(Rte[contVars]), columns=contVars,
                       index=Rte.index)
    Xtr = pd.concat([Str, Ctr], axis=1).astype(float)
    Xte = pd.concat([Ste, Cte], axis=1).astype(float)
    return Xtr, ytr, Xte, yte, Rtr, Rte


def fixed_estimators():
    """主分析已确定的最优参数，敏感性分析中冻结，避免重复 GridSearch。"""
    m = {
        "LR": LogisticRegression(max_iter=2000, C=0.01, penalty="l2",
                                 random_state=SEED),
        "XGBoost": XGBClassifier(learning_rate=0.03, max_depth=3,
                                 n_estimators=100, eval_metric="logloss",
                                 random_state=SEED, n_jobs=1, verbosity=0),
        "LightGBM": LGBMClassifier(learning_rate=0.03, max_depth=5,
                                   n_estimators=100, verbosity=-1,
                                   random_state=SEED, n_jobs=1),
        "MLP": MLPClassifier(alpha=0.001, hidden_layer_sizes=(100,),
                             max_iter=500, random_state=SEED),
        "SVM": SVC(C=1, kernel="rbf", probability=True, random_state=SEED),
    }
    if HAS_CATBOOST:
        m["CatBoost"] = CatBoostClassifier(iterations=200, depth=6,
                                           learning_rate=0.03, verbose=False,
                                           random_state=SEED,
                                           allow_writing_files=False)
    return m


def new_cb():
    return CatBoostClassifier(iterations=200, depth=6, learning_rate=0.03,
                              verbose=False, random_state=SEED,
                              allow_writing_files=False)


def hosmer_lemeshow(y, p, g=10):
    order = np.argsort(p)
    ys, ps = y[order], p[order]
    chi = 0.0
    for idx in np.array_split(np.arange(len(ps)), g):
        o = ys[idx].sum(); e = ps[idx].sum(); nn = len(idx)
        if e > 0 and (nn - e) > 0:
            chi += (o - e) ** 2 / e + ((nn - o) - (nn - e)) ** 2 / (nn - e)
    return chi, 1 - stats.chi2.cdf(chi, df=g - 2)


def cal_slope_intercept(y, p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p)).reshape(-1, 1)
    X = np.column_stack([np.ones(len(logit)), logit])
    beta = np.zeros(2)
    for _ in range(100):
        eta = X @ beta
        pr = 1 / (1 + np.exp(-np.clip(eta, -30, 30)))
        W = pr * (1 - pr)
        hess = (X.T * W) @ X
        step = np.linalg.solve(hess, X.T @ (pr - y))
        beta -= step
        if np.max(np.abs(step)) < 1e-9:
            break
    return beta[1], beta[0]


# ---------------- DeLong ----------------
def placement_values(pos, neg):
    """pos: m x K, neg: n x K。返回阳性放置值 V10(m x K)、阴性放置值 V01(n x K)。"""
    V10 = np.zeros_like(pos)
    V01 = np.zeros_like(neg)
    for k in range(pos.shape[1]):
        ps = np.sort(pos[:, k]); negs = np.sort(neg[:, k])
        left = np.searchsorted(negs, pos[:, k], side="left")
        right = np.searchsorted(negs, pos[:, k], side="right")
        V10[:, k] = (left + 0.5 * (right - left)) / len(negs)
        lp = np.searchsorted(ps, neg[:, k], side="left")
        rp = np.searchsorted(ps, neg[:, k], side="right")
        greater = len(ps) - rp
        equal = rp - lp
        V01[:, k] = (greater + 0.5 * equal) / len(ps)
    return V10, V01


def fast_delong(pred, y, n1):
    order = np.argsort(-y)
    pred = pred[order]
    pos = pred[:n1]; neg = pred[n1:]
    m, n = pos.shape[0], neg.shape[0]
    V10, V01 = placement_values(pos, neg)
    auc = V10.mean(axis=0)
    s10 = np.cov(V10, rowvar=False)
    s01 = np.cov(V01, rowvar=False)
    return auc, s10 / m + s01 / n


def delong_compare(y, p1, p2):
    n1 = int(np.sum(y))
    auc, cov = fast_delong(np.column_stack([p1, p2]), y, n1)
    var = cov[0, 0] + cov[1, 1] - 2 * cov[0, 1]
    z = (auc[0] - auc[1]) / np.sqrt(var)
    return auc[0], auc[1], z, 2 * (1 - stats.norm.cdf(abs(z)))


# ---------------- 读取 ----------------
train = pd.read_csv(resolve_path("训练集_插补未缩放.csv"), encoding="utf-8-sig")
test = pd.read_csv(resolve_path("测试集_插补未缩放.csv"), encoding="utf-8-sig")
X_train, y_train, X_test, y_test, Rtr, Rte = preprocess(train, test)
print("训练", X_train.shape, "| 测试", X_test.shape, "| 特征", X_train.shape[1])

# ===== 先做便宜的 DeLong：全模型 CatBoost 拟合一次 =====
p_cat = None
if HAS_CATBOOST:
    cb = new_cb(); cb.fit(X_train, y_train)
    p_cat = cb.predict_proba(X_test)[:, 1]

# B. 单用 TBIL
print("\n=== B. 单用基线 TBIL 的增量 ===")
uni = LogisticRegression(max_iter=2000, C=1.0)
uni.fit(X_train[["Baseline TBIL"]], y_train)
p_tbil = uni.predict_proba(X_test[["Baseline TBIL"]])[:, 1]
if p_cat is not None:
    a1, a2, z, pv = delong_compare(y_test, p_cat, p_tbil)
    print("CatBoost AUC %.3f vs 单用TBIL AUC %.3f | z=%.3f p=%.4f"
          % (a1, a2, z, pv))
    pd.DataFrame([{"CatBoost_AUC": a1, "TBILonly_AUC": a2, "z": z,
                   "p": pv}]).to_csv(os.path.join(OUT_DIR, "TBIL增量DeLong.csv"),
                                     index=False, encoding="utf-8-sig")

# C. 五变量 vs CatBoost
print("\n=== C. 五变量 vs CatBoost ===")
five = ["Baseline TBIL", "PTT", "Sodium", "Platelets", "Lactate"]
lr5 = LogisticRegression(max_iter=2000, C=1.0)
lr5.fit(X_train[five], y_train)
p5 = lr5.predict_proba(X_test[five])[:, 1]
if p_cat is not None:
    a1, a2, z, pv = delong_compare(y_test, p_cat, p5)
    print("CatBoost AUC %.3f vs 五变量 AUC %.3f | z=%.3f p=%.4f"
          % (a1, a2, z, pv))
    pd.DataFrame([{"CatBoost_AUC": a1, "FiveVar_AUC": a2, "z": z,
                   "p": pv}]).to_csv(os.path.join(OUT_DIR, "模型比较DeLong.csv"),
                                     index=False, encoding="utf-8-sig")

# A. 低基线
print("\n=== A. 低基线 TBIL<=1.5 人群 ===")
mk_tr = (Rtr["Baseline TBIL"] <= 1.5).values
mk_te = (Rte["Baseline TBIL"] <= 1.5).values
Xl_tr, yl_tr = X_train[mk_tr], y_train[mk_tr]
Xl_te, yl_te = X_test[mk_te], y_test[mk_te]
print("低基线: 训练 %d(%d事件) | 测试 %d(%d事件)"
      % (len(yl_tr), yl_tr.sum(), len(yl_te), yl_te.sum()))
kf10 = StratifiedKFold(10, shuffle=True, random_state=SEED)
low_rows = []
for name, est in fixed_estimators().items():
    cvr, cvc = [], []
    for tr_i, va_i in kf10.split(Xl_tr, yl_tr):
        m = est.__class__(**est.get_params())
        m.fit(Xl_tr.iloc[tr_i], yl_tr[tr_i])
        pr = m.predict_proba(Xl_tr.iloc[va_i])[:, 1]
        cvr.append(roc_auc_score(yl_tr[va_i], pr))
        cvc.append(average_precision_score(yl_tr[va_i], pr))
    mf = est.__class__(**est.get_params()); mf.fit(Xl_tr, yl_tr)
    pte = mf.predict_proba(Xl_te)[:, 1]
    low_rows.append({"Model": name, "CV_ROC": np.mean(cvr), "CV_PR": np.mean(cvc),
                     "Test_ROC": roc_auc_score(yl_te, pte),
                     "Test_PR": average_precision_score(yl_te, pte)})
    print("[低基线] %s: CV ROC %.3f | Test ROC %.3f / PR %.3f"
          % (name, np.mean(cvr), roc_auc_score(yl_te, pte),
             average_precision_score(yl_te, pte)))
pd.DataFrame(low_rows).to_csv(os.path.join(OUT_DIR, "低基线敏感性.csv"),
                              index=False, encoding="utf-8-sig")

# D. CV 内重校准（内层 OOF 拟合单一校准器，外层验证）
print("\n=== D. CatBoost 交叉验证内重校准 ===")
if HAS_CATBOOST:
    kf5 = StratifiedKFold(5, shuffle=True, random_state=SEED)
    kin = StratifiedKFold(5, shuffle=True, random_state=SEED)
    n = len(y_train)
    pu = np.zeros(n); pp = np.zeros(n); pi = np.zeros(n)
    for tr_i, va_i in kf5.split(X_train, y_train):
        Xa, ya = X_train.iloc[tr_i], y_train[tr_i]
        m0 = new_cb(); m0.fit(Xa, ya)
        pu[va_i] = m0.predict_proba(X_train.iloc[va_i])[:, 1]
        # 内层 OOF 预测
        inner_oof = np.zeros(len(ya))
        for itr, iva in kin.split(Xa, ya):
            mi = new_cb(); mi.fit(Xa.iloc[itr], ya[itr])
            inner_oof[iva] = mi.predict_proba(Xa.iloc[iva])[:, 1]
        # Platt：对内层 OOF 概率做逻辑校准
        lg = LogisticRegression(C=1e6, max_iter=2000)
        lg.fit(np.clip(inner_oof, 1e-6, 1 - 1e-6).reshape(-1, 1), ya)
        pp[va_i] = lg.predict_proba(np.clip(pu[va_i], 1e-6, 1 - 1e-6)
                                     .reshape(-1, 1))[:, 1]
        # Isotonic
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(inner_oof, ya)
        pi[va_i] = iso.predict(pu[va_i])
    cal_rows = []
    for nm, ppred in [("未校准", pu), ("Platt", pp), ("Isotonic", pi)]:
        sl, ic = cal_slope_intercept(y_train, ppred)
        chi, hlp = hosmer_lemeshow(y_train, ppred)
        cal_rows.append({"Method": nm, "Brier": brier_score_loss(y_train, ppred),
                         "Slope": sl, "Intercept": ic, "HL_p": hlp})
        print("[%s] Brier %.4f 斜率 %.3f 截距 %.3f HL p=%.3f"
              % (nm, brier_score_loss(y_train, ppred), sl, ic, hlp))
    pd.DataFrame(cal_rows).to_csv(os.path.join(OUT_DIR, "重校准.csv"),
                                  index=False, encoding="utf-8-sig")

# E. 持续事件（可选）
print("\n=== E. 持续事件（可选） ===")
lab_path = resolve_path("sali_event_labels.csv")
if os.path.exists(lab_path):
    labels = pd.read_csv(lab_path)
    id_path = resolve_path("训练集_含ID.csv")
    if os.path.exists(id_path):
        tr_id = pd.read_csv(id_path)
        merged = tr_id.merge(labels[["stay_id", "sali_persistent"]],
                             on="stay_id", how="left")
        print("持续事件训练标签: %d 事件 / %d 例"
              % (int(merged.sali_persistent.sum()), len(merged)))
    else:
        print("找到 sali_event_labels.csv 但缺少 训练集_含ID.csv，",
              "请按退修说明用 R 导出含 stay_id 的数据。")
else:
    print("未提供 sali_event_labels.csv，跳过（见退修说明中的 R 导出步骤）。")

print("\n全部完成。输出目录:", os.path.abspath(OUT_DIR))
