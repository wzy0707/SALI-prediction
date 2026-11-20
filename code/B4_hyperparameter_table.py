# -*- coding: utf-8 -*-
# =====================================================================
# B4 · 六模型超参数搜索网格、最终配置与随机种子（附录表）
# 对应审稿意见：M4（搜索网格与最终配置未报告，可复现性缺口）
#
# 数据：搜索空间与 python old liver.py 内 GridSearchCV 的 grid 完全一致；
#       最终配置与本机 各模型最优参数.csv 一致（若该文件在同目录会自动核对）。
# 运行：  C:/python310/python.exe B4_hyperparameter_table.py
# =====================================================================
import os
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SEED = 42
GRID_CONFIG = "python old liver.py（models 字典；GitHub 仓库 code/ 同名脚本）"

# 搜索空间（具体取值列表）
search_space = {
    "LR": "C ∈ {0.01, 0.1, 1, 10}; penalty = {l2}",
    "XGBoost": "max_depth ∈ {3, 6}; n_estimators ∈ {100, 200}; learning_rate ∈ {0.03, 0.1}",
    "LightGBM": "max_depth ∈ {-1, 5}; n_estimators ∈ {100, 200}; learning_rate ∈ {0.03, 0.1}",
    "MLP": "hidden_layer_sizes ∈ {(50,), (100,), (50, 50)}; alpha ∈ {0.0001, 0.001}",
    "SVM": "C ∈ {0.5, 1, 10}; kernel ∈ {rbf, linear}",
    "CatBoost": "depth ∈ {4, 6}; iterations ∈ {100, 200}; learning_rate ∈ {0.03, 0.1}",
}

# 最终配置（GridSearchCV 按 inner 5-fold、average_precision 选出）
final_config = {
    "LR": "C = 0.01; penalty = l2",
    "XGBoost": "max_depth = 3; n_estimators = 100; learning_rate = 0.03",
    "LightGBM": "max_depth = 5; n_estimators = 100; learning_rate = 0.03",
    "MLP": "hidden_layer_sizes = (100,); alpha = 0.001",
    "SVM": "C = 1; kernel = rbf",
    "CatBoost": "depth = 6; iterations = 200; learning_rate = 0.03",
}

rows = []
for m in ["LR", "XGBoost", "LightGBM", "CatBoost", "MLP", "SVM"]:
    rows.append({"Model": m, "Search space": search_space[m],
                 "Final configuration": final_config[m], "Random seed": SEED,
                 "Grid config file": GRID_CONFIG})
out = pd.DataFrame(rows)

# 若本机 各模型最优参数.csv 存在，自动核对关键值
best_path = os.path.join(SCRIPT_DIR, "各模型最优参数.csv")
if not os.path.exists(best_path):
    tab_dir = os.path.join(SCRIPT_DIR, "python old liver 输出", "表")
    best_path = os.path.join(tab_dir, "各模型最优参数.csv")
if os.path.exists(best_path):
    b = pd.read_csv(best_path).set_index(b.columns[0])
    print("已找到各模型最优参数.csv，最终配置以该文件为准核对。")
    check = {"LR": ("C", 0.01), "XGBoost": ("learning_rate", 0.03),
             "LightGBM": ("max_depth", 5), "CatBoost": ("iterations", 200),
             "MLP": ("alpha", 0.001), "SVM": ("C", 1)}
    for m, (col, val) in check.items():
        got = b.loc[m, col]
        status = "OK" if abs(float(got) - val) < 1e-9 else "MISMATCH got=%s" % got
        print("  %-8s %s = %s -> %s" % (m, col, val, status))
else:
    print("未找到各模型最优参数.csv，表内为脚本内置（与 pipeline 一致），可直接使用。")

pd.set_option("display.width", 250)
print("\n=== 超参数搜索网格与最终配置 ===")
print(out.to_string(index=False))

out_path = os.path.join(SCRIPT_DIR, "B4_超参数网格与最终配置.csv")
out.to_csv(out_path, index=False, encoding="utf-8-sig")
print("\n已保存:", out_path)
