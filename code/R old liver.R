# =====================================================================
# 老年 ICU 脓毒症患者 SALI 机器学习预测 —— 数据预处理 + 变量筛选（R 版）
# 数据：origindata.csv（3567 x 51；不可能值已在取数阶段剔除）
# 口径：先 7:3 分层划分，再在训练集 fit 预处理、测试集仅 transform（无泄漏）
# 说明：正文报告的精确指标来自 Python 主流程；本脚本复刻同一无泄漏方法，
#       因 R 与 Python 划分引擎不同，逐例结果不会完全一致，但方法学一致。
# =====================================================================

## ---------------- 0. 安装并加载包（仅在缺失时安装） ----------------
need <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE)) install.packages(p, repos = "https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}
need(c("caret", "dplyr", "ggplot2", "tableone", "glmnet", "rms",
       "ResourceSelection", "car", "Boruta", "naniar"))

## ---------------- 1. 工作目录与读数据 ----------------
# 按你本机实际路径修改；origindata.csv 需放在该目录下
setwd("C:/Users/12431/Desktop/机器学习预测模型全流程代码_全网同名_派大珍/机器学习预测模型全流程代码_全网同名_派大珍/1.R数据预处理+变量筛选")
if (.Platform$OS.type == "windows") {
  windowsFonts(Times = windowsFont("Times New Roman"))
}

# check.names = FALSE 保留含空格/斜杠的展示名（如 Infection source、Sedative/analgesic）
data <- read.csv("origindata.csv", check.names = FALSE, stringsAsFactors = FALSE)
colnames(data)
stopifnot(ncol(data) == 51)

## ---------------- 2. 变量类型与列位置 ----------------
# 第 1 列 Outcome；第 2-15 列 14 个分类；第 16-51 列 36 个连续
catVars <- c("Sex", "Race", "Infection source", "Positive culture",
             "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
             "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer")
contVars <- c("Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
              "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
              "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
              "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
              "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
              "NE dose", "OASIS", "SAPS II", "APS III", "LODS")
stopifnot(length(catVars) == 14, length(contVars) == 36)
stopifnot(all(c(catVars, contVars, "Outcome") %in% colnames(data)))

# 结局因子化（0=非 SALI，1=SALI）
data$Outcome <- factor(data$Outcome, levels = c(0, 1), labels = c("0", "1"))

# 不做 3-SD / IQR 删值：极端但现实可能的 ICU 数值一律保留，不可能值已在取数阶段剔除
summary(data)

## ---------------- 3. 缺失值分布（可视化，可选） ----------------
gg_miss_var(data, show_pct = TRUE) +
  theme_bw(base_size = 14) +
  labs(y = "Missing (%)")
missing_ratio <- data.frame(Variable = names(data),
                            Missing_Ratio = round(colMeans(is.na(data)), 4))
write.csv(missing_ratio, "缺失值比例.csv", row.names = FALSE)

## ---------------- 4. 先划分：7:3 分层（seed 42） ----------------
set.seed(42)
inTrain <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_raw <- data[inTrain, ]
test_raw  <- data[-inTrain, ]
c(nrow(train_raw), table(train_raw$Outcome))
c(nrow(test_raw),  table(test_raw$Outcome))

## ---------------- 5. 无泄漏插补（训练 fit，测试 transform） ----------------
getmode <- function(x) {
  x <- x[!is.na(x)]
  ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}

# 5.1 NE dose 结构性零：未用血管活性药者必为 0；缺失仅出现在 Vasopressor=1，按组内中位数
ne_med <- median(train_raw[["NE dose"]][train_raw$Vasopressor == 1], na.rm = TRUE)  # 约 0.2001
imp_ne <- function(df) {
  df[["NE dose"]][df$Vasopressor == 0] <- 0
  idx <- is.na(df[["NE dose"]]) & df$Vasopressor == 1
  df[["NE dose"]][idx] <- ne_med
  df
}

# 5.2 其余连续变量：训练集中位数
cont_other <- setdiff(contVars, "NE dose")
meds <- sapply(train_raw[cont_other], median, na.rm = TRUE)
imp_cont <- function(df) {
  df <- imp_ne(df)
  for (v in cont_other) df[[v]][is.na(df[[v]])] <- meds[v]
  df
}

# 5.3 分类变量：训练集众数
modes <- sapply(train_raw[catVars], getmode)
imp_cat <- function(df) {
  for (v in catVars) df[[v]][is.na(df[[v]])] <- modes[v]
  df
}

train_imp <- imp_cat(imp_cont(train_raw))
test_imp  <- imp_cat(imp_cont(test_raw))
stopifnot(sum(is.na(train_imp)) == 0, sum(is.na(test_imp)) == 0)

## ---------------- 6. 因子水平对齐（以训练集为准） ----------------
for (v in catVars) {
  lev <- sort(unique(train_imp[[v]]))
  train_imp[[v]] <- factor(train_imp[[v]], levels = lev)
  test_imp[[v]]  <- factor(test_imp[[v]],  levels = lev)
}

## ---------------- 7. 标准化（训练 fit，测试 transform） ----------------
mu  <- sapply(train_imp[contVars], mean)
sds <- sapply(train_imp[contVars], sd)
scale_by_train <- function(df) {
  for (v in contVars) df[[v]] <- (df[[v]] - mu[v]) / sds[v]
  df
}
train_scaled <- scale_by_train(train_imp)
test_scaled  <- scale_by_train(test_imp)

## ---------------- 8. 基线表（用未缩放的插补数据，中位数[IQR]） ----------------
myVars <- c(catVars, contVars)
# 全部连续变量按非正态处理，报告中位数 [IQR]，与论文 Table 1 一致
t1_overall <- CreateTableOne(vars = myVars, factorVars = catVars,
                             data = train_imp, strata = "Outcome", addOverall = TRUE)
tab_overall <- print(t1_overall, showAllLevels = TRUE, nonnormal = contVars,
                     catDigits = 2, contDigits = 2, pDigits = 3)
write.csv(tab_overall, "训练集基线表.csv")

# 全队列（训练+测试）总体基线表
all_imp <- bind_rows(train_imp, test_imp)
t1_all <- CreateTableOne(vars = myVars, factorVars = catVars,
                         data = all_imp, strata = "Outcome", addOverall = TRUE)
tab_all <- print(t1_all, showAllLevels = TRUE, nonnormal = contVars,
                 catDigits = 2, contDigits = 2, pDigits = 3)
write.csv(tab_all, "总体基线表.csv")

## ---------------- 9. 变量筛选 ----------------
# 9.1 多因素逻辑（SHAP 前 5 变量）：OR/CI 表 + VIF
# 含空格的变量在公式中用反引号包裹
f5 <- Outcome ~ `Baseline TBIL` + PTT + Sodium + Platelets + Lactate
Multi_log <- glm(f5, data = train_imp, family = binomial)
cs <- summary(Multi_log)$coefficients
result_df <- data.frame(
  Variable = rownames(cs),
  Beta = round(cs[, "Estimate"], 3),
  SE = round(cs[, "Std. Error"], 3),
  OR = round(exp(cs[, "Estimate"]), 3),
  Lower_CI = round(exp(cs[, "Estimate"] - 1.96 * cs[, "Std. Error"]), 3),
  Upper_CI = round(exp(cs[, "Estimate"] + 1.96 * cs[, "Std. Error"]), 3),
  p_value = round(cs[, "Pr(>|z|)"], 3))
print(result_df)
write.csv(result_df, "多因素逻辑_OR表.csv", row.names = FALSE)
print(vif(Multi_log))  # 多重共线性检验

# 9.2 LASSO（用已标准化、无缺失的训练数据；model.matrix 自动 one-hot）
set.seed(42)
x <- model.matrix(Outcome ~ . - 1, data = train_scaled)
y <- as.numeric(as.character(train_scaled$Outcome))
stopifnot(sum(is.na(x)) == 0)   # 修复 glmnet "x has missing values" 的关键自检
lasso.cv <- cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 10)
plot(lasso.cv)
abline(v = log(c(lasso.cv$lambda.min, lasso.cv$lambda.1se)), col = "black", lty = 2)
lasso <- glmnet(x, y, family = "binomial", alpha = 1, nlambda = 1000)
plot(lasso, xvar = "lambda")
abline(v = log(c(lasso.cv$lambda.min, lasso.cv$lambda.1se)), col = "black", lty = 2)
lasso_coefs <- as.matrix(coef(lasso.cv, s = "lambda.1se"))
write.csv(lasso_coefs, "LASSO筛选变量.csv")

# 9.3 Boruta（运行较慢，按需执行；使用无缺失的插补数据）
set.seed(42)
boruta_model <- Boruta(Outcome ~ ., data = train_imp, pValue = 0.01,
                       maxRuns = 100, doTrace = 0)
print(TentativeRoughFix(boruta_model))
plot(boruta_model, sort = TRUE, las = 2, xlab = "")
write.csv(data.frame(Variables = getSelectedAttributes(
  TentativeRoughFix(boruta_model), withTentative = FALSE)),
  "Boruta筛选变量.csv", row.names = FALSE)

## ---------------- 10. 列线图（rms，5 变量） ----------------
dd <- datadist(train_imp)
options(datadist = "dd")
fit1 <- lrm(f5, data = train_imp, x = TRUE, y = TRUE)
nom1 <- nomogram(fit1, fun = plogis,
                 fun.at = c(0.01, 0.05, 0.1, 0.2, 0.4, 0.6),
                 funlabel = "SALI risk")
plot(nom1)

## ---------------- 11. 校准曲线（训练 bootstrap + 测试） ----------------
cal_train <- calibrate(fit1, method = "boot", B = 1000, smooth = TRUE)
plot(cal_train, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted probability", ylab = "Observed probability",
     subtitles = FALSE, legend = FALSE)
lines(cal_train[, c("predy", "calibrated.corrected")], lwd = 3, col = "#2166AC")
lines(cal_train[, c("predy", "calibrated.orig")], lwd = 3, col = "tomato")
abline(0, 1, lty = 2, lwd = 2)
legend("bottomright", bty = "n",
       legend = c("Ideal", "Bias-corrected", "Apparent"),
       lty = c(2, 1, 1), lwd = c(2, 3, 3),
       col = c("black", "#2166AC", "tomato"))

# 测试集校准（在测试集上用同一公式重拟合）
dd_t <- datadist(test_imp)
options(datadist = "dd_t")
fit_test <- lrm(f5, data = test_imp, x = TRUE, y = TRUE)
cal_test <- calibrate(fit_test, method = "boot", B = 1000, smooth = TRUE)
plot(cal_test, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted probability", ylab = "Observed probability",
     subtitles = FALSE, legend = FALSE)
lines(cal_test[, c("predy", "calibrated.corrected")], lwd = 3, col = "#2166AC")
abline(0, 1, lty = 2, lwd = 2)

## ---------------- 12. Hosmer-Lemeshow 检验 ----------------
hl_tr <- hoslem.test(as.numeric(as.character(train_imp$Outcome)),
                     predict(fit1, type = "fitted"), g = 10)
hl_te <- hoslem.test(as.numeric(as.character(test_imp$Outcome)),
                     predict(fit_test, type = "fitted"), g = 10)
print(hl_tr)
print(hl_te)
# ============================== 结束 ==============================
