# =====================================================================
# 老年 ICU 脓毒症患者 SALI 预测 —— R 侧全部图片一键重跑（全真矢量 PDF）
# 覆盖：缺失值 / LASSO 偏差 / LASSO 系数 / Boruta / 列线图 /
#       训练校准 / 测试校准 / 竞争风险 CIF(放大) / 外部 ROC / 外部校准
# 全部输出到脚本同目录的 “矢量PDF输出” 文件夹，均为矢量（可无限缩放、文字可选中）
# 运行：在 RStudio 打开本脚本，点 Source；或  source("SALI_全部图片_R矢量PDF.R")
# 需与 origindata.csv 同目录；CIF 需 sali_competing.csv；外部图需 eicu_predictions.csv
# =====================================================================

## ---------------- 0. 依赖包（缺则自动安装） ----------------
need <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE))
      install.packages(p, repos = "https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}
need(c("caret", "dplyr", "ggplot2", "glmnet", "rms", "Boruta",
       "naniar", "survival", "cmprsk"))

## ---------------- 1. 路径与输出目录 ----------------
get_script_dir <- function() {
  # 1) RStudio：取当前活动文档（即本脚本）所在目录
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) return(dirname(normalizePath(p)))
  }
  # 2) 命令行 Rscript：取 --file= 路径
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(fa) && nzchar(fa[1])) return(dirname(normalizePath(fa[1])))
  # 3) 兜底：当前工作目录
  getwd()
}
here <- get_script_dir()
setwd(here)
OUT <- file.path(here, "矢量PDF输出")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
if (.Platform$OS.type == "windows") windowsFonts(Times = windowsFont("Times New Roman"))

## ---------------- 2. 读数据与变量清单 ----------------
data <- read.csv("origindata.csv", check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(ncol(data) == 51)

catVars <- c("Sex", "Race", "Infection source", "Positive culture",
             "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
             "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer")
contVars <- c("Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
              "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
              "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
              "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
              "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
              "NE dose", "OASIS", "SAPS II", "APS III", "LODS")
data$Outcome <- factor(data$Outcome, levels = c(0, 1), labels = c("0", "1"))

## ---------------- 3. 7:3 分层划分（seed 42） ----------------
set.seed(42)
inTrain <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_raw <- data[inTrain, ]; test_raw <- data[-inTrain, ]

## ---------------- 4. 无泄漏插补 ----------------
getmode <- function(x) {
  x <- x[!is.na(x)]; ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
ne_med <- median(train_raw[["NE dose"]][train_raw$Vasopressor == 1], na.rm = TRUE)
imp_ne <- function(df) {
  df[["NE dose"]][df$Vasopressor == 0] <- 0
  idx <- is.na(df[["NE dose"]]) & df$Vasopressor == 1
  df[["NE dose"]][idx] <- ne_med; df
}
cont_other <- setdiff(contVars, "NE dose")
meds <- sapply(train_raw[cont_other], median, na.rm = TRUE)
imp_cont <- function(df) {
  df <- imp_ne(df)
  for (v in cont_other) df[[v]][is.na(df[[v]])] <- meds[v]
  df
}
modes <- sapply(train_raw[catVars], getmode)
imp_cat <- function(df) {
  for (v in catVars) df[[v]][is.na(df[[v]])] <- modes[v]
  df
}
train_imp <- imp_cat(imp_cont(train_raw)); test_imp <- imp_cat(imp_cont(test_raw))
stopifnot(sum(is.na(train_imp)) == 0, sum(is.na(test_imp)) == 0)
for (v in catVars) {
  lev <- sort(unique(train_imp[[v]]))
  train_imp[[v]] <- factor(train_imp[[v]], levels = lev)
  test_imp[[v]]  <- factor(test_imp[[v]],  levels = lev)
}

# 导出插补后（未缩放）的划分文件，供 Python 脚本读取，保证与论文同一次划分
write.csv(train_imp, "训练集_插补未缩放.csv", row.names = FALSE)
write.csv(test_imp,  "测试集_插补未缩放.csv", row.names = FALSE)

## ---------------- 5. 标准化（训练 fit） ----------------
mu <- sapply(train_imp[contVars], mean); sds <- sapply(train_imp[contVars], sd)
scale_by_train <- function(df) {
  for (v in contVars) df[[v]] <- (df[[v]] - mu[v]) / sds[v]
  df
}
train_scaled <- scale_by_train(train_imp); test_scaled <- scale_by_train(test_imp)

# =====================================================================
# 图片 1：缺失值（纵向，ggplot/cairo_pdf 矢量）
# =====================================================================
p_miss <- naniar::gg_miss_var(data, show_pct = TRUE) +
  theme_bw(base_size = 13) + labs(y = "Missing (%)")
ggsave(file.path(OUT, "缺失值.pdf"), p_miss, width = 7, height = 10,
       device = cairo_pdf)

# =====================================================================
# 图片 2-3：LASSO（glmnet 基图，pdf 矢量）
# =====================================================================
set.seed(42)
x <- model.matrix(Outcome ~ . - 1, data = train_scaled)
y <- as.numeric(as.character(train_scaled$Outcome))
lasso.cv <- cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 10)
lasso <- glmnet(x, y, family = "binomial", alpha = 1, nlambda = 1000)

pdf(file.path(OUT, "LASSO偏差.pdf"), width = 8, height = 6.6)
plot(lasso.cv)
abline(v = log(c(lasso.cv$lambda.min, lasso.cv$lambda.1se)), col = "black", lty = 2)
dev.off()

pdf(file.path(OUT, "LASSO系数.pdf"), width = 8, height = 6.6)
plot(lasso, xvar = "lambda")
abline(v = log(c(lasso.cv$lambda.min, lasso.cv$lambda.1se)), col = "black", lty = 2)
dev.off()

# =====================================================================
# 图片 4：Boruta（pdf 矢量）
# =====================================================================
set.seed(42)
boruta_model <- Boruta(Outcome ~ ., data = train_imp, pValue = 0.01,
                       maxRuns = 100, doTrace = 0)
boruta_fix <- TentativeRoughFix(boruta_model)
pdf(file.path(OUT, "Boruta.pdf"), width = 8, height = 6.6)
plot(boruta_fix, sort = TRUE, las = 2, xlab = "")
dev.off()

# =====================================================================
# 图片 5：列线图（rms，pdf 矢量）
# =====================================================================
f5 <- Outcome ~ `Baseline TBIL` + PTT + Sodium + Platelets + Lactate
dd <- datadist(train_imp); options(datadist = "dd")
fit1 <- lrm(f5, data = train_imp, x = TRUE, y = TRUE)
# PTT 系数极小（beta=0.002），默认刻度会挤成一团；指定稀疏端点 25/130
nom1 <- nomogram(fit1, fun = plogis,
                 PTT = c(25, 130),
                 fun.at = c(0.01, 0.05, 0.1, 0.2, 0.35, 0.5),
                 funlabel = "SALI risk")
pdf(file.path(OUT, "列线图.pdf"), width = 11, height = 8.5)
plot(nom1, cex.axis = 0.68, cex.var = 0.85,
     total.points.label = "Total points",
     force.label = TRUE, var.label = TRUE)
dev.off()

# =====================================================================
# 图片 6-7：训练 / 测试校准（rms，pdf 矢量）
# =====================================================================
cal_train <- calibrate(fit1, method = "boot", B = 1000, smooth = TRUE)
pdf(file.path(OUT, "训练校准.pdf"), width = 8, height = 6.6)
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
dev.off()

dd_t <- datadist(test_imp); options(datadist = "dd_t")
fit_test <- lrm(f5, data = test_imp, x = TRUE, y = TRUE)
cal_test <- calibrate(fit_test, method = "boot", B = 1000, smooth = TRUE)
pdf(file.path(OUT, "测试校准.pdf"), width = 8, height = 6.6)
plot(cal_test, xlim = c(0, 1), ylim = c(0, 1),
     xlab = "Predicted probability", ylab = "Observed probability",
     subtitles = FALSE, legend = FALSE)
lines(cal_test[, c("predy", "calibrated.corrected")], lwd = 3, col = "#2166AC")
abline(0, 1, lty = 2, lwd = 2)
dev.off()

# =====================================================================
# 图片 8：竞争风险 CIF（cmprsk，纵轴 0-0.1 放大，pdf 矢量）
# =====================================================================
if (file.exists("sali_competing.csv")) {
  dc <- read.csv("sali_competing.csv", check.names = FALSE)
  dc$time_day <- dc$time_hr / 24
  ci <- cmprsk::cuminc(ftime = dc$time_day, fstatus = dc$event_type)
  pdf(file.path(OUT, "CIF放大.pdf"), width = 8, height = 6.6)
  plot(ci, xlab = "Days since ICU admission", ylab = "Cumulative incidence",
       lwd = 2, col = c(2, 4), ylim = c(0, 0.1))
  legend("topleft", legend = c("SALI", "Death"), col = c(2, 4), lwd = 2)
  dev.off()
} else {
  message("未找到 sali_competing.csv，跳过 CIF（请先按 competing_risk.R 顶部 SQL 导出）")
}

# =====================================================================
# 图片 9-10：外部 ROC / 外部校准（base R，pdf 矢量）
# =====================================================================
if (file.exists("eicu_predictions.csv")) {
  de <- read.csv("eicu_predictions.csv", check.names = FALSE)
  ye <- de$y

  roc_xy <- function(score, yy) {
    ord <- order(score, decreasing = TRUE)
    s <- score[ord]; g <- yy[ord]
    P <- sum(yy == 1); N <- sum(yy == 0)
    tpr <- c(0); fpr <- c(0); tp <- 0; fp <- 0; i <- 1
    while (i <= length(s)) {
      j <- i
      while (j <= length(s) && s[j] == s[i]) {
        if (g[j] == 1) tp <- tp + 1 else fp <- fp + 1
        j <- j + 1
      }
      tpr <- c(tpr, tp / P); fpr <- c(fpr, fp / N); i <- j
    }
    auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
    list(fpr = fpr, tpr = tpr, auc = auc)
  }
  r <- roc_xy(de$prob_frozen, ye)
  pdf(file.path(OUT, "外部ROC.pdf"), width = 8, height = 7.1)
  par(mar = c(4.5, 4.5, 2, 1), family = "serif")
  plot(r$fpr, r$tpr, type = "l", lwd = 2.5, col = "#1f4e79",
       xlab = "1 - Specificity", ylab = "Sensitivity",
       xlim = c(0, 1), ylim = c(0, 1), las = 1)
  abline(0, 1, lty = 2, col = "gray50")
  legend("bottomright", sprintf("eICU external ROC, AUC = %.3f", r$auc),
         lwd = 2.5, col = "#1f4e79", bty = "n")
  dev.off()

  cal_pts <- function(score, yy, g = 10) {
    ord <- order(score)
    grp <- cut(seq_along(score),
               breaks = quantile(seq_along(score), probs = seq(0, 1, length.out = g + 1)),
               include.lowest = TRUE, labels = FALSE)
    data.frame(pred = as.numeric(tapply(score[ord], grp, mean)),
               obs  = as.numeric(tapply(yy[ord], grp, mean)))
  }
  cf <- cal_pts(de$prob_frozen, ye); cr <- cal_pts(de$prob_logistic, ye)
  pdf(file.path(OUT, "外部校准.pdf"), width = 8, height = 7.1)
  par(mar = c(4.5, 4.5, 2, 1), family = "serif")
  xmax <- max(cf$pred, cr$pred) * 1.05; ymax <- max(cf$obs, cr$obs) * 1.1
  plot(NA, xlim = c(0, xmax), ylim = c(0, ymax), las = 1,
       xlab = "Predicted probability", ylab = "Observed probability")
  abline(0, 1, lty = 2, col = "gray40", lwd = 1.5)
  points(cf$pred, cf$obs, pch = 19, col = "#c0392b", cex = 1.2)
  lines(cf$pred, cf$obs, col = "#c0392b", lwd = 2)
  points(cr$pred, cr$obs, pch = 17, col = "#1f4e79", cex = 1.2)
  lines(cr$pred, cr$obs, col = "#1f4e79", lwd = 2)
  legend("topleft", bty = "n",
         legend = c("Ideal", "Frozen model", "Logistic recalibration"),
         pch = c(NA, 19, 17), lty = c(2, 1, 1), lwd = c(1.5, 2, 2),
         col = c("gray40", "#c0392b", "#1f4e79"))
  dev.off()
} else {
  message("未找到 eicu_predictions.csv，跳过外部两图")
}

message("\nR 侧全部完成，矢量 PDF 位于：", OUT)
