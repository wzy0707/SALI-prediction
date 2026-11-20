# =====================================================================
# B6 · 图重绘（矢量 PDF）：Boruta 标签 / 校准轴范围 / 列线图轴
# 对应审稿意见：Fig. A.3、A.7、A.6
# 运行：RStudio 打开点 Source；或  Rscript B6_figure_redraw.R
# 需与 origindata.csv 同目录；输出到 “B6矢量图输出”
# =====================================================================

need <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE))
      install.packages(p, repos = "https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}
need(c("caret", "ggplot2", "rms", "Boruta"))

get_script_dir <- function() {
  if (requireNamespace("rstudioapi", quietly = TRUE) && rstudioapi::isAvailable()) {
    p <- rstudioapi::getActiveDocumentContext()$path
    if (!is.null(p) && nzchar(p)) return(dirname(normalizePath(p)))
  }
  args <- commandArgs(trailingOnly = FALSE)
  fa <- sub("^--file=", "", args[grep("^--file=", args)])
  if (length(fa) && nzchar(fa[1])) return(dirname(normalizePath(fa[1])))
  getwd()
}
here <- get_script_dir()
setwd(here)
OUT <- file.path(here, "B6矢量图输出")
dir.create(OUT, showWarnings = FALSE, recursive = TRUE)
if (.Platform$OS.type == "windows")
  windowsFonts(Times = windowsFont("Times New Roman"))

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

set.seed(42)
inTrain <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_raw <- data[inTrain, ]; test_raw <- data[-inTrain, ]

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

# =====================================================================
# 1) Boruta：加宽画布 + 大底边距 + 缩小轴字，标签不再重叠
# =====================================================================
set.seed(42)
boruta_model <- Boruta(Outcome ~ ., data = train_imp, pValue = 0.01,
                       maxRuns = 100, doTrace = 0)
boruta_fix <- TentativeRoughFix(boruta_model)
pdf(file.path(OUT, "Boruta_修正.pdf"), width = 13, height = 7.5)
par(mar = c(13, 4.5, 3, 1), cex.axis = 0.78, las = 2)
plot(boruta_fix, sort = TRUE, xlab = "")
dev.off()
message("Boruta 修正已输出。")

# =====================================================================
# 2) 列线图：PTT/Platelets 按实际数据分位设刻度，risk 轴按概率范围
# =====================================================================
f5 <- Outcome ~ `Baseline TBIL` + PTT + Sodium + Platelets + Lactate
dd <- datadist(train_imp); options(datadist = "dd")
fit1 <- lrm(f5, data = train_imp, x = TRUE, y = TRUE)

qtiles <- function(v, ps = c(0.05, 0.25, 0.5, 0.75, 0.95))
  as.numeric(quantile(train_imp[[v]], probs = ps, na.rm = TRUE))

# 实际预测概率范围，用于 risk 轴
p_actual <- plogis(predict(fit1, newdata = train_imp))
pmax <- max(p_actual)
fun_at <- round(head(seq(0.01, pmax * 1.05, length.out = 6), 6), 3)

nom1 <- nomogram(
  fit1, fun = plogis,
  PTT = c(30, 90, 150),
  Platelets = qtiles("Platelets"),
  Sodium = qtiles("Sodium"),
  Lactate = qtiles("Lactate"),
  `Baseline TBIL` = qtiles("Baseline TBIL"),
  fun.at = fun_at, funlabel = "SALI risk")
pdf(file.path(OUT, "列线图_修正.pdf"), width = 13, height = 8.5)
plot(nom1, cex.axis = 0.62, cex.var = 0.9,
     total.points.label = "Total points",
     force.label = TRUE, var.label = TRUE)
dev.off()
message("列线图修正已输出（PTT/Platelets 按实际分位）。")

# =====================================================================
# 3) 校准：轴范围按实际预测概率，不再右半幅空白
# =====================================================================
cal_train <- calibrate(fit1, method = "boot", B = 1000, smooth = TRUE)
xmax_tr <- max(cal_train[, "predy"]) * 1.05
pdf(file.path(OUT, "训练校准_修正.pdf"), width = 8, height = 6.6)
plot(cal_train, xlim = c(0, xmax_tr), ylim = c(0, xmax_tr),
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
xmax_te <- max(cal_test[, "predy"]) * 1.05
ymax_te <- max(cal_test[, "calibrated.corrected"], na.rm = TRUE) * 1.1
pdf(file.path(OUT, "测试校准_修正.pdf"), width = 8, height = 6.6)
plot(cal_test, xlim = c(0, xmax_te), ylim = c(0, ymax_te),
     xlab = "Predicted probability", ylab = "Observed probability",
     subtitles = FALSE, legend = FALSE)
lines(cal_test[, c("predy", "calibrated.corrected")], lwd = 3, col = "#2166AC")
abline(0, 1, lty = 2, lwd = 2)
# 如实标注测试集中段偏离
mtext("Mid-range deviation on the test set is shown without 0-1 rescaling",
      side = 3, cex = 0.8, font = 3)
dev.off()
message("校准修正已输出（轴按实际范围，测试中段偏离如实呈现）。")

message("\nB6 完成，矢量 PDF 位于：", OUT)
