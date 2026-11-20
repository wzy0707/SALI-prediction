# =====================================================================
# S0 · 干净重跑：只做 7:3 划分 + 无泄漏插补，导出 Python 流水线需要的两个文件
# 数据：origindata.csv（3,567 × 51；锁定口径，205 事件）
# 输出：训练集_插补未缩放.csv / 测试集_插补未缩放.csv（与 python old liver.py 配对）
# 运行：RStudio 打开本脚本点 Source；脚本自动定位到自身所在目录，无需手改路径
# =====================================================================

need <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE))
      install.packages(p, repos = "https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}
need(c("caret"))

# 自动定位脚本所在目录（RStudio / Rscript 兼容）
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
if (.Platform$OS.type == "windows")
  windowsFonts(Times = windowsFont("Times New Roman"))

catVars <- c("Sex", "Race", "Infection source", "Positive culture",
             "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
             "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer")
contVars <- c("Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
              "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
              "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
              "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
              "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
              "NE dose", "OASIS", "SAPS II", "APS III", "LODS")

data <- read.csv("origindata.csv", check.names = FALSE, stringsAsFactors = FALSE)
stopifnot(ncol(data) == 51, nrow(data) == 3567)
data$Outcome <- factor(data$Outcome, levels = c(0, 1), labels = c("0", "1"))

# 7:3 分层划分（seed 42，确定性，每次一致）
set.seed(42)
inTrain <- createDataPartition(data$Outcome, p = 0.7, list = FALSE)
train_raw <- data[inTrain, ]; test_raw <- data[-inTrain, ]
cat(sprintf("划分: 训练 %d | 测试 %d\n", nrow(train_raw), nrow(test_raw)))

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
train_imp <- imp_cat(imp_cont(train_raw))
test_imp <- imp_cat(imp_cont(test_raw))
stopifnot(sum(is.na(train_imp)) == 0, sum(is.na(test_imp)) == 0)

# 因子水平以训练集为准
for (v in catVars) {
  lev <- sort(unique(train_imp[[v]]))
  train_imp[[v]] <- factor(train_imp[[v]], levels = lev)
  test_imp[[v]] <- factor(test_imp[[v]], levels = lev)
}

write.csv(train_imp, "训练集_插补未缩放.csv", row.names = FALSE)
write.csv(test_imp, "测试集_插补未缩放.csv", row.names = FALSE)
cat("已导出两个干净的插补文件，可直接运行 python old liver.py。\n")
