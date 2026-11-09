# =====================================================================
# persistent_rerun.R —— 持续事件（persistent SALI）完整重跑
# 前置：已跑 SALI_退修_持续事件与选择偏倚.sql，建好 sali_event_labels。
# ---------------------------------------------------------------------
# 需要的文件（都放本脚本同目录）：
#   1) origindata.csv                 —— 现有 51 列规范建模数据（3567 行）
#   2) sali_ml_dataset_full.csv      —— 已导出的超集原始表（6398 行，含 stay_id）
#   3) sali_event_labels.csv          —— 由 sali_event_labels 表导出
#
# 对齐方法（不依赖行顺序）：用 34 个连续变量生成每行唯一指纹，从超集表
#   反查 stay_id，再关联持续标签；强制校验原 Outcome == sali_single 全部一致。
# 排除 NE dose、Creatinine（这两列在 origindata 做过不可能值置缺失处理）。
# =====================================================================

need <- function(pkgs) {
  for (p in pkgs) {
    if (!requireNamespace(p, quietly = TRUE))
      install.packages(p, repos = "https://cloud.r-project.org")
    library(p, character.only = TRUE)
  }
}
need(c("caret", "dplyr"))

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile, mustWork = NA)),
                 error = function(e) NA_character_)
if (is.na(here) || here == "") here <- getwd()
setwd(here)
if (.Platform$OS.type == "windows")
  windowsFonts(Times = windowsFont("Times New Roman"))

read_in <- function(f) read.csv(file.path(here, f), check.names = FALSE,
                                stringsAsFactors = FALSE)
# pgAdmin 导出可能把 SQL 空值写成文本 "NULL"，统一还原为缺失 NA
null_to_na <- function(df) {
  as.data.frame(lapply(df, function(col) {
    if (is.character(col)) {
      col[trimws(col) %in% c("NULL", "null", "")] <- NA
    }
    col
  }), check.names = FALSE, row.names = rownames(df))
}
orig <- null_to_na(read_in("origindata.csv"))
full <- null_to_na(read_in("sali_ml_dataset_full.csv"))
lab  <- null_to_na(read_in("sali_event_labels.csv"))
stopifnot(nrow(orig) == 3567, "stay_id" %in% names(full),
          "has_follow_7d" %in% names(full), "stay_id" %in% names(lab))

# 原始连续列 → 展示名（36 个）
raw2disp <- c(age = "Age", charlson = "CCI", hr_mean = "HR", mbp_mean = "MBP",
  rr_mean = "RR", temp_max = "Temperature", spo2_mean = "SpO2", gcs_min = "GCS",
  uo_24h = "Urine output", sofa = "SOFA", wbc_max = "WBC", plt_min = "Platelets",
  rdw_max = "RDW", sodium_max = "Sodium", bicarbonate_min = "Bicarbonate",
  glucose_max = "Glucose", creatinine_max = "Creatinine", albumin_min = "Albumin",
  lactate_max = "Lactate", tb0_max = "Baseline TBIL", alt_max = "ALT",
  ast_max = "AST", inr_max = "Baseline INR", hgb_min = "Hemoglobin",
  potassium_max = "Potassium", aniongap_max = "Anion gap", bun_max = "BUN",
  ph_min = "pH", alp_max = "ALP", pt_max = "PT", ptt_max = "PTT",
  neq_max = "NE dose", oasis = "OASIS", sapsii = "SAPS II",
  apsiii = "APS III", lods = "LODS")
miss_raw <- setdiff(names(raw2disp), names(full))
if (length(miss_raw)) stop("超集表缺少列: ", paste(miss_raw, collapse = ", "))

# 取入组 3567，重命名连续列
full$has_follow_7d <- as.numeric(as.character(full$has_follow_7d))
raw_in <- full[full$has_follow_7d == 1, ]
stopifnot(nrow(raw_in) == 3567)
raw_id <- raw_in$stay_id
raw_cont <- raw_in[, names(raw2disp), drop = FALSE]
colnames(raw_cont) <- raw2disp

# 指纹列：34 个连续变量（排除 NE dose、Creatinine）
key_cols <- setdiff(as.character(raw2disp), c("NE dose", "Creatinine"))
miss_disp <- setdiff(key_cols, names(orig))
if (length(miss_disp)) stop("origindata 缺少列: ", paste(miss_disp, collapse = ", "))

check_numeric <- function(df, cols, tag) {
  for (v in cols) {
    x <- df[[v]]
    conv <- as.numeric(as.character(x))
    # 原缺失位置之外若出现新 NA，说明存在无法解析的非数值
    bad <- is.na(conv) & !is.na(x)
    if (any(bad)) stop(sprintf("%s 的 %s 存在无法解析的非数值，停止。", tag, v))
  }
}
check_numeric(orig, key_cols, "origindata")
check_numeric(raw_cont, key_cols, "超集表")

mk_key <- function(df, cols) {
  sub <- df[, cols, drop = FALSE]
  # 显式转数值，避免读取时被识别为字符型导致 round 失败
  num <- sapply(sub, function(x) as.numeric(as.character(x)))
  apply(format(round(num, 6), nsmall = 6, scientific = FALSE),
        1, paste, collapse = "|")
}
key_orig <- mk_key(orig, key_cols)
key_raw  <- mk_key(raw_cont, key_cols)
if (anyDuplicated(key_orig)) stop("origindata 指纹不唯一，需增加指纹变量。")
if (anyDuplicated(key_raw))  stop("超集表指纹不唯一，需增加指纹变量。")

ix <- match(key_orig, key_raw)

# 对极少数因第 6 位小数舍入边界未匹配的行，用标准化最近邻补配
resid <- which(is.na(ix))
if (length(resid)) {
  used <- unique(ix[!is.na(ix)])
  avail <- setdiff(seq_len(nrow(raw_cont)), used)
  Mo <- sapply(orig[, key_cols, drop = FALSE],
               function(x) as.numeric(as.character(x)))
  Mr <- sapply(raw_cont[, key_cols, drop = FALSE],
               function(x) as.numeric(as.character(x)))
  cmean <- colMeans(Mo, na.rm = TRUE)
  csd <- apply(Mo, 2, sd, na.rm = TRUE); csd[csd == 0] <- 1
  std <- function(M) {
    Z <- sweep(sweep(M, 2, cmean, "-"), 2, csd, "/")
    Z[is.na(Z)] <- 0; Z
  }
  Zo <- std(Mo); Zr <- std(Mr)
  for (i in resid) {
    d <- colSums((t(Zr[avail, , drop = FALSE]) - Zo[i, ])^2)
    ord <- order(d)
    best_d <- d[ord[1]]
    if (best_d > 1e-4)
      stop("最近邻距离过大，疑似非同一行，停止。")
    if (length(ord) > 1 && abs(d[ord[2]] - best_d) < 1e-12)
      stop("存在并列最近邻，无法唯一匹配，停止。")
    ix[i] <- avail[ord[1]]
  }
  cat("最近邻补配行数:", length(resid), "\n")
}
if (any(is.na(ix))) stop("仍有未匹配行，停止。")
if (anyDuplicated(ix)) stop("匹配非一对一，停止。")
orig$stay_id <- raw_id[ix]

# 关联持续标签并强制校验原结局
jm <- match(orig$stay_id, lab$stay_id)
stopifnot(!any(is.na(jm)))
orig$sali_single     <- as.numeric(as.character(lab$sali_single[jm]))
orig$sali_persistent <- as.numeric(as.character(lab$sali_persistent[jm]))
orig$sali_transient  <- as.numeric(as.character(lab$sali_transient[jm]))
if (!all(orig$Outcome == orig$sali_single))
  stop("校验失败：关联后 Outcome 与 sali_single 不一致，停止。")
cat("指纹匹配与对齐校验通过：3567 行全部 1:1 匹配，Outcome 与 sali_single 一致。\n")

catVars <- c("Sex", "Race", "Infection source", "Positive culture",
             "Mechanical ventilation", "Vasopressor", "CRRT", "Corticosteroid",
             "Sedative/analgesic", "CHF", "Diabetes", "COPD", "CKD", "Cancer")
contVars <- c("Age", "CCI", "HR", "MBP", "RR", "Temperature", "SpO2", "GCS",
              "Urine output", "SOFA", "WBC", "Platelets", "RDW", "Sodium",
              "Bicarbonate", "Glucose", "Creatinine", "Albumin", "Lactate",
              "Baseline TBIL", "ALT", "AST", "Baseline INR", "Hemoglobin",
              "Potassium", "Anion gap", "BUN", "pH", "ALP", "PT", "PTT",
              "NE dose", "OASIS", "SAPS II", "APS III", "LODS")

# 以持续事件为新结局
d <- orig
d$Outcome <- factor(d$sali_persistent, levels = c(0, 1), labels = c("0", "1"))
cat("持续事件数:", sum(d$sali_persistent == 1), "/", nrow(d), "\n")

set.seed(42)
inTrain <- createDataPartition(d$Outcome, p = 0.7, list = FALSE)
train_raw <- d[inTrain, ]; test_raw <- d[-inTrain, ]

getmode <- function(x) {
  x <- x[!is.na(x)]; ux <- unique(x)
  ux[which.max(tabulate(match(x, ux)))]
}
ne_med <- median(train_raw[["NE dose"]][train_raw$Vasopressor == 1], na.rm = TRUE)
imp_ne <- function(x) {
  x[["NE dose"]][x$Vasopressor == 0] <- 0
  idx <- is.na(x[["NE dose"]]) & x$Vasopressor == 1
  x[["NE dose"]][idx] <- ne_med; x
}
meds <- sapply(train_raw[setdiff(contVars, "NE dose")], median, na.rm = TRUE)
modes <- sapply(train_raw[catVars], getmode)
prep <- function(x) {
  x <- imp_ne(x)
  for (v in setdiff(contVars, "NE dose")) x[[v]][is.na(x[[v]])] <- meds[v]
  for (v in catVars) x[[v]][is.na(x[[v]])] <- modes[v]
  x
}
train_imp <- prep(train_raw); test_imp <- prep(test_raw)
for (v in catVars) {
  lev <- sort(unique(train_imp[[v]]))
  train_imp[[v]] <- factor(train_imp[[v]], levels = lev)
  test_imp[[v]]  <- factor(test_imp[[v]], levels = lev)
}
mu <- sapply(train_imp[contVars], mean); sds <- sapply(train_imp[contVars], sd)
sc <- function(x) {
  for (v in contVars) x[[v]] <- (x[[v]] - mu[v]) / sds[v]
  x
}
train_scaled <- sc(train_imp); test_scaled <- sc(test_imp)

keep_cols <- c("Outcome", catVars, contVars)
write.csv(train_raw[, keep_cols],  "训练集_原始含缺失.csv", row.names = FALSE)
write.csv(test_raw[, keep_cols],   "测试集_原始含缺失.csv", row.names = FALSE)
write.csv(train_imp[, keep_cols],  "训练集_插补未缩放.csv", row.names = FALSE)
write.csv(test_imp[, keep_cols],   "测试集_插补未缩放.csv", row.names = FALSE)
write.csv(train_scaled[, keep_cols], "训练集_标准化.csv", row.names = FALSE)
write.csv(test_scaled[, keep_cols],  "测试集_标准化.csv", row.names = FALSE)
cat("已导出三档训练/测试，可直接运行 python old liver.py 完整重跑。\n")
