# =====================================================================
# B10 · Figure 4(a)(b) 矢量版（base R，无额外包依赖）
# 读取 eicu_predictions.csv（含 y 与三种概率），输出：
#   Fig 4a eICU ROC.pdf / .tiff          外部 ROC（AUC=0.799）
#   Fig 4b eICU calibration.pdf / .tiff  冻结 vs 逻辑重校准
# 把本脚本与 eicu_predictions.csv 放同一目录，source 即可。
# =====================================================================

if (!is.null(sys.frame(1)$ofile)) {
  here <- dirname(sys.frame(1)$ofile)
} else if (requireNamespace("rstudioapi", quietly = TRUE) &&
           rstudioapi::isAvailable()) {
  here <- dirname(rstudioapi::getActiveDocumentContext()$path)
} else {
  here <- getwd()
}
setwd(here)

d <- read.csv("eicu_predictions.csv", check.names = FALSE)
y <- d$y

# ---- 手动 ROC / AUC ----
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
  list(fpr = fpr, tpr = tpr,
       auc = sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2))
}
r <- roc_xy(d$prob_frozen, y)

draw_roc <- function() {
  par(mar = c(4.5, 4.5, 2, 1), las = 1)
  tryCatch(par(family = "Times"), error = function(e) par(family = "serif"))
  plot(r$fpr, r$tpr, type = "l", lwd = 2.5, col = "#1f4e79",
       xlab = "1 - Specificity", ylab = "Sensitivity",
       xlim = c(0, 1), ylim = c(0, 1))
  abline(0, 1, lty = 2, col = "gray50")
  legend("bottomright",
         legend = sprintf("eICU external ROC, AUC = %.3f", r$auc),
         lwd = 2.5, col = "#1f4e79", bty = "n", cex = 1.05)
}

# ---- 校准（十分位）----
cal_pts <- function(score, yy, g = 10) {
  ord <- order(score)
  grp <- cut(seq_along(score),
             breaks = quantile(seq_along(score),
                               probs = seq(0, 1, length.out = g + 1)),
             include.lowest = TRUE, labels = FALSE)
  data.frame(pred = as.numeric(tapply(score[ord], grp, mean)),
             obs = as.numeric(tapply(yy[ord], grp, mean)))
}
cf <- cal_pts(d$prob_frozen, y)
cr <- cal_pts(d$prob_logistic, y)

draw_cal <- function() {
  par(mar = c(4.5, 4.5, 2, 1), las = 1)
  tryCatch(par(family = "Times"), error = function(e) par(family = "serif"))
  xmax <- max(cf$pred, cr$pred) * 1.05
  ymax <- max(cf$obs, cr$obs) * 1.1
  plot(NA, xlim = c(0, xmax), ylim = c(0, ymax),
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
}

# ---- 输出矢量 PDF + TIFF 备份 ----
cairo_pdf("Fig 4a eICU ROC.pdf", width = 7.2, height = 6.6)
draw_roc(); dev.off()
cairo_pdf("Fig 4b eICU calibration.pdf", width = 7.2, height = 6.6)
draw_cal(); dev.off()

tiff("Fig 4a eICU ROC.tiff", 1800, 1600, res = 300, compression = "lzw")
draw_roc(); dev.off()
tiff("Fig 4b eICU calibration.tiff", 1800, 1600, res = 300,
     compression = "lzw")
draw_cal(); dev.off()

cat(sprintf("Figure 4 矢量版已输出，ROC AUC=%.3f\n", r$auc))
