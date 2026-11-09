# =====================================================================
# eICU 外部验证 —— 出图（base R，无额外包依赖）
# 读取 eicu_predictions.csv（含 y 与三种概率），输出：
#   fig_eicu_roc.png          外部 ROC（AUC=0.799）
#   fig_eicu_calibration.png  冻结 vs 逻辑重校准 校准曲线
# 运行：把本脚本与 eicu_predictions.csv 放同一目录，source 即可。
# =====================================================================

# 以脚本所在目录为工作目录（兼容不同终端路径）
if (!is.null(sys.frame(1)$ofile)) {
  setwd(dirname(sys.frame(1)$ofile))
} else if (requireNamespace("rstudioapi", quietly = TRUE) &&
           rstudioapi::isAvailable()) {
  setwd(dirname(rstudioapi::getActiveDocumentContext()$path))
}

d <- read.csv("eicu_predictions.csv", check.names = FALSE)
y <- d$y

# ---- 手动计算 ROC 与 AUC（不依赖 pROC）----
roc_xy <- function(score, yy) {
  ord <- order(score, decreasing = TRUE)
  s <- score[ord]; g <- yy[ord]
  P <- sum(yy == 1); N <- sum(yy == 0)
  tpr <- c(0); fpr <- c(0)
  tp <- 0; fp <- 0
  i <- 1
  while (i <= length(s)) {
    j <- i
    while (j <= length(s) && s[j] == s[i]) {
      if (g[j] == 1) tp <- tp + 1 else fp <- fp + 1
      j <- j + 1
    }
    tpr <- c(tpr, tp / P); fpr <- c(fpr, fp / N)
    i <- j
  }
  auc <- sum(diff(fpr) * (tpr[-1] + tpr[-length(tpr)]) / 2)
  list(fpr = fpr, tpr = tpr, auc = auc)
}
r <- roc_xy(d$prob_frozen, y)

png("fig_eicu_roc.png", width = 1800, height = 1600, res = 300)
par(mar = c(4.5, 4.5, 2, 1), family = "serif")
plot(r$fpr, r$tpr, type = "l", lwd = 2.5, col = "#1f4e79",
     xlab = "1 - Specificity", ylab = "Sensitivity",
     xlim = c(0, 1), ylim = c(0, 1), las = 1)
abline(0, 1, lty = 2, col = "gray50")
legend("bottomright",
       legend = sprintf("eICU external ROC, AUC = %.3f", r$auc),
       lwd = 2.5, col = "#1f4e79", bty = "n", cex = 1.05)
dev.off()

# ---- 校准曲线：按十分位分组，观察率 vs 平均预测 ----
cal_pts <- function(score, yy, g = 10) {
  ord <- order(score)
  grp <- cut(seq_along(score), breaks = quantile(seq_along(score),
              probs = seq(0, 1, length.out = g + 1)),
             include.lowest = TRUE, labels = FALSE)
  m <- tapply(score[ord], grp, mean)
  o <- tapply(yy[ord], grp, mean)
  data.frame(pred = as.numeric(m), obs = as.numeric(o))
}
cf <- cal_pts(d$prob_frozen, y)
cr <- cal_pts(d$prob_logistic, y)

png("fig_eicu_calibration.png", width = 1800, height = 1600, res = 300)
par(mar = c(4.5, 4.5, 2, 1), family = "serif")
xmax <- max(cf$pred, cr$pred) * 1.05
ymax <- max(cf$obs, cr$obs) * 1.1
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

cat(sprintf("已输出 fig_eicu_roc.png（AUC=%.3f）与 fig_eicu_calibration.png\n", r$auc))
