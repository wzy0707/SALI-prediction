# =============================================================================
# B9 · Fig. A.8 — Cumulative incidence of SALI and competing death over 7 days
# 依赖：survival（自带）、cmprsk（已安装）
# 输入：sali_competing.csv（由 competing_risk.R 文档中的 SQL 导出，与本脚本同目录）
# 输出：矢量 PDF（投稿用）+ 300dpi TIFF（备份）
# 核对：Fine-Gray day-7 SALI CIF ≈ 0.0320，KM day-7 ≈ 0.0326
# =============================================================================

# ---- 脚本目录锚定，自动定位输入文件 ----
here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile, mustWork = NA)),
                 error = function(e) NA_character_)
if (is.na(here) || here == "") here <- getwd()

cand <- c(file.path(here, "sali_competing.csv"),
          file.path(dirname(here), "sali_competing.csv"))
fpath <- cand[file.exists(cand)][1]
if (is.na(fpath))
  stop("未找到 sali_competing.csv，请先按 competing_risk.R 中的 SQL 导出并放在本脚本同目录。")

d <- read.csv(fpath, check.names = FALSE, stringsAsFactors = FALSE)
cat("样本数", nrow(d),
    "| SALI", sum(d$event_type == 1),
    "| 竞争死亡", sum(d$event_type == 2),
    "| 删失", sum(d$event_type == 0), "\n")

d$time_day <- d$time_hr / 24

# ---- 1) KM（忽略竞争，对照）----
km <- survival::survfit(survival::Surv(time_day, event_type == 1) ~ 1, data = d)
km7 <- 1 - summary(km, times = 7, extend = TRUE)$surv
cat("KM day-7 SALI 累积概率（忽略竞争）:", round(km7, 4), "\n")

# ---- 2) Fine-Gray 累积发生函数 ----
stopifnot(requireNamespace("cmprsk", quietly = TRUE))
ci <- cmprsk::cuminc(ftime = d$time_day, fstatus = d$event_type)

# 稳健定位组件并清洗：去除 NA/非有限值，按时间排序（正常数据下为恒等操作）
get_comp <- function(ev) {
  nm <- paste("1", ev)
  comp <- if (nm %in% names(ci)) ci[[nm]] else ci[[ev]]
  tt <- comp$time; ee <- comp$est
  ok <- is.finite(tt) & is.finite(ee)
  tt <- tt[ok]; ee <- ee[ok]
  o <- order(tt)
  list(time = tt[o], est = ee[o])
}
sali <- get_comp(1)      # 事件 1 = SALI
death <- get_comp(2)     # 事件 2 = 竞争死亡

cat("原始 time_hr 非有限值数:", sum(!is.finite(d$time_hr)),
    "| 组件:", paste(intersect(c("1 1", "1 2"), names(ci)), collapse = ", "), "\n")

# 取第 7 天（含）最后一个阶梯估计，不依赖 findInterval 的排序假设
cif_at <- function(comp, t0 = 7) {
  idx <- which(comp$time <= t0)
  if (length(idx) == 0) 0 else comp$est[idx[length(idx)]]
}
cif_sali <- cif_at(sali)
cif_death <- cif_at(death)
cat("Fine-Gray day-7 CIF — SALI:", round(cif_sali, 4),
    "| 竞争死亡:", round(cif_death, 4), "\n")

# ---- 3) 发布级绘图（矢量 PDF + TIFF）----
COL_SALI <- "#D6604D"
COL_DEATH <- "#2166AC"
YLIM <- c(0, 0.10)

draw <- function() {
  par(mar = c(4.6, 4.8, 1.6, 1.4), las = 1)
  tryCatch(par(family = "Times"),
           error = function(e) par(family = "serif"))
  plot(0, type = "n", xlim = c(0, 7), ylim = YLIM, bty = "l",
       xlab = "Days since ICU admission",
       ylab = "Cumulative incidence",
       cex.lab = 1.25, cex.axis = 1.05)
  add_step <- function(comp, col, lty) {
    t <- c(0, comp$time); e <- c(0, comp$est)
    lines(t, e, type = "s", col = col, lty = lty, lwd = 2.4)
  }
  add_step(sali, COL_SALI, 1)
  add_step(death, COL_DEATH, 2)
  legend("topleft", legend = c("SALI", "Competing death"),
         col = c(COL_SALI, COL_DEATH), lty = c(1, 2), lwd = 2.4,
         bty = "n", cex = 1.15, inset = 0.02)
}

pdf_out <- file.path(here, "Fig A8 Cumulative incidence.pdf")
cairo_pdf(pdf_out, width = 8.2, height = 6.6)
draw(); dev.off()

tiff_out <- file.path(here, "Fig A8 Cumulative incidence.tiff")
tiff(tiff_out, width = 3000, height = 2400, res = 300, compression = "lzw")
draw(); dev.off()

cat("\n已导出:\n ", pdf_out, "\n ", tiff_out, "\n")
write.csv(data.frame(time = sali$time, cif_sali = sali$est),
          file.path(here, "sali_cif.csv"), row.names = FALSE)
