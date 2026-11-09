# =============================================================================
# competing_risk.R
# 审稿意见 三.1③：将 7d 内死亡作为竞争事件，估计 SALI 累积发生函数（Fine-Gray）
# 依赖包：survival（通常自带）；若装了 cmprsk 可选用 cuminc
# -----------------------------------------------------------------------------
# 第一步：先在 pgAdmin 执行下面 SQL，导出 sali_competing.csv 与本脚本同目录
#   （依赖主 SQL 已建好的 sali_tb_elev / sali_ml_dataset_full / sali_base）
#
# DROP TABLE IF EXISTS mimic_work.sali_first_sali;
# CREATE TABLE mimic_work.sali_first_sali AS
#   SELECT stay_id, MIN(hr_since_icu) AS sali_hr
#   FROM mimic_work.sali_tb_elev GROUP BY stay_id;
#
# DROP TABLE IF EXISTS mimic_work.sali_competing;
# CREATE TABLE mimic_work.sali_competing AS
# SELECT f.stay_id,
#   COALESCE(s.sali_hr,
#     CASE WHEN f.no_follow_death7d = 1 THEN
#       EXTRACT(EPOCH FROM (CAST(b.dod AS timestamp) - b.icu_intime))/3600
#     ELSE 168 END) AS time_hr,
#   CASE WHEN f.label = 1 THEN 1
#        WHEN f.no_follow_death7d = 1 THEN 2
#        ELSE 0 END AS event_type
# FROM mimic_work.sali_ml_dataset_full f
# JOIN mimic_work.sali_base b ON f.stay_id = b.stay_id
# LEFT JOIN mimic_work.sali_first_sali s ON f.stay_id = s.stay_id;
#
#   SELECT * FROM mimic_work.sali_competing;  -- 导出 CSV
#
# event_type: 1 = SALI（目标事件）, 2 = 死亡（竞争事件）, 0 = 删失
# =============================================================================

here <- tryCatch(dirname(normalizePath(sys.frame(1)$ofile, mustWork = NA)),
                 error = function(e) NA_character_)
if (is.na(here) || here == "") here <- getwd()
d <- read.csv(file.path(here, "sali_competing.csv"), check.names = FALSE,
              stringsAsFactors = FALSE)
cat("样本数", nrow(d), "\n")
cat("事件计数: SALI", sum(d$event_type == 1),
    "| 竞争死亡", sum(d$event_type == 2),
    "| 删失", sum(d$event_type == 0), "\n")

d$time_day <- d$time_hr / 24
d$status <- factor(d$event_type, levels = c(0, 1, 2),
                   labels = c("censored", "SALI", "death"))

# ---- 1) 原始（未考虑竞争）KM：会把竞争死亡当删失，仅作对照 ----
km <- survival::survfit(survival::Surv(time_day, event_type == 1) ~ 1, data = d)
cat("\nKM 第7天 SALI 累积概率（忽略竞争）:",
    1 - summary(km, times = 7)$surv, "\n")

# ---- 2) Fine-Gray 累积发生函数（CIF，cmprsk）----
if (!requireNamespace("cmprsk", quietly = TRUE))
  stop("请先安装 cmprsk：install.packages('cmprsk')")
ci <- cmprsk::cuminc(ftime = d$time_day, fstatus = d$event_type)
print(ci)
# 纵轴缩到 0-0.1 放大低概率曲线；同时直接导出高清 PNG
plot(ci, xlab = "Days since ICU admission",
     ylab = "Cumulative incidence", lwd = 2, col = c(2, 4),
     ylim = c(0, 0.1))
legend("topleft", legend = c("SALI", "Death"), col = c(2, 4), lwd = 2)
png(file.path(here, "fig_cif_zoom.png"), width = 1800, height = 1400, res = 300)
plot(ci, xlab = "Days since ICU admission",
     ylab = "Cumulative incidence", lwd = 2, col = c(2, 4),
     ylim = c(0, 0.1))
legend("topleft", legend = c("SALI", "Death"), col = c(2, 4), lwd = 2)
dev.off()
cat("已保存 fig_cif_zoom.png（纵轴 0-0.1）\n")

# cuminc 列表组件名形如 "1 1"（SALI）、"1 2"（死亡）；定位 event 1 的 SALI 组件
sali_ci <- if ("1 1" %in% names(ci)) ci[["1 1"]] else ci[[1]]

# 阶梯函数在第 7 天的 CIF：取时间 <=7 的最后一个估计
k <- findInterval(7, sali_ci$time)
cif7 <- sali_ci$est[k]
cat("Fine-Gray 第7天 SALI CIF（考虑竞争死亡）:", cif7, "\n")

# 保存 CIF 曲线数据
out <- data.frame(time = sali_ci$time, cif = sali_ci$est)
write.csv(out, file.path(here, "sali_cif.csv"), row.names = FALSE)
cat("CIF 已保存: sali_cif.csv\n")
