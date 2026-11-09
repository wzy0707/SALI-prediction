/* =============================================================================
   SALI 一键取数（从头到尾）—— 直接产出最终建模数据
   老年 ICU 脓毒症患者 24h-7d 新发 TBIL>2（MIMIC-IV v3.1，回顾性）
   -----------------------------------------------------------------------------
   用法：在 pgAdmin 里整段执行（F5）。脚本会自动按顺序重建全部依赖表，
        最后产出 sali_ml_final（主队列 n=3,567 / 事件 205 / 5.75%），
        该表即“最终建模数据”，右键导出 CSV 即可进入 Python。
   预期结果（已锁定，跑完应一致）：
     · sali_ml_final 主队列：3,567 行，label=205（5.75%），label_all=251
     · 超集 sali_ml_dataset_full：6,398 行（无复查 2,831 = 7d内死亡294 + 存活2,537）
     · 全 ICU 敏感①：3,749 / 260
   口径：≥65岁 + Sepsis-3 + 首次入ICU + LOS≥48h；排除慢性肝病/肝胆恶性/外伤梗死/
        机械性胆道梗阻；基线仅 TBIL≤2；24h-7d 有 TBIL 复查；结局=24h-7d 新发 TBIL>2。
   ============================================================================= */

CREATE SCHEMA IF NOT EXISTS mimic_work;

/* =============================================================================
   步骤1：最终诊断表 sali_dx_final（住院级；去点后正则，K80 末位奇数 [13579]）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_dx_final;
CREATE TABLE mimic_work.sali_dx_final AS
WITH d AS (
  SELECT hadm_id, icd_version, REPLACE(icd_code,'.','') AS code
  FROM mimiciv_hosp.diagnoses_icd
)
SELECT hadm_id,
  MAX(CASE WHEN icd_version=9  AND code ~ '^(07022|07023|07032|07033|07044|07054|V0261|V427|571[1-9]|570|5723|5728|5730|5733|5734|456[012])' THEN 1 ELSE 0 END) AS chronic9,
  MAX(CASE WHEN icd_version=10 AND code ~ '^(B18|Z225|K70|K71|K72|K73|K74|K75[1489]|K76[1367]|Z944)' THEN 1 ELSE 0 END) AS chronic10,
  MAX(CASE WHEN icd_version=9  AND code ~ '^(155|156|1977)'     THEN 1 ELSE 0 END) AS malign9,
  MAX(CASE WHEN icd_version=10 AND code ~ '^(C22|C23|C24|C787)' THEN 1 ELSE 0 END) AS malign10,
  MAX(CASE WHEN icd_version=9  AND code ~ '^864'  THEN 1 ELSE 0 END) AS trauma9,
  MAX(CASE WHEN icd_version=10 AND code ~ '^S361' THEN 1 ELSE 0 END) AS trauma10,
  MAX(CASE WHEN icd_version=9 AND (code='5762' OR code ~ '^574[0-9]1$' OR code='5752')
           THEN 1 ELSE 0 END) AS obs9,
  MAX(CASE WHEN icd_version=10 AND (code='K831' OR code='K820'
                                    OR (code ~ '^K80[0-9][13579]$'))
           THEN 1 ELSE 0 END) AS obs10,
  MAX(CASE WHEN icd_version=9 AND (code ~ '^(5761|5750|5751|5759)'
                                    OR code ~ '^574[0-9]0$' OR code ~ '^574[0-9]$')
           THEN 1 ELSE 0 END) AS inf9,
  MAX(CASE WHEN icd_version=10 AND (code ~ '^(K830|K81[0-9])'
                                    OR code ~ '^K80[0134]$' OR code='K806')
           THEN 1 ELSE 0 END) AS inf10,
  CASE
    WHEN MAX(CASE WHEN icd_version=10 AND code ~ '^(K80|K81|K83)' THEN 1
                  WHEN icd_version=9  AND code ~ '^(574|575|576)' THEN 1 ELSE 0 END)=1 THEN 'biliary'
    WHEN MAX(CASE WHEN icd_version=10 AND code ~ '^(K35|K57|K6[0-5]|K75[0-3]|A047)' THEN 1
                  WHEN icd_version=9  AND code ~ '^(540|542|562|566|567|569|00845)' THEN 1 ELSE 0 END)=1 THEN 'abdominal'
    WHEN MAX(CASE WHEN icd_version=10 AND code ~ '^(N390|N1[0-5])' THEN 1
                  WHEN icd_version=9  AND code ~ '^(590|595|5990)' THEN 1 ELSE 0 END)=1 THEN 'urinary'
    WHEN MAX(CASE WHEN icd_version=10 AND code ~ '^(J1[0-9]|J2[0-2]|J69|J85|J86|J90)' THEN 1
                  WHEN icd_version=9  AND code ~ '^(48[0-8]|507)' THEN 1 ELSE 0 END)=1 THEN 'pulmonary'
    ELSE 'other'
  END AS source_category
FROM d
GROUP BY hadm_id;

ALTER TABLE mimic_work.sali_dx_final ADD PRIMARY KEY (hadm_id);

/* =============================================================================
   步骤2：基础池 sali_base（Sepsis-3 + 老年 + 首次入ICU + LOS≥48h；含死亡字段）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_base;
CREATE TABLE mimic_work.sali_base AS
SELECT
  s.subject_id, d.hadm_id, s.stay_id,
  d.gender, d.admission_age, d.race,
  d.icu_intime, d.icu_outtime, d.los_icu,
  d.hospital_expire_flag, d.dod,
  p.anchor_year_group,
  CASE WHEN d.dod IS NOT NULL AND d.dod <= CAST(d.icu_intime AS date) + 28 THEN 1 ELSE 0 END AS death_28d,
  CASE WHEN d.dod IS NOT NULL AND d.dod <= CAST(d.icu_intime AS date) + 90 THEN 1 ELSE 0 END AS death_90d,
  CASE WHEN d.dod IS NOT NULL AND d.dod >= CAST(d.icu_intime AS date)
        AND d.dod <= CAST(d.icu_outtime AS date) THEN 1 ELSE 0 END AS icu_death
FROM mimiciv_derived.sepsis3 s
JOIN mimiciv_derived.icustay_detail d USING (stay_id)
JOIN mimiciv_hosp.patients p ON s.subject_id = p.subject_id
WHERE s.sepsis3 = TRUE
  AND d.first_icu_stay = TRUE
  AND d.admission_age >= 60
  AND d.los_icu >= 2;

ALTER TABLE mimic_work.sali_base ADD PRIMARY KEY (stay_id);

/* =============================================================================
   步骤3：24h 后肝/凝血随访长表（TBIL 取自 derived.enzyme；INR 取自 derived.coagulation）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_follow_liver;
CREATE TABLE mimic_work.sali_follow_liver AS
WITH st AS (SELECT stay_id, hadm_id, icu_intime, icu_outtime FROM mimic_work.sali_base)
SELECT stay_id, 'TB' AS lab_type, e.charttime,
       EXTRACT(EPOCH FROM (e.charttime - b.icu_intime))/3600 AS hr_since_icu,
       FLOOR(EXTRACT(EPOCH FROM (e.charttime - b.icu_intime))/3600/24) AS day_idx,
       e.bilirubin_total AS val
FROM mimiciv_derived.enzyme e
JOIN st b USING (hadm_id)
WHERE e.charttime >= b.icu_intime + INTERVAL '24 hours'
  AND e.charttime <= b.icu_outtime
  AND e.bilirubin_total IS NOT NULL
UNION ALL
SELECT stay_id, 'INR' AS lab_type, c.charttime,
       EXTRACT(EPOCH FROM (c.charttime - b.icu_intime))/3600 AS hr_since_icu,
       FLOOR(EXTRACT(EPOCH FROM (c.charttime - b.icu_intime))/3600/24) AS day_idx,
       c.inr AS val
FROM mimiciv_derived.coagulation c
JOIN st b USING (hadm_id)
WHERE c.charttime >= b.icu_intime + INTERVAL '24 hours'
  AND c.charttime <= b.icu_outtime
  AND c.inr IS NOT NULL;

CREATE INDEX idx_sali_follow ON mimic_work.sali_follow_liver (stay_id, day_idx);

/* =============================================================================
   步骤4：ICU 日级聚合（day_idx：1=24-48h，…，6=144-168h；>6 为第7天后）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_daily_liver;
CREATE TABLE mimic_work.sali_daily_liver AS
SELECT stay_id, day_idx,
       MAX(val) FILTER (WHERE lab_type='TB')  AS tb_max,
       MAX(val) FILTER (WHERE lab_type='INR') AS inr_max
FROM mimic_work.sali_follow_liver
GROUP BY stay_id, day_idx;

CREATE INDEX idx_sali_daily ON mimic_work.sali_daily_liver (stay_id, day_idx);

/* =============================================================================
   步骤5：结局表 sali_outcome（每 stay 一行）
     sali_7d_tb=主结局label（7d 内 TBIL>2）；sali_7d=同日双达标；sali_7d_loose=各自达标
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_outcome;
CREATE TABLE mimic_work.sali_outcome AS
SELECT
  stay_id,
  COUNT(*) FILTER (WHERE day_idx <= 6 AND tb_max  IS NOT NULL) AS tb_follow_n,
  COUNT(*) FILTER (WHERE day_idx <= 6 AND inr_max IS NOT NULL) AS inr_follow_n,
  COUNT(*) FILTER (WHERE tb_max  IS NOT NULL) AS tb_follow_n_all,
  COUNT(*) FILTER (WHERE inr_max IS NOT NULL) AS inr_follow_n_all,
  CASE WHEN MAX(CASE WHEN day_idx <= 6 AND tb_max > 2 AND inr_max > 1.5 THEN 1 ELSE 0 END) = 1
       THEN 1 ELSE 0 END AS sali_7d,
  CASE WHEN MAX(CASE WHEN day_idx <= 6 AND tb_max > 2 THEN 1 ELSE 0 END) = 1
         AND MAX(CASE WHEN day_idx <= 6 AND inr_max > 1.5 THEN 1 ELSE 0 END) = 1
       THEN 1 ELSE 0 END AS sali_7d_loose,
  CASE WHEN MAX(CASE WHEN day_idx <= 6 AND tb_max > 2 THEN 1 ELSE 0 END) = 1
       THEN 1 ELSE 0 END AS sali_7d_tb,
  CASE WHEN MAX(CASE WHEN day_idx >= 1 AND tb_max > 2 AND inr_max > 1.5 THEN 1 ELSE 0 END) = 1
       THEN 1 ELSE 0 END AS sali_all
FROM mimic_work.sali_daily_liver
GROUP BY stay_id;

ALTER TABLE mimic_work.sali_outcome ADD PRIMARY KEY (stay_id);

/* =============================================================================
   步骤6：首24h 特征表（RDW / 有创通气 / NE 当量 / 激素·镇静）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_cbc_24h;
CREATE TABLE mimic_work.sali_cbc_24h AS
SELECT b.stay_id, MAX(c.rdw) AS rdw_max
FROM mimic_work.sali_base b
JOIN mimiciv_derived.complete_blood_count c ON b.hadm_id = c.hadm_id
WHERE c.charttime >= b.icu_intime AND c.charttime < b.icu_intime + INTERVAL '24 hours'
  AND c.rdw IS NOT NULL
GROUP BY b.stay_id;

DROP TABLE IF EXISTS mimic_work.sali_vent_24h;
CREATE TABLE mimic_work.sali_vent_24h AS
SELECT b.stay_id,
       MAX(CASE WHEN v.ventilation_status = 'InvasiveVent' THEN 1 ELSE 0 END) AS mech_vent
FROM mimic_work.sali_base b
JOIN mimiciv_derived.ventilation v ON b.stay_id = v.stay_id
WHERE v.starttime <  b.icu_intime + INTERVAL '24 hours'
  AND v.endtime   >  b.icu_intime
GROUP BY b.stay_id;

DROP TABLE IF EXISTS mimic_work.sali_neq_24h;
CREATE TABLE mimic_work.sali_neq_24h AS
SELECT b.stay_id, MAX(n.norepinephrine_equivalent_dose) AS neq_max
FROM mimic_work.sali_base b
JOIN mimiciv_derived.norepinephrine_equivalent_dose n ON b.stay_id = n.stay_id
WHERE n.starttime <  b.icu_intime + INTERVAL '24 hours'
  AND n.endtime   >  b.icu_intime
GROUP BY b.stay_id;

DROP TABLE IF EXISTS mimic_work.sali_med_24h;
CREATE TABLE mimic_work.sali_med_24h AS
SELECT b.stay_id,
  MAX(CASE WHEN pr.drug ILIKE '%hydrocortisone%' OR pr.drug ILIKE '%dexamethasone%'
            OR pr.drug ILIKE '%methylpred%' OR pr.drug ILIKE '%prednisol%'
            OR pr.drug ILIKE '%prednisone%' OR pr.drug ILIKE '%cortisone%'
            OR pr.drug ILIKE '%betamethasone%'
       THEN 1 ELSE 0 END) AS steroid_24h,
  MAX(CASE WHEN pr.drug ILIKE '%propofol%' OR pr.drug ILIKE '%midazolam%'
            OR pr.drug ILIKE '%lorazepam%' OR pr.drug ILIKE '%fentanyl%'
            OR pr.drug ILIKE '%dexmedetomidine%' OR pr.drug ILIKE '%morphine%'
       THEN 1 ELSE 0 END) AS sedative_24h
FROM mimic_work.sali_base b
JOIN mimiciv_hosp.prescriptions pr ON b.hadm_id = pr.hadm_id
WHERE pr.starttime IS NOT NULL
  AND pr.starttime >= b.icu_intime
  AND pr.starttime <  b.icu_intime + INTERVAL '24 hours'
GROUP BY b.stay_id;

/* =============================================================================
   步骤7：超集 sali_ml_dataset_full（6,398 行；全部特征 + 标志，不要求有复查）
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_ml_dataset_full;
CREATE TABLE mimic_work.sali_ml_dataset_full AS
WITH tb_all AS (
  SELECT stay_id, MAX(CASE WHEN lab_type='TB' AND day_idx>=1 AND val>2 THEN 1 ELSE 0 END) AS ev_all
  FROM mimic_work.sali_follow_liver GROUP BY stay_id
),
cult AS (
  SELECT stay_id, MAX(COALESCE(positive_culture,0)) AS positive_culture
  FROM mimiciv_derived.suspicion_of_infection GROUP BY stay_id
)
SELECT
  b.stay_id, b.subject_id, b.hadm_id,
  CASE WHEN COALESCE(o.tb_follow_n,0) >= 1 THEN 1 ELSE 0 END     AS has_follow_7d,
  CASE WHEN COALESCE(o.tb_follow_n_all,0) >= 1 THEN 1 ELSE 0 END AS has_follow_all,
  COALESCE(o.tb_follow_n,0) AS tb_follow_n, COALESCE(o.tb_follow_n_all,0) AS tb_follow_n_all,
  COALESCE(o.sali_7d_tb,0) AS label,
  CASE WHEN COALESCE(o.tb_follow_n,0) >= 1 THEN COALESCE(o.sali_7d_tb,0) ELSE NULL END AS label_known,
  COALESCE(ta.ev_all,0)    AS label_all,
  CASE WHEN fl.inr_max IS NOT NULL AND fl.inr_max <= 1.5 AND o.sali_7d = 1 THEN 1 ELSE 0 END AS severe_new_sameday,
  CASE WHEN fl.inr_max IS NOT NULL AND fl.inr_max <= 1.5 AND o.sali_7d_loose = 1 THEN 1 ELSE 0 END AS severe_new_loose,
  COALESCE(o.sali_7d,0) AS severe_dual_anybase,
  b.icu_death, b.death_28d, b.death_90d,
  CASE WHEN b.dod IS NOT NULL AND b.dod <= CAST(b.icu_intime AS date) + 7 THEN 1 ELSE 0 END AS death_7d,
  CASE WHEN COALESCE(o.tb_follow_n,0)=0 AND b.dod IS NOT NULL
            AND b.dod <= CAST(b.icu_intime AS date) + 7 THEN 1 ELSE 0 END AS no_follow_death7d,
  CASE WHEN COALESCE(o.tb_follow_n,0)=0 AND NOT (b.dod IS NOT NULL
            AND b.dod <= CAST(b.icu_intime AS date) + 7) THEN 1 ELSE 0 END AS no_follow_alive,
  CASE WHEN fl.inr_max IS NULL THEN 1 ELSE 0 END AS inr0_missing,
  CASE WHEN fl.inr_max IS NOT NULL AND fl.inr_max <= 1.5 THEN 1 ELSE 0 END AS inr0_normal,
  CASE WHEN fl.inr_max > 1.5 THEN 1 ELSE 0 END AS inr0_high,
  CASE WHEN b.admission_age BETWEEN 65 AND 74 THEN '65-74' WHEN b.admission_age >= 75 THEN '>=75' END AS age_group,
  b.anchor_year_group,
  b.admission_age AS age, b.gender,
  CASE WHEN b.race ILIKE 'White%' THEN 'White' WHEN b.race ILIKE 'Black%' THEN 'Black' ELSE 'Other' END AS race3,
  ch.charlson_comorbidity_index AS charlson,
  fv.heart_rate_mean AS hr_mean, fv.mbp_mean, fv.resp_rate_mean AS rr_mean,
  fv.temperature_max AS temp_max, fv.spo2_mean,
  fg.gcs_min, fu.urineoutput AS uo_24h, fs.sofa,
  fl.wbc_max, fl.platelets_min AS plt_min, cbc.rdw_max,
  fl.sodium_max, fl.bicarbonate_min, fl.glucose_max, fl.creatinine_max,
  fl.albumin_min, bg.lactate_max, bg.pao2fio2ratio_min AS pao2fio2_min,
  fl.bilirubin_total_max AS tb0_max, fl.alt_max, fl.ast_max, fl.inr_max,
  dx.source_category, COALESCE(cu.positive_culture,0) AS positive_culture,
  COALESCE(ve.mech_vent,0) AS mech_vent,
  CASE WHEN ne.stay_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor,
  COALESCE(fr.dialysis_present,0) AS crrt,
  COALESCE(me.steroid_24h,0) AS steroid_24h, COALESCE(me.sedative_24h,0) AS sedative_24h,
  COALESCE(ch.congestive_heart_failure,0) AS chf,
  COALESCE(ch.diabetes_without_cc,0)+COALESCE(ch.diabetes_with_cc,0) AS diabetes,
  COALESCE(ch.chronic_pulmonary_disease,0) AS copd, COALESCE(ch.renal_disease,0) AS ckd,
  COALESCE(ch.malignant_cancer,0)+COALESCE(ch.metastatic_solid_tumor,0) AS cancer,
  fl.hemoglobin_min AS hgb_min, fl.potassium_max, fl.aniongap_max, fl.bun_max, bg.ph_min,
  fl.alp_max, fl.ggt_max, fl.pt_max, fl.ptt_max, ne.neq_max,
  oi.oasis, sp.sapsii, ap.apsiii, lo.lods, b.los_icu
FROM mimic_work.sali_base b
LEFT JOIN mimic_work.sali_dx_final dx ON b.hadm_id = dx.hadm_id
LEFT JOIN mimiciv_derived.first_day_lab      fl ON b.stay_id = fl.stay_id
LEFT JOIN mimiciv_derived.first_day_vitalsign fv ON b.stay_id = fv.stay_id
LEFT JOIN mimiciv_derived.first_day_bg       bg ON b.stay_id = bg.stay_id
LEFT JOIN mimiciv_derived.first_day_sofa     fs ON b.stay_id = fs.stay_id
LEFT JOIN mimiciv_derived.first_day_gcs      fg ON b.stay_id = fg.stay_id
LEFT JOIN mimiciv_derived.first_day_urine_output fu ON b.stay_id = fu.stay_id
LEFT JOIN mimiciv_derived.first_day_rrt      fr ON b.stay_id = fr.stay_id
LEFT JOIN mimiciv_derived.charlson           ch ON b.hadm_id = ch.hadm_id
LEFT JOIN mimiciv_derived.oasis              oi ON b.stay_id = oi.stay_id
LEFT JOIN mimiciv_derived.sapsii             sp ON b.stay_id = sp.stay_id
LEFT JOIN mimiciv_derived.apsiii             ap ON b.stay_id = ap.stay_id
LEFT JOIN mimiciv_derived.lods               lo ON b.stay_id = lo.stay_id
LEFT JOIN mimic_work.sali_cbc_24h            cbc ON b.stay_id = cbc.stay_id
LEFT JOIN mimic_work.sali_vent_24h           ve  ON b.stay_id = ve.stay_id
LEFT JOIN mimic_work.sali_neq_24h            ne  ON b.stay_id = ne.stay_id
LEFT JOIN mimic_work.sali_med_24h            me  ON b.stay_id = me.stay_id
LEFT JOIN mimic_work.sali_outcome            o   ON b.stay_id = o.stay_id
LEFT JOIN tb_all                             ta  ON b.stay_id = ta.stay_id
LEFT JOIN cult                               cu  ON b.stay_id = cu.stay_id
WHERE b.admission_age >= 65
  AND (COALESCE(dx.chronic9,0)+COALESCE(dx.chronic10,0)
      +COALESCE(dx.malign9,0)+COALESCE(dx.malign10,0)
      +COALESCE(dx.trauma9,0)+COALESCE(dx.trauma10,0)
      +COALESCE(dx.obs9,0)+COALESCE(dx.obs10,0)) = 0
  AND fl.bilirubin_total_max IS NOT NULL
  AND fl.bilirubin_total_max <= 2;

ALTER TABLE mimic_work.sali_ml_dataset_full ADD PRIMARY KEY (stay_id);

/* =============================================================================
   步骤8：最终建模数据 sali_ml_final（主队列 = 3,567）★ 直接导出这张表
   ============================================================================= */
DROP TABLE IF EXISTS mimic_work.sali_ml_final;
CREATE TABLE mimic_work.sali_ml_final AS
SELECT *
FROM mimic_work.sali_ml_dataset_full
WHERE has_follow_7d = 1;

ALTER TABLE mimic_work.sali_ml_final ADD PRIMARY KEY (stay_id);

/* =============================================================================
   步骤9：核对（预期：主队列 3,567/205/5.75；超集 6,398；全ICU 3,749/260）
   ============================================================================= */
-- 9-A 主队列 + 严重表型分母
SELECT COUNT(*) AS n, SUM(label) AS ev7d, ROUND(AVG(label)*100,2) AS pct_7d,
       SUM(label_all) AS ev_allwindow_in_7dcohort,
       SUM(severe_new_sameday) AS severe_sameday, SUM(severe_new_loose) AS severe_loose,
       SUM(inr0_missing) AS inr_missing, SUM(inr0_normal) AS inr_le15, SUM(inr0_high) AS inr_gt15
FROM mimic_work.sali_ml_final;

-- 9-B 超集总行数 + 无复查者死亡截尾分解
SELECT COUNT(*) AS n_total,
       SUM(has_follow_7d) AS n_follow7d, SUM(1-has_follow_7d) AS n_no_follow7d,
       SUM(no_follow_death7d) AS n_nofollow_death7d, SUM(no_follow_alive) AS n_nofollow_alive,
       SUM(label) AS label_worstcase, ROUND(AVG(label)*100,2) AS pct_lowerbound,
       SUM(icu_death) AS icu_death, SUM(death_28d) AS death_28d
FROM mimic_work.sali_ml_dataset_full;

-- 9-C 全 ICU 独立队列（敏感①）
SELECT COUNT(*) FILTER (WHERE has_follow_all=1) AS n_fullicu,
       SUM(label_all) FILTER (WHERE has_follow_all=1) AS ev_fullicu,
       COUNT(*) FILTER (WHERE has_follow_7d=1) AS n_7d,
       SUM(label_all) FILTER (WHERE has_follow_7d=1) AS ev_allwindow_within7d
FROM mimic_work.sali_ml_dataset_full;

/* =============================================================================
   导出：在 pgAdmin 左侧或查询里
     SELECT * FROM mimic_work.sali_ml_final;
   点 Data Output 下载图标，导出 CSV（UTF-8），即为最终建模数据。
   ============================================================================= */
