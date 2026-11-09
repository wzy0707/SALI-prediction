-- =====================================================================
-- eICU-CRD 外部验证 —— 队列提取（PostgreSQL）
-- 目标：在 eICU 中镜像 MIMIC 主队列纳排，导出五变量 + 结局，供
--       eicu_external_validation.py 套用 MIMIC 五变量模型系数。
-- 时间口径：以入 ICU 为 0，offset 单位为分钟（24h=1440，7d=10080）。
-- 检验名医院间有差异，统一用 lower(labname) 模糊匹配，避免大小写/命名差异。
-- 运行前确认已把 eICU 表（patient/lab/diagnosis）载入本库。
-- =====================================================================

-- 0. 基础合格：首次入 ICU、年龄数值化（>89 按 90 计）
DROP TABLE IF EXISTS eicu_elig;
CREATE TEMP TABLE eicu_elig AS
SELECT patientunitstayid, patienthealthsystemstayid, hospitalid,
       gender, unittype, unitvisitnumber, unitdischargeoffset,
       unitdischargestatus, hospitaldischargestatus,
       CASE WHEN age = '> 89' THEN 90.0
            ELSE CAST(NULLIF(age, '') AS numeric) END AS age_num
FROM patient
WHERE unitvisitnumber = 1;

-- 1. 脓毒症识别（结构化诊断；eICU 以诊断表为主，附 ICD-9 码）
DROP TABLE IF EXISTS eicu_sepsis;
CREATE TEMP TABLE eicu_sepsis AS
SELECT DISTINCT patientunitstayid
FROM diagnosis
WHERE lower(diagnosisstring) LIKE '%sepsis%'
   OR lower(diagnosisstring) LIKE '%septic%'
   OR icd9code LIKE '038,%' OR icd9code LIKE '038.%'
   OR icd9code LIKE '995.91%' OR icd9code LIKE '995.92%'
   OR icd9code LIKE '785.52%';

-- 2. 排除诊断：慢性肝病 / 肝胆恶性 / 肝胆外伤 / 机械性胆道梗阻
--    （沿用 MIMIC 锁定的 ICD-9 规则，eICU 以 ICD-9 为主，兼容 ICD-10）
DROP TABLE IF EXISTS eicu_excl;
CREATE TEMP TABLE eicu_excl AS
SELECT DISTINCT patientunitstayid
FROM diagnosis
WHERE icd9code ~ '^(07022|07023|07032|07033|07044|07054|V0261|V427|571[1-9]|570|5723|5728|5730|5733|5734|456[012])'
   OR icd9code ~ '^(B18|Z225|K70|K71|K72|K73|K74|K75[1489]|K76[1367]|Z944)'
   OR icd9code ~ '^(155|156|1977)'
   OR icd9code ~ '^(C22|C23|C24|C787)'
   OR icd9code ~ '^864'
   OR icd9code ~ '^S361'
   OR icd9code = '5762' OR icd9code ~ '^574[0-9]1$' OR icd9code = '5752'
   OR icd9code = 'K831' OR icd9code = 'K820' OR icd9code ~ '^K80[0-9][13579]$';

-- 3. 基线实验室（入 ICU 24h 内最早一次）
DROP TABLE IF EXISTS eicu_b0;  CREATE TEMP TABLE eicu_b0 AS
SELECT DISTINCT ON (patientunitstayid) patientunitstayid, labresult AS base_tbil
FROM lab
WHERE lower(labname) LIKE '%bilirubin%' AND lower(labname) NOT LIKE '%direct%'
  AND labresultoffset BETWEEN 0 AND 1440
ORDER BY patientunitstayid, labresultoffset;

DROP TABLE IF EXISTS eicu_ptt;  CREATE TEMP TABLE eicu_ptt AS
SELECT DISTINCT ON (patientunitstayid) patientunitstayid, labresult AS base_ptt
FROM lab
WHERE lower(labname) LIKE '%ptt%' AND labresultoffset BETWEEN 0 AND 1440
ORDER BY patientunitstayid, labresultoffset;

DROP TABLE IF EXISTS eicu_na;  CREATE TEMP TABLE eicu_na AS
SELECT DISTINCT ON (patientunitstayid) patientunitstayid, labresult AS base_sodium
FROM lab
WHERE lower(labname) LIKE '%sodium%' AND labresultoffset BETWEEN 0 AND 1440
ORDER BY patientunitstayid, labresultoffset;

DROP TABLE IF EXISTS eicu_plt;  CREATE TEMP TABLE eicu_plt AS
SELECT DISTINCT ON (patientunitstayid) patientunitstayid, labresult AS base_platelets
FROM lab
WHERE lower(labname) LIKE '%platelet%' AND labresultoffset BETWEEN 0 AND 1440
ORDER BY patientunitstayid, labresultoffset;

DROP TABLE IF EXISTS eicu_lac;  CREATE TEMP TABLE eicu_lac AS
SELECT DISTINCT ON (patientunitstayid) patientunitstayid, labresult AS base_lactate
FROM lab
WHERE lower(labname) LIKE '%lactate%' AND labresultoffset BETWEEN 0 AND 1440
ORDER BY patientunitstayid, labresultoffset;

-- 4. 24h–7d 随访胆红素：是否新发 TBIL>2
DROP TABLE IF EXISTS eicu_follow;
CREATE TEMP TABLE eicu_follow AS
SELECT patientunitstayid,
       COUNT(*) AS n_follow_tbil,
       MIN(CASE WHEN labresult > 2 THEN labresultoffset END) AS first_hi_offset
FROM lab
WHERE lower(labname) LIKE '%bilirubin%' AND lower(labname) NOT LIKE '%direct%'
  AND labresultoffset BETWEEN 1440 AND 10080
GROUP BY patientunitstayid;

-- 5. 汇总建模集（保存为实体表，便于导出）
DROP TABLE IF EXISTS eicu_sali_validation;
CREATE TABLE eicu_sali_validation AS
SELECT e.patientunitstayid, e.hospitalid, e.gender, e.age_num,
       b.base_tbil AS "Baseline TBIL",
       p.base_ptt AS PTT,
       n.base_sodium AS Sodium,
       pl.base_platelets AS Platelets,
       l.base_lactate AS Lactate,
       CASE WHEN f.first_hi_offset IS NOT NULL THEN 1 ELSE 0 END AS Outcome
FROM eicu_elig e
JOIN eicu_sepsis s  ON e.patientunitstayid = s.patientunitstayid
JOIN eicu_follow f  ON e.patientunitstayid = f.patientunitstayid
JOIN eicu_b0 b      ON e.patientunitstayid = b.patientunitstayid
JOIN eicu_ptt p     ON e.patientunitstayid = p.patientunitstayid
JOIN eicu_na n      ON e.patientunitstayid = n.patientunitstayid
JOIN eicu_plt pl    ON e.patientunitstayid = pl.patientunitstayid
JOIN eicu_lac l     ON e.patientunitstayid = l.patientunitstayid
LEFT JOIN eicu_excl x ON e.patientunitstayid = x.patientunitstayid
WHERE e.age_num >= 65
  AND e.unitdischargeoffset >= 2880          -- LOS>=48h
  AND b.base_tbil <= 2                       -- 基线 TBIL<=2
  AND x.patientunitstayid IS NULL;           -- 未命中排除诊断

-- 6. 汇总与导出
SELECT COUNT(*) AS n,
       SUM(outcome) AS events,
       ROUND((AVG(outcome) * 100)::numeric, 2) AS event_rate_pct
FROM eicu_sali_validation;

-- 导出：右键该表 → Export Data → CSV（UTF-8），文件名 eicu_sali_validation.csv，
-- 与 eicu_external_validation.py 放同一目录后运行。
SELECT * FROM eicu_sali_validation;
