-- check_quality.sql
--
-- This script runs quality control checks on the A/B test data.

-- Check 1: Assignment Integrity (one exposure per user)
-- This query identifies any users who were assigned to multiple experiment variants, which should not happen
-- with sticky assignment.
SELECT
    user_id,
    COUNT(DISTINCT variant) AS variant_count
FROM assignments
GROUP BY 1
HAVING COUNT(DISTINCT variant) > 1;

-- Check 2: Strata Balance
-- This query checks for the balance of user counts across variants for each stratum (country x device).
-- The fractions should be roughly equal (close to 0.5 for a two-variant test).
WITH strata_counts AS (
    SELECT
        strata,
        variant,
        COUNT(DISTINCT user_id) AS user_count
    FROM assignments
    GROUP BY 1, 2
),
total_strata_counts AS (
    SELECT
        strata,
        SUM(user_count) AS total_users
    FROM strata_counts
    GROUP BY 1
)
SELECT
    s.strata,
    s.variant,
    s.user_count,
    t.total_users,
    (s.user_count * 1.0 / t.total_users) AS proportion
FROM strata_counts s
JOIN total_strata_counts t ON s.strata = t.strata
ORDER BY s.strata, s.variant;

-- Check 3: Missingness in Key Covariate
-- This query identifies any users in the experiment who are missing a value for the pre-experiment `past_7d_gpv`
-- covariate, which is required for CUPED analysis.
SELECT
    a.user_id,
    s.past_7d_gpv
FROM assignments a
LEFT JOIN sessions s ON a.user_id = s.user_id
WHERE s.past_7d_gpv IS NULL;

-- Check 4: Data Consistency
-- This query verifies that all orders and performance metrics are correctly linked back to an existing session.
SELECT COUNT(*) FROM orders WHERE session_id NOT IN (SELECT session_id FROM sessions);
SELECT COUNT(*) FROM perf WHERE session_id NOT IN (SELECT session_id FROM sessions);