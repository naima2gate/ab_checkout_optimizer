-- create_table.sql
--
-- This script defines the schema for the A/B testing data tables.

CREATE TABLE IF NOT EXISTS assignments (
    user_id VARCHAR PRIMARY KEY,
    exp_id VARCHAR,
    variant VARCHAR,
    bucket_ts TIMESTAMP,
    strata VARCHAR
);

CREATE TABLE IF NOT EXISTS sessions (
    session_id VARCHAR PRIMARY KEY,
    user_id VARCHAR,
    country VARCHAR,
    device VARCHAR,
    traffic_source VARCHAR,
    past_7d_gpv FLOAT,
    start_ts TIMESTAMP
);

CREATE TABLE IF NOT EXISTS events (
    event_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    ts TIMESTAMP,
    name VARCHAR,
    value FLOAT
);

CREATE TABLE IF NOT EXISTS orders (
    order_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    revenue FLOAT,
    discount FLOAT,
    var_cost FLOAT
);

CREATE TABLE IF NOT EXISTS perf (
    perf_id VARCHAR PRIMARY KEY,
    session_id VARCHAR,
    ts TIMESTAMP,
    checkout_latency_ms FLOAT
);