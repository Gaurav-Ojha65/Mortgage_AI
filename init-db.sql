-- Initialize Mortgage AI Database
-- Creates tables for decisions, audit logs, and model tracking

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Decisions table
CREATE TABLE IF NOT EXISTS decisions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    income DECIMAL(12, 2),
    loan_amount DECIMAL(12, 2),
    credit_score INTEGER,
    interest_rate DECIMAL(5, 2),
    loan_term INTEGER,
    existing_loans INTEGER,
    decision VARCHAR(50),
    approval_probability DECIMAL(5, 4),
    default_probability DECIMAL(5, 4),
    risk_level VARCHAR(50),
    emi DECIMAL(10, 2),
    model_version VARCHAR(50),
    shap_explanation JSONB,
    processing_time_ms INTEGER,
    client_ip INET,
    user_agent TEXT
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    action VARCHAR(100) NOT NULL,
    user_id VARCHAR(255),
    details JSONB,
    client_ip INET,
    severity VARCHAR(20) DEFAULT 'INFO'
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    version VARCHAR(50) UNIQUE NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    model_type VARCHAR(100),
    metrics JSONB,
    file_path TEXT,
    is_active BOOLEAN DEFAULT FALSE,
    deployed_by VARCHAR(255)
);

-- Drift detection logs
CREATE TABLE IF NOT EXISTS drift_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    drift_detected BOOLEAN,
    drift_score DECIMAL(5, 4),
    drift_type VARCHAR(50),
    feature_drifts JSONB,
    recommendation TEXT
);

-- Rate limiting table
CREATE TABLE IF NOT EXISTS rate_limit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    timestamp TIMESTAMPTZ DEFAULT NOW(),
    user_id VARCHAR(255),
    endpoint VARCHAR(255),
    requests_count INTEGER,
    limited BOOLEAN DEFAULT FALSE
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_decisions_timestamp ON decisions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_decisions_user_id ON decisions(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_timestamp ON audit_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_audit_logs_action ON audit_logs(action);
CREATE INDEX IF NOT EXISTS idx_drift_logs_timestamp ON drift_logs(timestamp DESC);

-- Create views for analytics
CREATE OR REPLACE VIEW daily_decisions AS
SELECT
    DATE(timestamp) as date,
    COUNT(*) as total_decisions,
    SUM(CASE WHEN decision = 'APPROVED' THEN 1 ELSE 0 END) as approved_count,
    SUM(CASE WHEN decision = 'REJECTED' THEN 1 ELSE 0 END) as rejected_count,
    AVG(approval_probability) as avg_approval_probability
FROM decisions
GROUP BY DATE(timestamp)
ORDER BY date DESC;

-- Function to archive old decisions
CREATE OR REPLACE FUNCTION archive_old_decisions(days INTEGER DEFAULT 90)
RETURNS INTEGER AS $$
DECLARE
    archived_count INTEGER;
BEGIN
    -- In production, this would move data to archive table
    -- For now, just return count
    SELECT COUNT(*) INTO archived_count
    FROM decisions
    WHERE timestamp < NOW() - INTERVAL '1 day' * days;

    RETURN archived_count;
END;
$$ LANGUAGE plpgsql;
