# Runbook: DB Connection Pool Exhaustion

## Overview
DB connection pool exhaustion is a specific failure mode that manifests as high latency
and errors on the service using the database — NOT a service logic bug.

## Symptoms
- p99 latency spike on one service (typically 5-10x baseline)
- error_rate_pct elevated on same service
- saturation_pct high on same service
- Logs contain: "HikariCP connection pool exhausted", "getConnection() timed out", "FATAL: remaining connection slots reserved"

## Key Distinction from Service Bug
- DB pool exhaustion is an INFRASTRUCTURE problem, not a code bug
- `rollback` will NOT fix it — the code is fine
- `restart_service` temporarily fixes it but pool will exhaust again under load
- The correct fix is `increase_db_pool`

## Diagnosis Steps
1. `diagnostic_query` on affected service — confirm high latency + errors + saturation
2. `log_inspection` on affected service — look specifically for HikariCP or psql pool exhaustion errors
3. If you see pool exhaustion errors: proceed directly to increase_db_pool
4. Do NOT rollback or restart before trying increase_db_pool

## Remediation
- `remediation(action_type="increase_db_pool", target="user-service", params={"pool_size": 50})`
- Pool size 50 is a safe default; go higher (100) only if errors persist
- After applying: query the service again — latency should drop within 1 cycle

## Verification
- Confirm user-service p99 drops to < 200ms
- Confirm error rate drops to < 1%
- Confirm saturation drops to < 50%
- Check logs — no more HikariCP exhaustion messages
