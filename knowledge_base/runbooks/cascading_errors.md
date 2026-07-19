# Runbook: Cascading Error Rate Increase

## Overview
When error rates appear on multiple services simultaneously, always identify the origin service
before attempting any remediation. Treating symptoms (downstream) before the root cause wastes
time and can introduce additional instability.

## Symptoms
- error_rate_pct elevated on 2+ services
- Errors appeared first (or are highest) on one specific service
- Log errors: "502 Bad Gateway", "upstream connection refused", "circuit breaker half-open"

## Diagnosis Steps
1. Run `diagnostic_query` on ALL services (metric=`error_rate_pct`)
2. The service with the HIGHEST error rate is most likely the origin
3. Confirm with `log_inspection` — look for "502" or "connection refused" patterns
4. Check if downstream services show "cascaded_errors" in active_faults

## Root Cause Identification
- The origin service will have both high error rate AND its own upstream errors in logs
- Downstream services will reference the upstream service name in their error logs

## Remediation
- Fix origin service FIRST: `rollback` (if recent deploy) or `restart_service`
- Do NOT restart downstream services first — cascade will clear once origin is fixed
- Use `circuit_breaker` on downstream services only as a last resort to stop cascade spread

## Quarantine Gate Notes
- `restart_service` on payment-service is HIGH RISK and will be blocked
- Use `circuit_breaker` on payment-service if it's downstream of the cascade

## Verification
- After fixing origin: confirm its error_rate_pct drops to < 1%
- Downstream services should self-heal within 1-2 diagnostic cycles
