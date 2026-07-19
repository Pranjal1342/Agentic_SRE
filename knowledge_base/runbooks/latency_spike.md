# Runbook: Latency Spike Response

## Overview
This runbook covers diagnosis and remediation of sudden p99 latency spikes on individual services.

## Symptoms
- p99 latency > 500ms on a single service
- Upstream services appear healthy
- Error rate may be slightly elevated (timeouts)

## Diagnosis Steps
1. Run `diagnostic_query` on the affected service with metric=`all`
2. Check `p99_latency_ms` — if > 5x baseline, it's a spike not drift
3. Run `log_inspection` to look for: timeout errors, slow consumer warnings, upstream connection refused
4. Confirm no cascading effects on downstream services

## Root Causes
- Memory leak causing GC pressure → high latency
- Upstream dependency timeout (e.g., auth calling a slow external service)
- Thread pool exhaustion under load

## Remediation
- If isolated to one service with no downstream impact: `restart_service`
- If saturation is high: `scale_up` first (factor=2.0), then monitor
- Do NOT rollback unless you have evidence of a bad deployment

## Verification
- Wait for 2 diagnostic cycles after remediation
- Confirm p99 latency returns to ≤ baseline × 1.2
- Confirm error rate ≤ 1%
