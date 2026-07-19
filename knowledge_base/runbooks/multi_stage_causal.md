# Runbook: Multi-Stage Causal Decomposition

## Overview
Some incidents involve a chain of consequences where fixing downstream symptoms doesn't resolve
the incident. This is the hardest class of incident. You must trace causality back to the root.

## Pattern Recognition
Causal chain incidents have this signature:
- Multiple services degraded simultaneously
- One service degraded FIRST (root cause) — others are consequences
- Fixing consequence services gives only temporary relief if root isn't fixed

## Diagnosis Protocol (Causal Tracing)
1. Observe ALL services first — never start remediating immediately
2. Build a mental causal model: which service's degradation would cause the others?
3. The root cause service is typically:
   - Earliest in the request path (auth → api-gateway → user-service → payment-service)
   - Shows the highest absolute degradation relative to its baseline
   - Its logs show self-caused errors (not upstream references)

## The Auth → Gateway → User-Service → Payment Chain
This specific chain has a clear causal order:
1. **auth** slow → gateway can't authenticate requests → queue builds
2. **api-gateway** saturated → user-service requests time out → errors
3. **user-service** erroring → payment-service overwhelmed → saturation

**FIX AUTH FIRST.** If auth is fixed, the rest of the chain clears automatically.

## Remediation Order
1. Fix auth: `restart_service` on auth
2. Wait and verify auth p99 drops
3. Verify api-gateway queue clears (saturation drops)
4. Verify user-service errors clear
5. If payment-service still saturated after auth fix: `scale_up` payment-service (NOT restart — blocked by gate)
6. Submit resolution only after ALL services are within targets

## Common Mistakes
- Starting with `restart_service` on payment-service → BLOCKED by Quarantine gate (high-risk)
- Remediating gateway before auth → cascade partially clears then recurs
- Calling `submit_resolution` too early before verifying all 4 services
