"""
probe_tool_calling.py — §5B verification: confirm GLM-5.2's function-calling
response format before wiring the full four-tool reasoning loop.

Run this ONCE after setting up MODEL_* env vars:
  python probe_tool_calling.py

Exit code 0 = format OK, safe to run full episodes.
Exit code 1 = format mismatch — inspect output, fix adapter in mcp/tools.py.
"""
import json
import sys

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from agents.reasoning_loop import probe_tool_calling
from config import settings

print(f"\n{'='*60}")
print("GLM-5.2 Tool-Calling Format Probe")
print(f"{'='*60}")
print(f"  Model     : {settings.model_name}")
print(f"  Base URL  : {settings.model_base_url}")
print(f"  Thinking  : {settings.model_thinking_mode}")
print(f"{'='*60}\n")

result = probe_tool_calling()

print(f"Finish reason : {result.get('finish_reason')}")
print(f"Format OK     : {result['format_ok']}")
print(f"Tool calls    : {json.dumps(result['tool_calls_found'], indent=2)}")

if result.get("notes"):
    print(f"\nNotes:")
    for note in result["notes"]:
        print(f"  - {note}")

if result["format_ok"]:
    print("\n[OK] Format matches OpenAI function-calling spec.")
    print("     Safe to run full episodes with the four-tool set.\n")
    sys.exit(0)
else:
    print("\n[ERROR] Format mismatch detected.")
    print("        Inspect the output above. If tool_call schema differs,")
    print("        add an adapter layer in mcp/tools.py (not in reasoning_loop.py).\n")
    sys.exit(1)
