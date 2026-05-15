#!/usr/bin/env bash
# Walks the agent through one example of each path so a reviewer can see
# all flows in <30 seconds.
#
# Usage:
#   ./scripts/demo.sh                    # uses http://localhost:8000
#   API=http://api:8000 ./scripts/demo.sh
set -euo pipefail

API=${API:-http://localhost:8000}
PRETTY="cat"
if command -v jq >/dev/null 2>&1; then PRETTY="jq ."; fi

run() {
  local label="$1" endpoint="$2" body="$3"
  echo
  echo "============================================================"
  echo "  $label"
  echo "  POST ${API}${endpoint}"
  echo "============================================================"
  curl -sS -X POST "${API}${endpoint}" -H "Content-Type: application/json" -d "$body" | $PRETTY
}

echo "FatWad — demo run"
echo "API: $API"

run "1. Plain Q&A — /ask (no tools, no web)" /ask \
  '{"query":"What is my net worth right now?"}'

run "2. Forecast — /agent (tool: compute_compound)" /agent \
  '{"query":"In 15 years, how much will my net worth be if I keep my current surplus?"}'

run "3. Scenario — /agent (tool: simulate_scenario)" /agent \
  '{"query":"What if I add $1500/mo to a total-market index fund for 7 years?"}'

run "4. Advice — /agent (Python tool: buy_list, rebalance)" /agent \
  '{"query":"Where should I deploy this month'\''s surplus and is anything off-balance?"}'

run "5. Market lookup — /agent (web-grounded prefill)" /agent \
  '{"query":"What is a defensible long-term return assumption for US equities right now?"}'
