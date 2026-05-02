#!/usr/bin/env bash
# test_duffel.sh — Search → Hold → Cancel → Saga log check
#
# Usage:
#   ./test_duffel.sh
#   API_BASE=http://localhost:8000 ./test_duffel.sh

set -euo pipefail

BASE="${API_BASE:-http://localhost:8000}"
DEPART_DATE=$(date -d "+30 days" +%Y-%m-%d 2>/dev/null || date -v+30d +%Y-%m-%d)

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; NC='\033[0m'
hr() { printf '%.0s─' {1..60}; echo; }

# ── 0. Health ──────────────────────────────────────────────────────────────────
hr; echo -e "${CYAN}[0] Health check${NC}"
STATUS=$(curl -sf "$BASE/healthz" | jq -r '.status')
[[ "$STATUS" == "ok" ]] || { echo -e "${RED}Backend not ready${NC}"; exit 1; }
echo -e "${GREEN}OK${NC}"

# ── 1. Search — get offer_id + passenger_id ────────────────────────────────────
hr; echo -e "${CYAN}[1] POST /duffel/search  TPE→NRT $DEPART_DATE${NC}"
SEARCH=$(curl -sf -X POST "$BASE/duffel/search" \
  -H "Content-Type: application/json" \
  -d "{\"origin\":\"TPE\",\"destination\":\"NRT\",\"date\":\"$DEPART_DATE\"}")
echo "$SEARCH" | jq .

OFFER_ID=$(echo "$SEARCH"     | jq -r '.offer_id')
PASSENGER_ID=$(echo "$SEARCH" | jq -r '.passenger_id')
echo -e "\n  offer_id:     $OFFER_ID"
echo -e "  passenger_id: $PASSENGER_ID"

# ── 2. Hold ────────────────────────────────────────────────────────────────────
hr; echo -e "${CYAN}[2] POST /duffel/hold${NC}"
HOLD=$(curl -sf -X POST "$BASE/duffel/hold" \
  -H "Content-Type: application/json" \
  -d "{\"offer_id\":\"$OFFER_ID\",\"passenger_id\":\"$PASSENGER_ID\"}")
echo "$HOLD" | jq .

ORDER_ID=$(echo "$HOLD" | jq -r '.order_id')
echo -e "\n  order_id: $ORDER_ID"

# ── 3. Cancel without paying ───────────────────────────────────────────────────
hr; echo -e "${CYAN}[3] DELETE /duffel/orders/$ORDER_ID  (no payment)${NC}"
CANCEL=$(curl -sf -X DELETE "$BASE/duffel/orders/$ORDER_ID")
echo "$CANCEL" | jq .

CANCEL_STATUS=$(echo "$CANCEL" | jq -r '.cancel_status')
[[ "$CANCEL_STATUS" == "204" ]] \
  && echo -e "${GREEN}Duffel returned 204${NC}" \
  || echo -e "${RED}Unexpected: $CANCEL_STATUS${NC}"

# ── 4. Saga compensation log ───────────────────────────────────────────────────
hr; echo -e "${CYAN}[4] GET /saga/log/$ORDER_ID${NC}"
LOG=$(curl -sf "$BASE/saga/log/$ORDER_ID")
echo "$LOG" | jq '{
  saga_id: .saga_id,
  steps: [.steps[] | {
    name,
    status,
    order_id:      .receipt.order_id,
    idem_key:      .receipt.idem_key,
    held_at:       .receipt.held_at,
    cancel_status: .receipt.cancel_status,
    cancelled_at:  .receipt.cancelled_at
  }]
}'

# ── 5. Assert ──────────────────────────────────────────────────────────────────
hr; echo -e "${CYAN}[5] Assert saga log records cancel_status=204${NC}"
STEP_NAME="duffel_hold_${ORDER_ID:0:8}"

LOGGED_204=$(echo "$LOG"    | jq -r --arg n "$STEP_NAME" '.steps[] | select(.name==$n) | .receipt.cancel_status')
LOGGED_STATUS=$(echo "$LOG" | jq -r --arg n "$STEP_NAME" '.steps[] | select(.name==$n) | .status')

if [[ "$LOGGED_204" == "204" && "$LOGGED_STATUS" == "compensated" ]]; then
  echo -e "${GREEN}✓ cancel_status=204 and status=compensated — Saga log correct${NC}"
else
  echo -e "${RED}✗ cancel_status=$LOGGED_204  status=$LOGGED_STATUS${NC}"
  exit 1
fi

hr; echo -e "${GREEN}All steps passed.${NC}"
