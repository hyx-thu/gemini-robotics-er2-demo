#!/usr/bin/env bash
# =====================================================================
# deploy.sh — Build & deploy the Gemini Robotics-ER demo to Cloud Run.
#
# The Gemini API key is NEVER passed on the command line or baked into
# the image. It lives in Secret Manager and is mounted into the service
# as the GEMINI_API_KEY / TTS_API_KEY environment variables at runtime.
#
# Usage:
#   ./deploy.sh                 # deploy using defaults below
#   PROJECT_ID=my-proj ./deploy.sh
#
# One-time setup (creates the secret from your local agent/.env):
#   ./deploy.sh --init-secret
# =====================================================================
set -euo pipefail

PROJECT_ID="${PROJECT_ID:-elated-terminal-503102-u0}"
REGION="${REGION:-us-central1}"
SERVICE="${SERVICE:-gemini-robotic-er2}"
SECRET_NAME="${SECRET_NAME:-gemini-api-key}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AGENT_DIR="${SCRIPT_DIR}/agent"

log() { printf '\033[1;34m[deploy]\033[0m %s\n' "$*"; }
die() { printf '\033[1;31m[deploy] ERROR:\033[0m %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------
# --init-secret: push the key from agent/.env into Secret Manager
# ---------------------------------------------------------------------
init_secret() {
  local env_file="${AGENT_DIR}/.env"
  [[ -f "$env_file" ]] || die "agent/.env not found. Copy agent/.env.example and fill in GEMINI_API_KEY."

  local key
  key="$(sed -n 's/^[[:space:]]*GEMINI_API_KEY[[:space:]]*=[[:space:]]*//p' "$env_file" | head -n1 | tr -d '"'"'"' \r')"
  [[ -n "$key" && "$key" != "your-gemini-api-key-here" ]] || die "GEMINI_API_KEY in agent/.env is empty or still the placeholder."

  log "Enabling Secret Manager API (idempotent)..."
  gcloud services enable secretmanager.googleapis.com --project="$PROJECT_ID" --quiet

  if gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" >/dev/null 2>&1; then
    log "Secret '$SECRET_NAME' exists — adding a new version."
    printf '%s' "$key" | gcloud secrets versions add "$SECRET_NAME" --project="$PROJECT_ID" --data-file=-
  else
    log "Creating secret '$SECRET_NAME'."
    printf '%s' "$key" | gcloud secrets create "$SECRET_NAME" --project="$PROJECT_ID" \
      --replication-policy=automatic --data-file=-
  fi

  # Grant the Cloud Run runtime service account read access.
  local project_number sa
  project_number="$(gcloud projects describe "$PROJECT_ID" --format='value(projectNumber)')"
  sa="${project_number}-compute@developer.gserviceaccount.com"
  log "Granting secretAccessor on '$SECRET_NAME' to $sa"
  gcloud secrets add-iam-policy-binding "$SECRET_NAME" --project="$PROJECT_ID" \
    --member="serviceAccount:${sa}" --role="roles/secretmanager.secretAccessor" --quiet >/dev/null
  log "Secret ready."
}

# ---------------------------------------------------------------------
# deploy: build from source and wire the secret in
# ---------------------------------------------------------------------
deploy() {
  gcloud secrets describe "$SECRET_NAME" --project="$PROJECT_ID" >/dev/null 2>&1 \
    || die "Secret '$SECRET_NAME' not found in $PROJECT_ID. Run: ./deploy.sh --init-secret"

  log "Deploying $SERVICE to $PROJECT_ID/$REGION from $AGENT_DIR"
  gcloud run deploy "$SERVICE" \
    --source="$AGENT_DIR" \
    --project="$PROJECT_ID" \
    --region="$REGION" \
    --allow-unauthenticated \
    --session-affinity \
    --timeout=3600 \
    --cpu=2 --memory=2Gi --max-instances=3 \
    --set-secrets="GEMINI_API_KEY=${SECRET_NAME}:latest,TTS_API_KEY=${SECRET_NAME}:latest" \
    --set-env-vars="ROBOT_URL=http://127.0.0.1:8888,TTS_PROVIDER=gemini,GEMINI_TTS_MODEL=gemini-3.8-flash-tts,SPOT_SIMULATED=1,DEFAULT_AGENT=spot" \
    --quiet

  local url
  url="$(gcloud run services describe "$SERVICE" --project="$PROJECT_ID" --region="$REGION" --format='value(status.url)')"
  log "Done → $url"
}

case "${1:-}" in
  --init-secret) init_secret ;;
  "")            deploy ;;
  *)             die "Unknown argument: $1 (use --init-secret or no args)" ;;
esac
