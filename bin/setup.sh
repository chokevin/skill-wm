#!/usr/bin/env bash
# setup.sh — one-shot setup for skill-wm researchers running on Rune.
#
# This repo's local loop runs entirely on a laptop (Crafter rollouts are CPU
# only). Rune is used for parallel data collection and (later) WM training
# on the voice-agent-flex cluster.
#
# Steps:
#   1. Verify az + kubectl + python3.
#   2. az login if needed.
#   3. Pull voice-agent-flex kubeconfig (idempotent).
#   4. Verify the rune CLI is on PATH (or instruct how to install).
#   5. Create .venv via uv and install rune-py.
#   6. Set default namespace from $SKILL_WM_NS (default: ray).
#
# Cluster targeting (override per researcher):
#   SKILL_WM_CLUSTER_NAME  (default: voice-agent-flex)
#   SKILL_WM_CLUSTER_RG    (default: voice-agent-flex-rg)
#   SKILL_WM_SUBSCRIPTION  (optional; az account set when provided)
#   SKILL_WM_NS            (default: ray)
set -euo pipefail

CLUSTER_NAME="${SKILL_WM_CLUSTER_NAME:-voice-agent-flex}"
CLUSTER_RG="${SKILL_WM_CLUSTER_RG:-voice-agent-flex-rg}"
CLUSTER_SUB="${SKILL_WM_SUBSCRIPTION:-}"
NS="${SKILL_WM_NS:-ray}"

cd "$(dirname "$0")/.."

green() { printf "\033[32m%s\033[0m\n" "$*"; }
red()   { printf "\033[31m%s\033[0m\n" "$*" >&2; }
step()  { printf "\n\033[1m▸ %s\033[0m\n" "$*"; }

# ---- prerequisites --------------------------------------------------------

step "checking prerequisites"
for bin in az kubectl python3 uv; do
  if ! command -v "$bin" >/dev/null 2>&1; then
    red "missing: $bin"
    case "$bin" in
      az)      red "  install: brew install azure-cli" ;;
      kubectl) red "  install: brew install kubectl  (or: az aks install-cli)" ;;
      python3) red "  install: brew install python" ;;
      uv)      red "  install: brew install uv" ;;
    esac
    exit 1
  fi
done
green "prerequisites ok"

# ---- azure login ----------------------------------------------------------

step "checking Azure login"
if ! az account show >/dev/null 2>&1; then
  az login --use-device-code
fi
if [ -n "$CLUSTER_SUB" ]; then
  az account set --subscription "$CLUSTER_SUB"
fi
green "azure: $(az account show --query user.name -o tsv) (sub: $(az account show --query name -o tsv))"

# ---- kubeconfig -----------------------------------------------------------

step "fetching kubeconfig for $CLUSTER_NAME"
az aks get-credentials \
  --resource-group "$CLUSTER_RG" \
  --name "$CLUSTER_NAME" \
  --overwrite-existing >/dev/null
kubectl config use-context "$CLUSTER_NAME" >/dev/null
kubectl config set-context --current --namespace="$NS" >/dev/null
green "kubectl context: $(kubectl config current-context), namespace: $NS"

if ! kubectl auth can-i get jobs -n "$NS" >/dev/null 2>&1; then
  red "WARN: current identity cannot get jobs in namespace $NS."
  red "      Researchers usually need the researcher RBAC bundle from aks-ai-runtime."
fi

# ---- rune CLI -------------------------------------------------------------

step "checking rune CLI"
if command -v rune >/dev/null 2>&1; then
  green "rune CLI: $(rune --version 2>/dev/null || echo 'present')"
else
  red "rune CLI not found on PATH."
  red "  Build from azure-management-and-platforms/aks-ai-runtime:"
  red "    git clone git@github.com:azure-management-and-platforms/aks-ai-runtime.git"
  red "    cd aks-ai-runtime/applications/rune && make release-binaries"
  red "    cp bin/release/rune-\$(uname | tr A-Z a-z)-\$(uname -m | sed s/x86_64/amd64/) ~/.local/bin/rune"
  red "  Or copy from a sibling project that has it vendored, e.g. ~/dev/voice-agent/bin/rune/"
  exit 1
fi

# ---- python venv via uv ---------------------------------------------------

step "syncing Python deps via uv"
uv sync
green "uv sync done"

# ---- rune-py (optional install) ------------------------------------------

step "installing rune-py SDK"
# rune-py is kept out of the default project deps because it requires
# GitHub SSH access. Install it explicitly here so the laptop loop
# (rollouts + smoke tests) stays unblocked when SSH isn't configured yet.
#
# Override RUNE_PY_LOCAL=/path/to/rune-sdk to install from a local checkout
# (for example, ~/dev/voice-agent/rune-sdk where it's already vendored).
if [ -n "${RUNE_PY_LOCAL:-}" ]; then
  if [ ! -f "$RUNE_PY_LOCAL/pyproject.toml" ]; then
    red "RUNE_PY_LOCAL=$RUNE_PY_LOCAL is not a valid rune-sdk checkout (no pyproject.toml)"
    exit 1
  fi
  uv pip install --editable "$RUNE_PY_LOCAL"
  green "rune-py installed from $RUNE_PY_LOCAL (editable)"
elif uv pip install "rune-py @ git+ssh://git@github.com/azure-management-and-platforms/aks-ai-runtime.git@main#subdirectory=applications/rune-py" 2>/dev/null; then
  green "rune-py installed via git+ssh"
else
  red "rune-py install via git+ssh failed (likely missing GitHub SSH access)."
  red "  Workaround: clone aks-ai-runtime and re-run with"
  red "    RUNE_PY_LOCAL=/path/to/aks-ai-runtime/applications/rune-py bash bin/setup.sh"
  red "  (Voice-agent users: RUNE_PY_LOCAL=\$HOME/dev/voice-agent/rune-sdk works.)"
  red "  Continuing without rune-py — local make targets still work, but"
  red "  experiments/*/config.py imports will fail until rune-py is installed."
fi

green ""
green "Setup complete. Activate the venv and submit a job:"
green "  source .venv/bin/activate"
green "  RUNE_NAME=skill-wm-collect-001 python experiments/collect_rollouts/config.py"
