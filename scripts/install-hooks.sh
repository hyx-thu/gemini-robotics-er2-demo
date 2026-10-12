#!/usr/bin/env bash
# Install the repo's git hooks. Run once after `git init` / clone.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOOKS_DIR="$(git -C "$ROOT" rev-parse --git-path hooks)"
mkdir -p "$HOOKS_DIR"
cp "$ROOT/scripts/pre-commit" "$HOOKS_DIR/pre-commit"
chmod +x "$HOOKS_DIR/pre-commit"
echo "Installed pre-commit secret scanner → $HOOKS_DIR/pre-commit"
