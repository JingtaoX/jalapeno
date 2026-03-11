#!/bin/bash
# Jalapeno compiler wrapper script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/venv/bin/activate"
# source "$./venv/bin/activate"
python "$SCRIPT_DIR/src/main.py" "$@"
