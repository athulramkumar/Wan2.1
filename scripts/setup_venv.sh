#!/bin/bash
# Setup virtual environment for Wan2.1 distributed inference.
#
# Usage:
#   ./scripts/setup_venv.sh coordinator   # For stable pod (1.3B model)
#   ./scripts/setup_venv.sh worker        # For spot pod (14B model)
#
# This creates .venv_coordinator/ or .venv_worker/ in the project root.
# Venvs persist on the shared RunPod volume between restarts.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
ROLE="${1:-coordinator}"

if [[ "$ROLE" != "coordinator" && "$ROLE" != "worker" ]]; then
    echo "Usage: $0 [coordinator|worker]"
    exit 1
fi

VENV_DIR="$PROJECT_ROOT/.venv_${ROLE}"

echo "========================================"
echo "  Setting up ${ROLE} environment"
echo "  Venv: ${VENV_DIR}"
echo "========================================"

# Create venv if it doesn't exist
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Upgrade pip
pip install --upgrade pip -q

# Install base requirements
echo "Installing base requirements..."
pip install -r "$PROJECT_ROOT/requirements.txt" -q

# Install API requirements
echo "Installing API requirements..."
pip install fastapi uvicorn requests pydantic -q

# Install test dependencies
pip install pytest -q

# Role-specific dependencies
if [ "$ROLE" = "worker" ]; then
    echo "Installing worker-specific dependencies (xfuser for multi-GPU)..."
    pip install xfuser -q 2>/dev/null || echo "Warning: xfuser install failed (may need specific CUDA version)"
fi

echo ""
echo "========================================"
echo "  ${ROLE} environment ready!"
echo "  Activate with: source ${VENV_DIR}/bin/activate"
echo "========================================"

# Print GPU info if available
python3 -c "
import torch
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        name = torch.cuda.get_device_name(i)
        mem = torch.cuda.get_device_properties(i).total_mem / 1e9
        print(f'  GPU {i}: {name} ({mem:.0f}GB)')
else:
    print('  No GPU detected')
" 2>/dev/null || true
