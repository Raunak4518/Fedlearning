#!/usr/bin/env bash
# One-time setup of a local GPU machine for the FedECS-LT runs (Ubuntu, or Ubuntu inside WSL2 on Windows).
#   bash fedecs_lt/local/setup.sh
# Creates the Python environment ~/fedecs_env with fixed versions (fedecs_lt/local/requirements.txt) and PyTorch's CUDA
# build that matches the NVIDIA driver, then checks that the GPU is visible. Safe to run again.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV="${FEDECS_ENV:-$HOME/fedecs_env}"
TORCH="torch==2.14.1 torchvision==0.29.1"

say() { printf '\n== %s\n' "$*"; }
stop() { printf '\nSTOP: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s)" = "Linux" ] || stop "this script is for Linux; on Windows double-click fedecs_lt/local/setup_windows.bat (README.md)."
case "$HERE" in /mnt/[a-z]/*) echo "Note: the repository is on the Windows drive ($HERE); runs are faster if it is cloned inside WSL2 (~/Fedlearning).";; esac

say "NVIDIA driver"
SMI="$(command -v nvidia-smi || true)"
[ -z "$SMI" ] && [ -x /usr/lib/wsl/lib/nvidia-smi ] && SMI=/usr/lib/wsl/lib/nvidia-smi
[ -n "$SMI" ] || stop "nvidia-smi not found. Install the NVIDIA driver (on Windows: the normal Windows driver from nvidia.com; never a driver inside WSL2)."
"$SMI" --query-gpu=name,memory.total,driver_version --format=csv,noheader
DRV="$("$SMI" --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)"
if [ "$DRV" -ge 580 ]; then CU=cu130; elif [ "$DRV" -ge 560 ]; then CU=cu126; else
  stop "driver $DRV is too old for PyTorch 2.14 (needs 560 or newer). Update the NVIDIA driver, then run this again."; fi
echo "PyTorch build: $CU (driver $DRV)"

say "System packages (python3-venv, git, compiler)"
PY="$(command -v python3.12 || command -v python3.13 || command -v python3.11 || true)"
NEED=""
[ -n "$PY" ] || NEED="python3 python3-venv python3-dev"
[ -n "$PY" ] && ! "$PY" -m venv --help >/dev/null 2>&1 && NEED="$NEED python3-venv"
[ -n "$PY" ] && ! "$PY" -c "import ensurepip" >/dev/null 2>&1 && NEED="$NEED python3-venv"
command -v git >/dev/null || NEED="$NEED git"
command -v gcc >/dev/null || NEED="$NEED build-essential"
if [ -n "$NEED" ]; then
  echo "Installing:$NEED (asks for your Linux password)"
  sudo apt-get update -y && sudo apt-get install -y $NEED
  [ -n "$PY" ] || PY="$(command -v python3)"
  # Ubuntu names the venv package after the version (python3.12-venv)
  "$PY" -c "import ensurepip" >/dev/null 2>&1 || sudo apt-get install -y "python$("$PY" -c 'import sys;print(f"{sys.version_info[0]}.{sys.version_info[1]}")')-venv"
fi
"$PY" -c "import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), sys.version" \
  || stop "Python 3.11, 3.12 or 3.13 is needed (Flower 1.42); found $("$PY" --version)."

say "Python environment $ENV"
[ -x "$ENV/bin/python" ] || "$PY" -m venv "$ENV"
"$ENV/bin/python" -m pip install -q --upgrade pip wheel setuptools
"$ENV/bin/python" -m pip install -q $TORCH --index-url "https://download.pytorch.org/whl/$CU"
"$ENV/bin/python" -m pip install -q -r "$HERE/requirements.txt" -r "$HERE/requirements_longtaillib.txt"

say "Check"
"$ENV/bin/python" - <<'EOF'
import flwr, ray, torch, torchvision, sklearn, reportlab
print("Python", __import__("sys").version.split()[0], "| torch", torch.__version__, "| Flower", flwr.__version__,
      "| Ray", ray.__version__)
assert torch.cuda.is_available(), "PyTorch does not see the GPU (driver / WSL2 GPU support)"
p = torch.cuda.get_device_properties(0)
print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB")
x = torch.randn(512, 512, device="cuda"); print("CUDA works:", float((x @ x).sum()) == float((x @ x).sum()))
EOF
cat <<EOF

Setup finished. Next (fedecs_lt/local/README.md):
  source $ENV/bin/activate
  python fedecs_lt/local/run_local.py check      # once per machine: measures what fits on this GPU
  python fedecs_lt/local/run_local.py list       # the jobs, and which ones fit here
EOF
