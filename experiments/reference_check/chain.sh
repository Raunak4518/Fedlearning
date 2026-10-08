#!/usr/bin/env bash
# Sequential validation queue (one GPU process at a time).
#   usage: bash chain.sh <ref_repo_copy> <data_root>
set -u
REF="$1"
DATA="$2"
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
export PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1

# wait for the run already in progress
while powershell -NoProfile -Command "if (Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | Where-Object { \$_.CommandLine -like '*run_reference*' }) { exit 0 } else { exit 1 }"; do sleep 30; done

for s in 0 1 2; do
  python run_reference.py --ref_dir "$REF" --setting lt --seed $s --avg_FE 1 > ref_lt_s${s}_fe1.log 2>&1
  echo "done lt seed $s fe1 $(date +%H:%M)"
done
python run_reference.py --ref_dir "$REF" --setting iid --seed 0 --avg_FE 0 > ref_iid_s0_fe0.log 2>&1
echo "done iid seed 0 fe0 $(date +%H:%M)"
python run_reference.py --ref_dir "$REF" --setting lt --seed 0 --avg_FE 0 > ref_lt_s0_fe0.log 2>&1
echo "done lt seed 0 fe0 $(date +%H:%M)"
cd ..
python F02_paper_setting.py --datasets mnist --only GeFL-F --data_root "$DATA" --out_dir results > results/logs/F02_gefl_mnist_check.log 2>&1
echo "done ours F02 GeFL-F mnist $(date +%H:%M)"
