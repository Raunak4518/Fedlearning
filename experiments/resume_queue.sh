#!/usr/bin/env bash
# Resume the local GPU queue. Finished runs are read from each experiment's runs.jsonl
# and skipped, so this continues exactly where it stopped. Progress: results/logs/queue.log
cd "$(dirname "$0")"
export CUDA_MODULE_LOADING=LAZY PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
DR="D:/federated learing/gefl_classbalanced_production/gefl_classbalanced/data"
for job in "E29_hybrid_generator.py" "E26_private_anchoring.py" "E27_final_iid_many_clients.py" "F04_clients.py"; do
  echo "$(date +%H:%M) START $job" >> results/logs/queue.log
  python $job --data_root "$DR" --out_dir results >> "results/logs/$(echo $job | cut -d_ -f1).log" 2>&1
  echo "$(date +%H:%M) END $job exit=$?" >> results/logs/queue.log
done
echo QUEUE-RESUMED-DONE >> results/logs/queue.log
