
ablation_type="no_upstream"
model="DeepSeekV3"
dataset="DevEval"
nohup_log_path="LOGS/Ablation/frontground/${ablation_type}_${model}_${data}.log"



conda activate inlineCoder

nohup python inline_coder/Ablation/generation_deepseek_dev_eval.py --ablation_type $ablation_type --model $model --dataset $dataset > $nohup_log_path 2>&1 &

echo "Started with PID: $!"

echo "Logging to: $nohup_log_path"