# Launch TxAgent inference for Tool Calling Generation on multiple GPUs using tmux

cd /home/azureuser/localfiles/cortex-project

for gpu in 0 2 3 4 5 6 7; do
    tmux kill-session -t txagent_gpu${gpu} 2>/dev/null
done

echo "Starting TxAgent on 7 GPUs..."

for gpu in 0 2 3 4 5 6 7; do
    echo "Starting GPU ${gpu}..."
    tmux new-session -d -s txagent_gpu${gpu}
    tmux send-keys -t txagent_gpu${gpu} "cd /home/azureuser/localfiles/cortex-project && conda activate azureml_py310_sdkv2 && python run_txagent_gpu${gpu}.py 2>&1 | tee logs/inference_gpu${gpu}.log" Enter
    sleep 2
done

echo ""
echo "All GPU sessions started!"
echo ""

