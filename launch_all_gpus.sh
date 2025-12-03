#!/bin/bash
# Launch TxAgent inference on multiple GPUs using tmux
# Each GPU runs in its own tmux session

cd /home/azureuser/localfiles/cortex-project

# Kill existing sessions if any
for gpu in 0 2 3 4 5 6 7; do
    tmux kill-session -t txagent_gpu${gpu} 2>/dev/null
done

echo "Starting TxAgent on 7 GPUs..."

# Launch each GPU in a separate tmux session
for gpu in 0 2 3 4 5 6 7; do
    echo "Starting GPU ${gpu}..."
    tmux new-session -d -s txagent_gpu${gpu}
    tmux send-keys -t txagent_gpu${gpu} "cd /home/azureuser/localfiles/cortex-project && conda activate azureml_py310_sdkv2 && python run_txagent_gpu${gpu}.py 2>&1 | tee logs/inference_gpu${gpu}.log" Enter
    sleep 2
done

echo ""
echo "✅ All GPU sessions started!"
echo ""
echo "To check status:"
echo "  tmux list-sessions"
echo ""
echo "To attach to a session:"
echo "  tmux attach -t txagent_gpu0"
echo ""
echo "To check progress:"
echo "  wc -l datasets/finetuning/model1_tool_selection_gpu*.jsonl"
echo ""
echo "To monitor all logs:"
echo "  tail -f logs/inference_gpu*.log"
