1. Symlink for huggingface models in RAM
```
ls -la /mnt/ && sudo mkdir -p /mnt/huggingface_cache && sudo chown azureuser:azureuser /mnt/huggingface_cache
ls -la /mnt/huggingface_cache
ls -la /home/azureuser/.cache/huggingface/hub/
```

2. Check GPU memory
```
nvidia-smi --query-gpu=index,memory.used,memory.total,memory.free --format=csv
```

3. Check processes on the GPU & Kill then
```
nvidia-smi --query-compute-apps=pid,process_name,used_memory,gpu_uuid --format=csv
```
kill the pid number

4. Dataset capacity
```
cd /home/azureuser/localfiles/cortex-project && echo "=== Dataset Files ===" && wc -l datasets/finetuning/*.jsonl 2>/dev/null && echo "" && echo "=== Results Files ===" && wc -l results/*.jsonl 2>/dev/null
```
```
wc -l /home/azureuser/localfiles/cortex-project/datasets/finetuning/model1_tool_selection_missing_gpu*.jsonl 2>/dev/null
```