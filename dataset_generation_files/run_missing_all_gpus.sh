source /anaconda/etc/profile.d/conda.sh
conda activate azureml_py310_sdkv2

echo "Starting missing samples processing at $(date)"
echo "================================================"

echo "Starting GPU 0..."
nohup python run_missing_samples.py --gpu 0 --start_offset 0 --batch_size 1172 > logs/missing_gpu0.log 2>&1 &
 
echo "Starting GPU 1..."
nohup python run_missing_samples.py --gpu 1 --start_offset 1172 --batch_size 1172 > logs/missing_gpu1.log 2>&1 &

echo "Starting GPU 3..."
nohup python run_missing_samples.py --gpu 3 --start_offset 2344 --batch_size 1172 > logs/missing_gpu3.log 2>&1 &

echo "Starting GPU 5..."
nohup python run_missing_samples.py --gpu 5 --start_offset 3516 --batch_size 1172 > logs/missing_gpu5.log 2>&1 &

echo "Starting GPU 7..."
nohup python run_missing_samples.py --gpu 7 --start_offset 4688 --batch_size 1171 > logs/missing_gpu7.log 2>&1 &

echo ""
echo "All processes started! Monitor with:"
echo "  tail -f logs/missing_gpu*.log"
echo "  wc -l datasets/finetuning/model1_tool_selection_missing_gpu*.jsonl"
