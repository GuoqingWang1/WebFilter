export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_DEBUG=WARN
export CUDA_VISIBLE_DEVICES= #Your Visible Device


python -m vllm.entrypoints.openai.api_server --enable-reasoning --reasoning-parser deepseek_r1  --served-model-name auto --model /root/models/Qwen3-30B-A3B   \
 --tensor-parallel-size 16 --max-model-len 38192 --trust-remote-code   --port 8000 --gpu_memory_utilization 0.90