
-----

# Dr. GRPO: Improving R1-Zero for Mathematical Reasoning

**AIST Final Project** | Exploring Dr. GRPO reasoning.

-----

## 🚀 Quick Start

### 1\. Installation

Recommended Environment: `python==3.10`

```bash
# Install dependencies
pip install vllm==0.8.4 oat-llm==0.1.3.post1

# Install project locally
git clone https://github.com/Quibbler6/AIST_final_proj.git && cd AIST_final_proj
pip install -e .
```

### 2\. Training with Dr. GRPO

```bash
export LD_LIBRARY_PATH=$(python -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))"):$LD_LIBRARY_PATH

python train_zero_math.py \
    --critic_type drgrpo \
    --gpus 8 \
    --enable_prefix_caching \
    --collocate \
    --vllm_sleep \
    --vllm_gpu_ratio 0.35 \
    --gradient-checkpointing \
    --flash-attn \
    --bf16 \
    --rnd-seed \
    --learning_rate 0.000001 \
    --lr_scheduler constant \
    --num_ppo_epochs 1 \
    --beta 0 \
    --oracle_type reward \
    --oracle math \
    --pretrain /data/lishiyu/model/Qwen/Qwen2.5-Math-1.5B \
    --prompt_template r1 \
    --zero-stage 2 \
    --ref_offload \
    --prompt_data ./datasets/train/math_12k \
    --train_split train \
    --input_key problem \
    --output_key answer \
    --max-train 9999999 \
    --num_prompt_epoch 10 \
    --prompt_max_length 1024 \
    --num_samples 8 \
    --temperature 1 \
    --top_p 1 \
    --generate_max_length 3000 \
    --save_steps -1 \
    --train_batch_size 128 \
    --train_batch_size_per_device 1 \
    --rollout_batch_size 128 \
    --rollout_batch_size_per_device 16 \
    --pi_buffer_maxlen_per_device 128 \
    --eval_batch_size 200 \
    --eval_steps 16 \
    --eval_temperature 0 \
    --eval_generate_max_length 3000 \
    --eval_data ./datasets/evaluation_suite \
    --eval_input_key input \
    --use-wb \
    --wb-run-name qwen2.5-Math-1.5b-r1-zero \
    --wb_project oat-zero
```

### 3\. Evaluation

```bash
# Evaluate trained model
python evaluate_model.py --model_name Qwen2.5-Math-1.5B-Oat-Zero

# Evaluate Baseline
python evaluate_model.py --model_name Qwen2.5-Math-1.5B
```

-----

## 📂 Project Structure

  * `train_zero_math.py`: Main training entry point using **Dr. GRPO**.
  * `evaluate_model.py`: Model performance evaluation.
  * `analysis/`: Log analysis and visualization.
  * `results_curve/`: Reward and loss curves.
  * `assets/`: Project figures and static resources.
  * `examples/`: Reference training scripts.

-----

## 📜 Acknowledgements

Special thanks to [Oat-LLM](https://www.google.com/search?q=https://github.com/sail-sg/oat-llm) and [vLLM](https://www.google.com/search?q=https://github.com/vllm-project/vllm) for the infrastructure support.
