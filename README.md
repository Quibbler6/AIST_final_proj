
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
    --pretrain Qwen/Qwen2.5-Math-1.5B \
    --prompt_data ./datasets/train/math_12k \
    --wb-run-name qwen2.5-Math-1.5b-dr-grpo \
    --enable_prefix_caching \
    --vllm_gpu_ratio 0.35 \
    --learning_rate 0.000001 \
    --num_samples 8 \
    --generate_max_length 3000
```

*Check [examples/](https://www.google.com/search?q=./examples/) for more scripts.*

### 3\. Evaluation

```bash
# Evaluate our trained model
python evaluate_model.py --model_name sail/Qwen2.5-Math-1.5B-Oat-Zero

# Evaluate Baseline
python evaluate_model.py --model_name Qwen/Qwen2.5-Math-1.5B
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
