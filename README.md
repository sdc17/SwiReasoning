
<div align="center">
<h1>SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs</h1>
</div>

<p align="center">
    <a href="https://arxiv.org/pdf/2510.05069">
        <img alt="ArXiv" src="https://img.shields.io/badge/arXiv-2510.05069-B31B1B?logo=arxiv" />
    </a>
    <a href="https://swireasoning.github.io/">
        <img alt="Website" src="https://img.shields.io/badge/website-555555?logo=googlechrome" />
    </a><br>
</p>


## 👀 TL;DR
SwiReasoning is a *training-free* method for Pareto-superior reasoning LLMs that dynamically switches between explicit and latent thinking, with a switch count control mechanism to suppress overthinking.

![swir](assets/swir.png)

https://github.com/user-attachments/assets/6b18911c-efe4-47fd-8a00-3cd9ae1eb010

Comparison of solving the same question with the same reasoning LLM (6s vs. 1min).

## 🔍 Supported Benchmarks

* Math: GSM8K, MATH500, AIME24, AIME25
* Coding: HumanEval, LeetCode-Contest, MBPP, LiveCodeBench
* General: GPQA Diamond, 2WikiMultihopQA, CommonsenseQA

## 🔍 Supported Models

* Qwen3 and DeepSeek-R1 model families across 1.7B to 32B

## ⚙️ Getting Started

### Clone the project
``` bash
git clone https://github.com/sdc17/SwiReasoning.git
cd SwiReasoning
```

### Environment setup
```bash
conda create -n swir python=3.12
conda activate swir
pip install -r requirements.txt
```

## 💻 Interactive Chat

```bash
python run_chat.py --model_name Qwen/Qwen3-8B --method swir --max_switch_count 2
```

* Modify `--model_name` to try different reasoning LLMs.
* Increase `--max_switch_count` to allow more thinking rounds (default: 2).

```bash
Commands:
  exit or q -> [Exit]
  switch <N|none> -> [Set] swir max_switch_count = N (integer >= 1) or None (disabled)
  method <swir|cot|cot_greedy> -> [Set] generation method
```
* Please check [run_chat.sh](./run_chat.sh) for more examples.

## 📈 Evaluation

```bash
# Evaluate without switch count control
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-1.7B \
    --dataset_name gsm8k --batch_size 512 --max_new_tokens 32768 --method swir --alpha 0.6
python merge.py --model_name Qwen/Qwen3-1.7B --dataset_name gsm8k --max_new_tokens 32768 --method swir

# Evaluate with switch count control
torchrun --nproc_per_node 1 --nnodes 1 --node_rank 0 --master_port $((RANDOM + 20000)) run.py --model_name Qwen/Qwen3-8B \
    --dataset_name gsm8k --batch_size 256 --max_new_tokens 32768 --method swir --alpha 0.5 --max_switch_count 2
python merge.py --model_name Qwen/Qwen3-8B --dataset_name gsm8k --max_new_tokens 32768 --method swir

```
* Increase ``--nproc_per_node`` to enable faster evaluation on multiple GPUs. 
* Modify ``--model_name`` and ``--dataset_name`` for evaluation with different models and datasets.
* Please use ``TOKENIZERS_PARALLELISM=false`` before ``torchrun`` when evaluating on LiveCodeBench.
* Please check [run.sh](./run.sh) for more examples.

## 💬 Acknowledgments

We thank the contributors of open-source projects [Transformers](https://github.com/huggingface/transformers), [Qwen3](https://github.com/QwenLM/Qwen3), and [Soft-Thinking](https://github.com/eric-ai-lab/Soft-Thinking).

## ✨ BibTeX

Please cite if you find our codebase helpful.

```bash
@misc{shi2025swireasoningswitchthinkinglatentexplicit,
      title={SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs}, 
      author={Dachuan Shi and Abedelkadir Asi and Keying Li and Xiangchi Yuan and Leyan Pan and Wenke Lee and Wen Xiao},
      year={2025},
      eprint={2510.05069},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2510.05069}, 
}
```




