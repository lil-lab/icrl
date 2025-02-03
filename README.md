# LLMs Are In-Context Bandit Reinforcement Learners

This repository contains the code for the paper ["LLMs Are In-Context Bandit Reinforcement Learners"](https://arxiv.org/abs/2410.05362).

## Abstract
> Large Language Models (LLMs) excel at in-context learning (ICL), a supervised learning technique that relies on adding annotated examples to the model context. We investigate a contextual bandit version of in-context reinforcement learning (ICRL), where models learn in-context, online, from external reward, instead of supervised data. We show that LLMs effectively demonstrate such learning, and provide a detailed study of the phenomena, experimenting with challenging classification tasks and models of sizes from 500M to 70B parameters. This includes identifying and addressing the instability of the process, demonstrating learning with both semantic and abstract labels, and showing scaling trends. Our findings highlight ICRL capabilities in LLMs, while also underscoring fundamental limitations in their implicit reasoning about errors.

## Citation

If you find this work useful for your research, please consider citing:
```
@misc{monea2025llmsincontextbanditreinforcement,
      title={LLMs Are In-Context Bandit Reinforcement Learners}, 
      author={Giovanni Monea and Antoine Bosselut and Kianté Brantley and Yoav Artzi},
      year={2025},
      eprint={2410.05362},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05362}, 
}
```

## Setup

### Create and activate virtual environment

```bash
conda create --name icrl --file spec-file.txt
conda activate icrl
```

### Install dependencies

```bash
pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
```

### Setup Gemini API key

```bash
export API_KEY=YOUR_API_KEY
```

## Run experiments

The script used to run the ICRL experiments is `run_experiment.py`.

As an example, we provide the commands to run the experiments reported in the main results plot of the paper.

### Main ICRL experiments
To run the ICRL experiments reported in the main results plot of the paper for all tasks, use the following commands:


#### Naive ICRL experiments

<details>
  <summary>Banking77 task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name banking77 \
    --context_strategy_name random_biased_end \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>CLINC150 task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name clinic150 \
    --context_strategy_name random_biased_end \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Coarse task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_coarse \
    --context_strategy_name random_biased_end \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Fine task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_fine \
    --context_strategy_name random_biased_end \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>NLU task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name nlu \
    --context_strategy_name random_biased_end \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>


#### Naive+ ICRL experiments

<details>
  <summary>Banking77 task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name banking77 \
    --context_strategy_name random_biased_end_only_positive \
    --temperature 2.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>CLINC150 task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name clinic150 \
    --context_strategy_name random_biased_end_only_positive \
    --temperature 2.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Coarse task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_coarse \
    --context_strategy_name random_biased_end_only_positive \
    --temperature 2.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Fine task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_fine \
    --context_strategy_name random_biased_end_only_positive \
    --temperature 2.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>NLU task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name nlu \
    --context_strategy_name random_biased_end_only_positive \
    --temperature 2.0 \
    --context_p_keep 1.0 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>


#### Stochastic ICRL experiments

<details>
  <summary>Banking77 task</summary>

  ```bash
 python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name banking77 \
    --context_strategy_name random_unbiased_only_positive \
    --temperature 1.0 \
    --context_p_keep 0.1 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>CLINC150 task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name clinic150 \
    --context_strategy_name random_unbiased_only_positive \
    --temperature 1.0 \
    --context_p_keep 0.1 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Coarse task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_coarse \
    --context_strategy_name random_unbiased_only_positive \
    --temperature 1.0 \
    --context_p_keep 0.1 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>TREC Fine task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_fine \
    --context_strategy_name random_unbiased_only_positive \
    --temperature 1.0 \
    --context_p_keep 0.1 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>

<details>
  <summary>NLU task</summary>

  ```bash
python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name nlu \
    --context_strategy_name random_unbiased_only_positive \
    --temperature 1.0 \
    --context_p_keep 0.1 \
    --icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
```

</details>


#### Upper Bound / Supervised ICL experiments

<details>
  <summary>Banking77 task</summary>

  ```bash
  python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name banking77 \
    --context_strategy_name random_unbiased \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --no-icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
  ```

</details>

<details>
  <summary>CLINC150 task</summary>

  ```bash
  python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name clinic150 \
    --context_strategy_name random_unbiased \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --no-icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
  ```

</details>

<details>
  <summary>TREC Coarse task</summary>

  ```bash
  python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_coarse \
    --context_strategy_name random_unbiased \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --no-icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
  ```

</details>

<details>
  <summary>TREC Fine task</summary>

  ```bash
  python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name trec_fine \
    --context_strategy_name random_unbiased \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --no-icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 5000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
  ```

</details>

<details>
  <summary>NLU task</summary>

  ```bash
  python run_experiment.py \
    --model_name MODEL_NAME \
    --task_name nlu \
    --context_strategy_name random_unbiased \
    --temperature 1.0 \
    --context_p_keep 1.0 \
    --no-icrl \
    --no-icrl_omit_feedback \
    --no-icrl_flip_feedback \
    --train_k 10000 \
    --test_every 500 \
    --test_k 500 \
    --debug_k 10 \
    --seed 100 \
    --training_seed 100 \
    --test_seed 100 \
    --hf_token YOUR_HUGGINGFACE_TOKEN_HERE \
    --no-verbose
  ```

</details>



### Notes


Replace:

1. `YOUR_HUGGINGFACE_TOKEN_HERE` with your actual HuggingFace token before running these commands.
2. `MODEL_NAME` with the model name, as reported on HF (e.g., `meta-llama/Meta-Llama-3.1-8B-Instruct`, `Qwen/Qwen2.5-7B-Instruct`, `microsoft/Phi-3.5-mini-instruct`,  ...). For Gemini models, please refer to [this](https://ai.google.dev/gemini-api/docs/models/gemini) (as an example, to use Gemini 1.5 Flash 8B like in our experiments, the correct model name is `gemini-1.5-flash-8b`).


For abstract labels tasks, simply add `_unsemantic` after the original task name (e.g., `banking77_unsemantic`).


## Check maximum number of examples fitting in context window given an input window

We checked the max context examples given the max context window for Llama and the tasks Banking77 and CLINC150. We share the commands we used to output the max context examples below.

For 4096 tokens:

1. BANKING77
   Script:
   ```
   python find_max_examples_given_max_window.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --task_name banking77 --max_window_tokens 4096
   ```
   Output: 34 examples

2. CLINC150
   Script:
   ```
   python find_max_examples_given_max_window.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --task_name clinic150 --max_window_tokens 4096
   ```
   Output: 60 examples

For 8192 tokens:

1. BANKING77
   Script:
   ```
   python find_max_examples_given_max_window.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --task_name banking77 --max_window_tokens 8192
   ```
   Output: 74 examples

2. CLINC150
   Script:
   ```
   python find_max_examples_given_max_window.py --model_name meta-llama/Meta-Llama-3.1-8B-Instruct --task_name clinic150 --max_window_tokens 8192
   ```
   Output: 126 examples

## Count number of tokens used

```bash
python run_count_tokens.py
```

This outputs the number of tokens used for each experiment in the main results plot of the paper.

## Make plots

The following steps can be followed to generate the plots used in the paper:
1. Download the data from [here](https://cornell.box.com/s/mugw39qran8x9ezxz6dj1l24dxtytn77 ).
2. Extract the data folder.
3. Put the data folder in the root directory of the project.
4. Run the plots script:
```bash
python run_draw_plots.py
```

