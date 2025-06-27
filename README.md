# ForkLift

Lift your understanding of any repo! 📦⬆ 

## TODO 

- build a basic pipeline driven from the CLI that allows 
  - synthesizing datasets (code directory -> dataset), 
  - training (dataset -> model), 
  - testing (dataset->scores), 
  - deploying (model->URL out) 
- explore transformers fine-tuning support
- understand whether RL can use the same dataset if we use something like BLEU to compare results... 
- find a model we can reasonably iterate on with our available compute (work rig)
- generate a dataset through iterative decomposition of some thing, if not a codebase
- 

## Problem 

It is common for software engineers and information technology practitioners to encounter unfamiliar codebases. These individuals may be asked to troubleshoot, refactor, or reason about the underlying capabilities or limitations. Approaching these tasks without a working knowledge of the codebase risks inducing unintended behavior or incorrectly assessing the scope of work. However, developing a working knowledge of a large codebase is a daunting task. Seasoned software engineers can take years to master a large software project. 

It is desirable to augment a human practitioner in all of these tasks to mitigate the aforementioned risks and accelerate the acquisition of working knowledge. Large language models (LLMs) have displayed an exceptional ability to summarize, annotate, generalize and reason about source code. A suitably proficient model can serve as a human 'copilot', supporting analysis, summarization and general knowledge reinforcement. However, models are limited by the data and latent patterns provided during training. Codebases that were only superficially represented or absent entirely during the training phase present a problem when recruited into this copilot role. 
- Model generates convincing language, and confidently recommends incorrect approaches based on the lack of underlying knowledge to support analysis 
- Models are extremely prone to hallucinating symbols, programming interfaces and patterns in codebases that didn't present a strong signal during training
- "prompt stuffing" or more sophisticated few-shot learning techqniues (e.g. retrieval-augmented generation - RAG), can improve results, but typically manifests as shallow knowledge
- Most codebases are constantly evolving, models often cpature knowlege of a software project at a moment in time, and can confidently recommend deprecated functionality while lacking an appreciation for critical changes introduced since training occurred

Depending on the nature of the code and it's complexity, and the degree to which it was available during training, the above issues can result in a significant gap betwween the promise of language models and their ultimate value in this context. Superfised fine tuning (SFT) and reinforcement learning (RL) methods provide an answer to this performance gap, however implementing these methods to improve LLM performance on codebase comprehension has not to the author's knowledge been advertised. A solution for improving LLM performance on specific codebases is proposed here accordingly. 

## Data Sources

## Prior Efforts 

### Codebase-specific Models

LLMs have demonstrated remarkable general performance in natural language and code generation, and they are frequently recruited into a developer or IT-practitioner copilot role to deploy that ability in support of software engineering or code comprehension. Prominent examples include Github Copilot, Windsurf, Cursor and Tabnine. While many of these support fine-tuned models, none explicitly provide support for codebase-specific models. 

A number of prior efforts exist to improve performance on specific codebases, but these exist only in the context of code-completion and don't support reasoning or comprehension tasks. For the reasons outlined below (SFT vs RL), it is expected that these methods actually *decrease* the associated model's ability to generalize markedly. All methods identified here are use code itself as the input to the training operation. That is, the dataset generation is naive, feeding code directly to the model during training. Whether completing code snippets or filling in the middle (FIM), no auxilliary mechanisms for generating code insights were included in model training. 


| Name | Dataset Strategy | Model | Approach | Notes  |
|-|-|-|-|-|
| Together.ai | Code completion | Mistral 7B instruct | SFT + RAG | [Article](https://www.together.ai/blog/rag-fine-tuning?utm_source=chatgpt.com) | 
| finetune-code-assistant | FIM | Qwen2 | SFT  | Svelte only, [Github](https://github.com/prvnsmpth/finetune-code-assistant/)|
| WandB.ai | FIM | Codellama series | SFT/LoRA | Code completion, [Article](https://wandb.ai/capecape/vllm_llm/reports/Finetunning-an-open-source-model-on-your-own-data-Part-1--Vmlldzo1NDQ1ODcw?utm_source=chatgpt.com) |
| Fine Tune Codebase | Code completion | Various | SFT/LoRA | [Github](https://github.com/ayminovitch/fine-tune-codebase)
| M. Khalusova Article | Code completion, FIM | bigcode/starcodebase-1b | SFT/PEFT | [HuggingFace Notebook](https://huggingface.co/learn/cookbook/en/fine_tuning_code_llm_on_single_gpu) | 

### Language Models and Transfer Learning

Transfer learning on language models has become status quo for those looking to imbue these models with domain-specific awareness or condition their behavior on a custom dataset [2]. Two primary techniques are widely used: 
- Superfised fine tuning (SFT) : Use of domain-specific data in a training loop to revise model weights to align with the provided data and an associated loss function 
- Reinforcement learning (RL) : 

As outlined in [3], SFT and RL have different outcomes on model behavior. The former drives rote memorization and decreases general performance, while the latter struggles to learn facts but supports generalization. In tandem, the two are a powerful combination to alter a language model's encoded knowledge *and* improve it's ability to generalize to tasks that require application of that knowledge. 

## Model Evaluation and Selection 

Contemporary language models based on neural networks (NN) display  emergent natural language fluency and reasoning abilities that dramatically outstrip their naive and classical machine-learning counterparts. However, to establish a baseline, and hedge against total model collapse during fine-tuning, results are also reported for 

- Hidden Markov Model 

### SFT Suitability 

In testing, SFT on Meta's OPT-350, a 350-million parameter language model, for 3 epochs of 25,000 training examples each had the following characteristics: 
- 100% GPU utilization throughout 
-  13GB VRAM usage thoughout
- 2 CPU cores partially utilized
- 13 minutes per epoch for a total run-time of 40 minutes

### RL Suitability 

A critical 

In [4], a 14-billion parameter code-specific models is the target of a reinforcement learning campaign supported by 24K high-quality coding challenge examples. The reward signal came from associated unit-tests that validated problem solutions. 

### Data Processing Pipeline 


**Repository Decomposition** 

1. Ctags : Generates an index of symbols for the target repository
   - Example invocation: `ctags -R --output-format=json --fields=+nksSaf --extras=+q -o linux_kernel.ctags ../../../linux/kernel/` will extract all symbols (even anonymous or implicit ones), enrich with some supplemental information and dump to disk at the file indicated by `-o`. 

**Superfised Fine-Tuning (SFT)**
1. Recursively decompose our target codebase to appreciate various *facets* that we can generate prompt pairs for by recruiting a foundation model (here GPT4.1 mini)
2. Apply a parameter-efficient fine-tuning (PEFT) strategy to limit the amount of parameters we need to learn/update when executing the transfer learning operation 
3. Train the model as usual, providing samples to support a forward pass and propagating the gradient of backwards to adjust model parameters for some number of epochs

**Reinforcement Learning (RL)** 
1. Rollout : pre-trained model is used to generate completions
2. Evaluation : we assess the value of the completion against our reference, yielding a scalar value to drive reinforcement learning 
3. Optimiztion : 

### Models

### 

## Repository Layout


The repository is laid out as follows

```
./
 ├── demo
 ├── models
 └── notebooks
```

- `demo` : Gradio demo application 
- `models` : Base and tuned models 
- `notebooks` : Various notebooks for experimentation and prototyping
  
## Quickstart 

All testing done with Python 3.12

1. `pip install -r requirements.txt` 

## Usage 

`forklift 
`--build`
`--train`
`--test`
`--deploy`

## Demo Application

The [demo](./demo) directory contains a Gradio app compatible with HuggingFace Spaces. This repository is manually mirrored to a HuggingFace repository which auto-deploys any changes to the hosted virtual environment. The resulting application can be found here: https://huggingface.co/spaces/3emaphor/forklift

## Results and Conclusions


## Ethics Statement

## References

1. Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU, https://huggingface.co/blog/trl-peft
2. Patil, R.; Gudivada, V. A Review of Current Trends, Techniques, and Challenges in Large Language Models (LLMs). Appl. Sci. 2024, 14, 2074. https://doi.org/10.3390/app14052074
3. Chu et al, "SFT Memorizes, RL Generalizes: A comparitive Study of Foundation Model Post-training", 42nd ICML, 2025 https://arxiv.org/pdf/2501.17161
4. Together.ai, DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level, https://www.together.ai/blog/deepcoder?utm_source=chatgpt.com
