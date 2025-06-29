# ForkLift

Lift your understanding of any repo! 📦⬆

## TODO 

- write an iterative decomposition and dataset building tool 
      - tree-sitter and AST integration point? 
    - alternative or in addition to: build a proper graph to drive DPO or validation? 
    - change the ingest to record path as if we're in the root of the linux kernel repo ... at present the paths are misleading
- explore transformers fine-tuning support
- find tensorboard traces asociated with model training
  - ensure we can see both training and validation loss
- decide on evaluation strategy ... 
  - how is this not just 1) build a dataset 2) hold out some data 3) use the same scalar we use to power the RL function ... i.e. evaluation is just a measure of similarity between desired output and actual ... 
    - BLEU is a problem, way too sensitive to evaluate the distanace between expected and provided output
    - Can we just use an embedding distance or angle here? perhaps cosine similarity to give us a better idea of how far off we are? -- we'll need to add embeddings to the dataset... compute at evaluation time or during generation? i
- use a PEFT fine-tuning operation and see how things change WRT training time, VRAM, etc. 
- figure out if we can accelerate the training with flash attention or flash attention 2 (see https://huggingface.co/docs/trl/sft_trainer#using-flash-attention-2)
- identify a suitable dataset type and schema
  - the SFTtrainer examples uses the stanfordnlp/imdb dataset, which is 25K rows of prompt/sentiment pairs
- use a canned DPO operation to understand what we're up for in terms of time, RAM, etc... 

## Problem 

It is common for software engineers and information technology practitioners to encounter unfamiliar codebases. These individuals may be asked to troubleshoot, refactor, or reason about the underlying capabilities or limitations. Approaching these tasks without a working knowledge of the codebase risks inducing unintended behavior or incorrectly assessing the scope of work. However, developing a working knowledge of a large codebase is a daunting task. Seasoned software engineers can take years to master a large software project. 

It is desirable to augment a human practitioner in all of these tasks to mitigate the aforementioned risks and accelerate the acquisition of working knowledge. Large language models (LLMs) have displayed an exceptional ability to summarize, annotate, generalize and reason about source code. A suitably proficient model can serve as a human 'copilot', supporting analysis, summarization and general knowledge reinforcement. However, models are limited by the data and latent patterns provided during training. Codebases that were only superficially represented or absent entirely during the training phase present a problem when recruited into this copilot role. 
- Model generates convincing language, and confidently recommends incorrect approaches based on the lack of underlying knowledge to support analysis 
- Models are extremely prone to hallucinating symbols, programming interfaces and patterns in codebases that didn't present a strong signal during training
- "prompt stuffing" or more sophisticated few-shot learning techqniues (e.g. retrieval-augmented generation - RAG), can improve results, but typically manifests as shallow knowledge
- Most codebases are constantly evolving, models often cpature knowlege of a software project at a moment in time, and can confidently recommend deprecated functionality while lacking an appreciation for critical changes introduced since training occurred

Example: Running up against the edge of knowledge and inducing hallucinations
```
ollama run qwen:0.5b
>>> What do you know about the Linux kernel? 
The Linux kernel is a high-level, open-source operating system. It has been designed to be easy to use and maintain. The Linux kernel is used by many different applications and programs are included in the kernel for various purposes.

>>> What is the 'init' subsystem for?          
The `init` subsystem for the Linux kernel is responsible for creating and initializing all of the system components such as hard drives, memory cards, network interfaces, etc.In addition to creating and initializing system components, the `init` subsystem also manages the state of the system and ensures that 
the correct version of the system is installed on all systems in the system.

>>> What does the `mount_root_generic` function do? 
The `mount__root_ generic` function creates a mount point for a root directory. This mount point can be used by multiple processes or applications that access and process data stored on the root directory. In addition to creating a mount point, this function also manages the state of the mount point, including creating and deleting directories and files that are mounted on the mount point. Overall, the `mount__root_ generic` function is a useful tool for creating and managing mount points on a Linux system.

>>> What file is it defined in? 
The `init` subsystem for the Linux kernel defines a specific file called "init.h". This file contains information about how to create and manage mount points on a Linux system. In addition to this specific file, the `init` subsystem also defines several other files that are used throughout the Linux kernel. These files include configuration files, such as `/etc/lsb.conf` for the Lubuntu distribution, and `/etc/products.conf` for the Ford product.
```

In the above example, we can be charitable about the limitations of a 500-billion parameter model yet want to improve on domain-specific Q&A that a model of this size could support. 

Depending on the nature of the code and it's complexity, and the degree to which it was available during training, the above issues can result in a significant gap betwween the promise of language models and their ultimate value in this context. Superfised fine tuning (SFT) and reinforcement learning (RL) methods provide an answer to this performance gap, however implementing these methods to improve LLM performance on codebase comprehension has not to the author's knowledge been advertised. A solution for improving LLM performance on specific codebases is proposed here accordingly. 

## Data Sources

- The Linux kernel [5]
- Latent knowledge and patterns as sourced during dataset synthesis from: 
  - ChatGPT 4.1 Nano [6]
  - Qwen 2.5 0.5B

  
###

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
2. Cscope: Extracts rich symbol reference information
   - a
   - b
3. Apply heuristics to generat
   - Apply a series of heuristics to generate prompts and reference information to pass to an LLM for completion
   - E.g. 
     - prompt: 'What file is BUG defined in?'
    - quality answer 1 (for SFT): 'The symbol `BUG` is defined in the `../../linux/init/Kconfig` file at line 1670.'
    - quality answer 2 (preferred DPO answer)): 'BUG is defined in `../../linux/init/Kconfig`.'
    - unpreferred answer (for DPO): 'The `BUG()` macro is defined in `<linux/bug.h>`.'
    - reframing of the question to induce variance for SFT: 'x' ='Where is the `BUG` symbol defined in the Linux kernel configuration?'
    - reframing of the qutestiont to induce variance for DPO: 'x2' ='Where is the `BUG` symbol defined in the Linux kernel source code?'

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
`--deploy` : requires HF_TOKEN environment variable to be set. This can be manually exported or alternatively,  is implicit after a `huggingface-cli login` completes. 

**Visualizing Training Loss** 

If tensorboard is installed (`pip install tensorboard`), we should be able to visualize SFT runs with the included web application. 
- ` tensorboard --logdir=runs` 
- visit http://localhost:6006
  
## Demo Application

The [demo](./demo) directory contains a Gradio app compatible with HuggingFace Spaces. This repository is manually mirrored to a HuggingFace repository which auto-deploys any changes to the hosted virtual environment. The resulting application can be found here: https://huggingface.co/spaces/3emaphor/forklift

## Results and Conclusions

### Challenges 

- Foundational model latency and potential throttling ... 
  - gpt-4.1-nano
    ```
    Sent 77 tokens to gpt-4.1-nano, received 61 tokens after 10.6s
    Sent 71 tokens to gpt-4.1-nano, received 13 tokens after 0.902s
    Sent 33 tokens to gpt-4.1-nano, received 13 tokens after 0.606s
    Sent 167 tokens to gpt-4.1-nano, received 18 tokens after 20.7s
    Sent 167 tokens to gpt-4.1-nano, received 18 tokens after 20.5s
    Sent 80 tokens to gpt-4.1-nano, received 101 tokens after 21.2s
    ```
  - gpt-4.1-mini
    ```
    Sent 77 tokens to gpt-4.1-mini, received 41 tokens after 1.15s
    Sent 71 tokens to gpt-4.1-mini, received 42 tokens after 0.936s
    Sent 33 tokens to gpt-4.1-mini, received 20 tokens after 0.554s
    Sent 147 tokens to gpt-4.1-mini, received 21 tokens after 20.9s
    Sent 147 tokens to gpt-4.1-mini, received 20 tokens after 20.9s
    Sent 80 tokens to gpt-4.1-mini, received 170 tokens after 23.4s
    Sent 74 tokens to gpt-4.1-mini, received 164 tokens after 23.6s
    Sent 32 tokens to gpt-4.1-mini, received 32 tokens after 21.0s
    Sent 280 tokens to gpt-4.1-mini, received 23 tokens after 20.9s
    Sent 280 tokens to gpt-4.1-mini, received 33 tokens after 20.9s
    Sent 51 tokens to gpt-4.1-mini, received 192 tokens after 24.5s
    Sent 45 tokens to gpt-4.1-mini, received 140 tokens after 22.7s
    Sent 33 tokens to gpt-4.1-mini, received 62 tokens after 1.46s
    Sent 272 tokens to gpt-4.1-mini, received 21 tokens after 21.1s
    Sent 272 tokens to gpt-4.1-mini, received 21 tokens after 20.9s
    Sent 50 tokens to gpt-4.1-mini, received 131 tokens after 22.6s
    Sent 44 tokens to gpt-4.1-mini, received 97 tokens after 21.6s
    Sent 32 tokens to gpt-4.1-mini, received 93 tokens after 21.6s
    ```
  

## Ethics Statement

### Provenance

**Data** 
This project was developed as a proof-of-concept to aid the rapid uptake and ultimate mastery of complex software projects. The data used to construct the synthetic training sets was sourced entirely from the Linux open source software project [5] which is licensed under GPL2, and other permissive licenses. The author is not aware of any underlying 

**Reproducability** 
The code written in this project is the author's work, made possible by a host of righteous open source software packages, tools, and the Ubuntu Linux distribution. Code snippets sourced from articles, tutorials and large-language model chat sessions are annotated in the source code where appropriate. All results here should be reproducible freely, without any licensing implications. 

**Harmful Content** 
The synthetic datasets generated are based purely on Linux kernel symbols and their associated references. Source code and comments are ingeseted into the question datasets used for training. A thorough review of this material has not been conducted, and latent bias, offensive content, or malicious code may have been unintentionally incorporated into the resulting dataset accordingly. 

## References

1. Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU, https://huggingface.co/blog/trl-peft
2. Patil, R.; Gudivada, V. A Review of Current Trends, Techniques, and Challenges in Large Language Models (LLMs). Appl. Sci. 2024, 14, 2074. https://doi.org/10.3390/app14052074
3. Chu et al, "SFT Memorizes, RL Generalizes: A comparitive Study of Foundation Model Post-training", 42nd ICML, 2025 https://arxiv.org/pdf/2501.17161
4. Together.ai, DeepCoder: A Fully Open-Source 14B Coder at O3-mini Level, https://www.together.ai/blog/deepcoder?utm_source=chatgpt.com
5. Torvalds, et al. (2025). Linux kernel (Version 6.16) [Computer software]. GitHub. https://github.com/torvalds/linux
6. OpenAI. (2024). ChatGPT (June 26 Version) [Large language model]. https://chat.openai.com/
