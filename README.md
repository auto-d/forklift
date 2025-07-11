# ForkLift

Lift your understanding of any repo! 📦⬆

The dedicated HuggingFace Spaces hoosted at https://3emaphor-forklift.hf.space/ is struggling with online inference. Please use https://43422445f01f1ec4e1.gradio.live/ to view a demonatration.

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

Depending on the nature of the code and it's complexity, and the degree to which it was available during training, the above issues can result in a significant gap betwween the promise of language models and their ultimate value in this context. Supervised fine tuning (SFT) and reinforcement learning (RL) methods provide an answer to this performance gap, however implementing these methods to improve LLM performance on codebase comprehension has not to the author's knowledge been advertised. A solution for improving LLM performance on specific codebases is proposed here accordingly. 

## Data Sources

- The Linux kernel [5]
- Latent knowledge and patterns as sourced during dataset synthesis from: 
  - ChatGPT 4.1 Nano [6]
  - Qwen 2.5 0.5B

- Resulting patterns imparted are a mix of knowledge mined out of models and links between all symbols in 
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
- Reinforcement learning (RL) : Use of a proxy reward function usually trained on human feedback. Recent advances such as direct preference optimization (DPO) eschew the training of a reward model and allow preference pairs to be used to improve generalization. 

As outlined in [3], SFT and RL/DPO have different outcomes on model behavior. The former drives rote memorization and decreases general performance, while the latter struggles to learn facts but supports generalization. In tandem, the two are a powerful combination to alter a language model's encoded knowledge *and* imorove it's ability to generalize to tasks that require application of that knowledge. 

## Model Evaluation and Selection 

Contemporary language models based on neural networks (NN) display  emergent natural language fluency and reasoning abilities that dramatically outstrip their naive and classical machine-learning counterparts. However, to establish a baseline, and hedge against total model collapse during fine-tuning, results are also reported for a classical method and a naive method. 

### SFT Suitability 

In testing, SFT on Meta's OPT-350, a 350-million parameter language model, for 3 epochs of 25,000 training examples each had the following characteristics: 
- 100% GPU utilization throughout 
- 13GB VRAM usage thoughout
- 2 CPU cores partially utilized
- 13 minutes per epoch for a total run-time of 40 minutes

### RL Suitability 

A critical 

In [4], a 14-billion parameter code-specific models is the target of a reinforcement learning campaign supported by 24K high-quality coding challenge examples. The reward signal came from associated unit-tests that validated problem solutions. 

### Data Processing Pipeline 

The data processing pipeline consists of the following: 
- Linux kernel source code download
- Synthetic dataset generation based on recursive decomposition of the codebase to generate Q&A pairs as well as preference data for potential future refinement with DPO

**Superfised Fine-Tuning (SFT)**
1. Recursively decompose our target codebase to appreciate various *facets* that we can generate prompt pairs for by recruiting a foundation model (here GPT4.1 mini)
2. Apply a parameter-efficient fine-tuning (PEFT) strategy to limit the amount of parameters we need to learn/update when executing the transfer learning operation 
3. Train the model as usual, providing samples to support a forward pass and propagating the gradient of backwards to adjust model parameters for some number of epochs

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
  

## Usage 

```
usage: forklift train [-h] --dataset DATASET --model_dir MODEL_DIR [--nn_steps NN_STEPS] [--nn_epochs NN_EPOCHS] [--type {naive,classic,neural}]
```

The `forklift` script runs in four modes:
- `--build`
- `--train`
- `--test`
- `--deploy` 

Note the deployment omde requires HF_TOKEN environment variable to be set. This can be manually exported or alternatively,  is implicit after a `huggingface-cli login` completes. 

**Examples**
- `python main.py test --model models/naive --dataset data/ipc/ipc.parquet --type naive`
- `python main.py train --dataset data/test/init.parquet --model_dir models/classic --type classic`

### Building a Dataset

Below is an example run that emits a synthetic fine-tuning dataset for the `init` subsystem of the Linux kernel. Here we are running a local model and an openAI model in parallel to generate completions. 4K samples with sepearate completions for SFT and DPO took about 7 hours.

```
forklift build --inputs linux/init --dataset ./data/30june0116
Extracting symbols from linux/init..
Extracted 1087 definitions.
Gathering input files...
Preparing cscope indices... 
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1087/1087 [6:51:14<00:00, 22.70s/it]
Dataset generated (4332 rows)!
Model usage:
          model  in_tokens  out_tokens          time
0  gpt-4.1-mini    2140116      993414  24558.816467
1   llama3.1:8b    2282488     1078927  11763.533077
Wrote dataset for linux/init to ./data/30june0116/init.parquet.
Generation campaign complete!
```

After completing, statistics will be shown on model usage:

```
Model usage:
          model  in_tokens  out_tokens        time        tps
0  gpt-4.1-mini      11504        6668  150.910258  44.185200
1  gpt-4.1-nano      11297        5508  136.195672  40.441814
2  llama3.1:8b      13748        7557   99.836880  75.693471
```

## Demo Application

The [demo](./demo) directory contains a Gradio app compatible with HuggingFace Spaces. This repository is manually mirrored to a HuggingFace repository which auto-deploys any changes to the hosted virtual environment. 

## Results and Conclusions

### Naive 

The naive symbol-matching strategy produces horrific results. We ultimately failed to nominate many reasonable synmbols based on input text. 

### Classic - Hidden Markov Model

Ex 1: 
- Input: "Where is the macro `COMPAT_SHMLBA` defined in the Linux kernel source code?"
- Target output: "`COMPAT_SHMLBA` is defined in `linux/ipc/shm.c` at line 1706."
- HMM output: "summary headers associated ` trace/ ` kernel?`create_trace_points ` usually ` events ` specifically suggests locate test ` initialization defined registration registration tracepoint \n\n indirectly ` tracepoint_probe_register trace/ ` functions defined tracepoint registration \n ` apis ` source create_trace_points include create_trace_points \n\n defined code trace_event file registration ` ` ` <"

Ex 2: 
- Input: "'Where is the `ipc_init` function defined in the Linux kernel source, and what is its linkage scope?'"
- Target output: "'The symbol `ipc_init` is defined as a static function in `linux/ipc/util.c` at line 90, indicating internal linkage within that translation unit. Since it is declared `static`, it is not visible outside `util.c`, and there are no other definitions of `ipc_init` elsewhere in the kernel source. Its only definition is within `linux/ipc/util.c`.'"
- HMM output: "'tracepoint ` macro ` summary pattern \n linux tracepoint_register trace point ` source tracepoints.h tracepoint_register \n include  \n invoked code \n ` create_trace_points main.c macro defined key context ` registration ` ` tracepoints define_trace create_trace_points ` ` actual tracepoints ` shown invocation subsystem code ` expansion ` tracepoints.c create_trace_points `'"

### SLM-based Fine-Tune

The SLM performed better than expected, with the desired knowledge being imparted from various Linux subsystems. 

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
