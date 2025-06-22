# ForkLift

Lift your understanding of any repo! 📦⬆ 

## TODO 

- explore transformers fine-tuning support
- find a model we can reasonably iterate on with our available compute (work rig)
- generate a dataset through iterative decomposition of some thing, if not a codebase

## Problem 

## Data Sources

## Prior Efforts 

## Model Evaluation and Selection 

### Data Processing Pipeline 

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

## Quickstart 

All testing done with Python 3.12

1. `pip install -r requirements.txt` 

## Demo Application

## Results and Conclusions

## Ethics Statement

## Referneces

1. Fine-tuning 20B LLMs with RLHF on a 24GB consumer GPU, https://huggingface.co/blog/trl-peft