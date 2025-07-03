import pandas as pd 
import torch
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM

def summarize_history(history): 
    """
    Mine some salient history 
    """
    df = pd.DataFrame(history) 
    s = (f"***************************\n")
    s += f"Completed {df.epoch.max():.4} epochs ({df.step.max()} steps)\n"
    s += f" - Loss {df.loss.max():.3f} -> {df.loss.min():.3f}\n"
    s += f" - Eval runtime: {df.eval_runtime.sum():.2f}s ({df.eval_runtime.mean():.2f}s/eval)\n"
    return s
    
def sft(dataset, model="facebook/opt-350m", batch_size=8, max_steps=-1, epochs=1, path=None):
    """
    Initiate a supervised fine-tuning run on the target model
    """

    model_args = {}
    config_args = {}
    gradient_checkpointing = False

    match model: 
        case "facebook/opt-350m":             
            config_args['max_length'] = 512

        case "Qwen/Qwen2.5-0.5B-Instruct" | "Qwen/Qwen2.5-0.5B":                         
            # With 24GB of VRAM we can just skate under the ceiling w/ 
            # a sequence of 512 and loading the base model in bfloat16, 
            # if full-precision is desired, try any of these: 
            #  - reduce sequence length here
            #  - enable gradient checkpointing (more time, less mem usage)
            #  - lower the batch size
            model_args['torch_dtype'] = torch_dtype=torch.bfloat16
            config_args['max_length'] = 512
            config_args['eos_token'] = "<|im_end|>"
        case _: 
            raise ValueError("Unsupported model!")
    
    # NOTE: Take care we don't violate this warning from SFTTrainer best practices (see docs)
    # If you create a model outside the trainer, make sure not to pass to the trainer any 
    # additional keyword arguments that are relative to from_pretrained() method.
    base = AutoModelForCausalLM.from_pretrained(model, **model_args)
    
    if gradient_checkpointing: 
        tqdm.write("Gradient checkpointing enabled")
        base.gradient_checkpointing_enable() 

    tqdm.write("Loaded base model: " + base.__class__.__name__)    

    # TODO: issue or warning or raise an exception when the inputs 
    # during training exceed the max_length provided here... 
    tqdm.write("Setting up training run")
    config = SFTConfig(
        # NOTE: From the documentation - SFTTrainer always truncates by default the 
        # sequences to the max_length argument of the SFTConfig. If none is passed, 
        # the trainer will retrieve that value from the tokenizer. Some tokenizers 
        # do not provide a default value, so there is a check to retrieve the minimum
        # between 1024 and that value. Make sure to check it before training.
        output_dir="runs",         
        num_train_epochs=epochs, 
        max_steps=max_steps, 
        per_device_train_batch_size=batch_size,
        eval_strategy="steps", 
        eval_steps=50, 
        report_to="tensorboard",         
        **config_args
    )
    
    # TODO: consider recruiting unsloth here to reduce memory usage and accelerate training ... 
    # see https://huggingface.co/docs/trl/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth

    trainer = SFTTrainer(
        base, 
        train_dataset=dataset['train'], 
        eval_dataset=dataset['test'], 
        args=config        
        ) 

    tqdm.write("Initiating training run.")
    
    # Support keyboard interrupts during training, yet still cough up some 
    # stats on the run to help assess efficiency, etc... 
    try: 
        trainer.train()
        tqdm.write('Training complete!')

        if path: 
            tqdm.write(f'Writing model to {path}...')
            trainer.save_model(path)

        return trainer.model, summarize_history(trainer.state.log_history)
    
    except KeyboardInterrupt as interrupt: 
        tqdm.write("Aborting run due to keyboard interrupt!")
        return None, summarize_history(trainer.state.log_history)

def dpo(dataset:Dataset, model):
    """
    Apply direct preference optimization in lieu of a struggling with the complexity 
    of learning a reward function as reinforcement learning requires. 
    """
    pass

def build_sft_dataset(dataset): 
    """
    Createa a huggingface SFTrainer-compatible Dataset 

    SFTrainer expects (per https://huggingface.co/docs/trl/en/sft_trainer#trl.SFTTrainer)
    inputs of the form:  
      {"prompt": "<prompt text>", "completion": "<ideal generated text>"}

    """    
    tqdm.write('Building SFT dataset...')

    copy = dataset.rename(columns={'x':'prompt','y':'completion'})
    copy.drop(['yp','yu','x0','x2'], axis=1, inplace=True)
    sft_dataset = Dataset.from_pandas(copy)

    tqdm.write(f"Done ({len(sft_dataset)} rows)!")

    return sft_dataset

def build_dpo_dataset(dataset): 
    """
    Assemble the DPO dataset for TRL/DPO interface 

    DPOTrainer exects (per https://huggingface.co/docs/trl/en/dpo_trainer#trl.DPOTrainer.tokenize_row) 
    inputs of the form 
      {"prompt": "The sky is", "chosen": " blue", "rejected": " green"}

    """
    tqdm.write('Building DPO dataset...')

    copy = dataset.rename(columns={'x2':'prompt','yp':'chosen','yu':'rejected'})
    copy.drop(['x0','x', 'y'], axis=1, inplace=True)
    dpo_dataset = Dataset.from_pandas(copy)
    
    tqdm.write(f"Done ({len(dpo_dataset)} rows)!")

    return dpo_dataset

def train(dataset_path, model_path, batch_size=8, steps=None, epochs=1): 
    """
    Train our derivitative model 
    """    
    tqdm.write(f"Loading dataset {dataset_path}...")
    dataset = pd.read_parquet(dataset_path)
    sft_dataset = build_sft_dataset(dataset) 

    test_train_ratio = 0.1
    tqdm.write(f"Holding {test_train_ratio*100}% out for validation...")    
    sft_dataset = sft_dataset.train_test_split(test_size=test_train_ratio)

    model, summary = sft(
        dataset=sft_dataset, 
        model="Qwen/Qwen2.5-0.5B-Instruct",
        path=model_path, batch_size=batch_size, max_steps=steps, epochs=epochs)    
    tqdm.write(summary)

    # TODO: add a sanity check to validate the model's sequence length and flag any 
    # training data that will be truncated during training
    tqdm.write("Supervised fine-tuning complete!") 

def test(dataset):
    pass