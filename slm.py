import pandas as pd 
from tqdm import tqdm
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM

def sft(dataset:Dataset, model="facebook/opt-350m", batch_size=1, max_steps=-1, epochs=1, path=None):
    """
    Initiate a supervised fine-tuning run on the target model
    """

    # NOTE: Take care we don't violate this warning from SFTTrainer best practices (see docs)
    # If you create a model outside the trainer, make sure not to pass to the trainer any 
    # additional keyword arguments that are relative to from_pretrained() method.
    base = AutoModelForCausalLM.from_pretrained(model)
    tqdm.write("Loaded base model: " + base.__class__.__name__)
    tqdm.write("-------------------------------------------------")
    tqdm.write(base)

    # TODO: investigate whether a change in precision is supportable here!
    # per https://huggingface.co/docs/trl/en/sft_trainer#control-over-the-pretrained-model
    # ... we can load the base model in an alternative precision here: 
    # model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m", torch_dtype=torch.bfloat16)
    # training_args = SFTConfig(
    #     model_init_kwargs={
    #         "torch_dtype": "bfloat16",
    #     },
    #     output_dir="/tmp",
    # )
    # trainer = SFTTrainer(
    #     "facebook/opt-350m",
    #     train_dataset=dataset,
    #     args=training_args,
    # )

    # TODO: issue or warning or raise an exception when the inputs 
    # during training exceed the max_length provided here... 
    config = SFTConfig(
        # NOTE: From the documentation - SFTTrainer always truncates by default the 
        # sequences to the max_length argument of the SFTConfig. If none is passed, 
        # the trainer will retrieve that value from the tokenizer. Some tokenizers 
        # do not provide a default value, so there is a check to retrieve the minimum
        #  between 1024 and that value. Make sure to check it before  training.
        max_length=512, 
        output_dir="runs", 
        num_train_epochs=epochs, 
        max_steps=max_steps, 
        per_device_train_batch_size=batch_size
    )
    
    # TODO: consider recruiting unsloth here to reduce memory usage and accelerate training ... 
    # see https://huggingface.co/docs/trl/en/sft_trainer#accelerate-fine-tuning-2x-using-unsloth

    trainer = SFTTrainer(
        base, 
        train_dataset=dataset, 
        # TODO: add a validation set here ... just split the input data or is there 
        # a reason to ask for a separate dataset? The same split could be used across multiple
        # training runs, which we'd want to leave to the client/caller
        eval_dataset=None, 
        args=config,
        ) 

    tqdm.write("Initiating training run.")
    trainer.train()

    tqdm.write('Training complete!')

    if path: 
        tqdm.write(f'Writing model to {path}...')
        trainer.save_model(path)

    return trainer.model

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

def train(dataset_path, model_path, steps=None, epochs=1): 
    """
    Train our derivitative model 
    """
    
    dataset = pd.read_parquet(dataset_path)
    sft_dataset = build_sft_dataset(dataset) 

    # TODO manage the validation and data splits here or earlier to avoid 
    # training data sneaking into validation sets (e.g. )
    model = sft(dataset=sft_dataset, path=model_path, max_steps=steps, epochs=epochs)
    
    #TODO: add a sanity check to validate the model's
    tqdm.write("Supervised fine-tuning complete!") 

    dpo_dataset = build_dpo_dataset(dataset)
    model = dpo(dataset=dpo_dataset, model=model)

    tqdm.write("DPO complete!") 

def test(dataset):
    pass