from trl import SFTTrainer, SFTConfig
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM

def sft(dataset:Dataset, model="facebook/opt-350m", batch_size=8, max_steps=None, epochs=1, path=None):     
    """
    Initiate a supervised fine-tuning run on the target model
    """

    # NOTE: Take care we don't violate this warning from SFTTrainer best practices (see docs)
    # If you create a model outside the trainer, make sure not to pass to the trainer any 
    # additional keyword arguments that are relative to from_pretrained() method.
    base = AutoModelForCausalLM.from_pretrained(model)
    print("Loaded base model: " + base)

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
        output_dir="/tmp", 
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
        args=training_args,
        ) 

    print("Initiating training run.")
    trainer.train()

    print('Training complete: ' + trainer.model)

    if path: 
        print(f'Writing model to {path}...')
        trainer.save_model(path)

    return trainer.model

def train(dataset, path): 
    
    # TODO: apply our own dataset here .. 
    dataset = load_dataset("stanfordnlp/imdb", split="train")

    # TODO: determine epochs we need if not 3, either way swap in here for the artificial step limit
    model = sft(dataset=dataset, path=path, max_steps=10)
    
    print("Trained model: " + model) 

def test(dataset):
    pass