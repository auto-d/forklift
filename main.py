import argparse 
import os
import naive
import hmm 
import slm 
import tempfile
import glob
import dataset
from process import run_subprocess

def deploy_demo(token): 
    """
    Deploy a model to HuggingFace Spaces, optionally refreshing the code in the 
    container prior
    """
    
    # Inconsistent results pushing via huggingface-cli and https, rely on 
    # SSH (pubkey loaded on distant end) 
    spaces_https_url = "https://huggingface.co/spaces/3emaphor/forklift"
    spaces_git_url = "git@hf.co:spaces/3emaphor/forklift"

    with tempfile.TemporaryDirectory() as tmp: 

        print("Attempting to push ./demo/* to {spaces_git_url}...")
        cmds = []
        cmds.append(["git", "clone", spaces_git_url, tmp]) 
        cmds.append(["cp"] + glob.glob("./demo/*") + [tmp])
        cmds.append(["git", "-C", tmp, "add", "."])
        cmds.append(["git", "-C", tmp, "commit", "-m", "automated deploy"])
        cmds.append(["git", "-C", tmp, "push"])
        
        for cmd in cmds: 
            result, text = run_subprocess(cmd) 

        print("Completed! {spaces_https_url} should be redeploying ... now. ")
        
    return result, text

def deploy(token, model, refresh=False): 
    """
    Deploy a model, optionall refreshing the underlying demo app in the process. 
    NOTE: we've got a copy of the token here, but really just needs to live in 
    the environment so the 
    """
    if refresh: 
        deploy_demo(token) 

def load_secrets(): 
    """
    Find our secrets and return them
    """
            
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token: 
        with open("secrets/huggingface.token", "r") as file: 
            hf_token = file.read()
    
    openai_key = os.environ.get("OPENAI_API_KEY")
    if not openai_key: 
        with open("secrets/openai.key", "r") as file: 
            openai_key = file.read() 
    
    return hf_token, openai_key 

def readable_file(path):
    """Test for a readable file"""
    if os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' already exists.")
    return path

def nonexistent_file(path):
    """Test for a non-existent file (to help avoid overwriting important stuff)"""
    if os.path.exists(path):
        raise argparse.ArgumentTypeError(f"'{path}' already exists.")
    return path

def readable_dir(path):
    """Test for a readable dir"""
    if os.path.isdir(path):
        return path
    else:
        raise argparse.ArgumentTypeError(f"'{path}' is not a valid directory.")
    
def nonexistent_dir(path): 
    """Test to ensure directory doesn't exist"""    
    if os.path.exists(path):
        if os.path.isdir(path):
            raise argparse.ArgumentTypeError(f"Directory '{path}' already exists.")
        else:
            raise argparse.ArgumentTypeError(f"Path '{path}' exists and is not a directory.")
    return path

def router(): 
    """
    Argument processor and router

    @NOTE: Argparsing with help from chatgpt: https://chatgpt.com/share/685ee2c0-76c8-8013-abae-304aa04b0eb1
    """

    parser = argparse.ArgumentParser("forklift", description="Code repository data synthesis and LLM fine-tuning")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build mode 
    build_parser = subparsers.add_parser("build") 
    build_parser.add_argument("--inputs", nargs="+", type=readable_dir, help="List of code directories to target", required=True)
    build_parser.add_argument("--dataset", help="Directory to write resulting dataset to", required=True)

    # Train mode 
    train_parser = subparsers.add_parser("train") 
    train_parser.add_argument("--dataset", type=readable_file, help="Directory containing input dataset to train on ", required=True)
    train_parser.add_argument("--model", type=nonexistent_file, help="File to write resulting models to (each will get a different suffix)", required=True)
    train_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Test mode 
    test_parser = subparsers.add_parser("test") 
    test_parser.add_argument("--model", type=readable_file, help="File to load model from")
    test_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Deploy mode 
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--model", type=readable_file, help="Directory to load model from")
    deploy_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')
    deploy_parser.add_argument("--refresh",  action="store_true", help="Whether or not to refresh the server code prior to deployment.", default=False)
    
    args = parser.parse_args()
    
    hf_token, openai_key = load_secrets()
    match args.mode:     
        case "build":
            if openai_key: 
                paths = args.inputs if type(args.inputs) == list else [args.inputs]
                dataset.build(paths, openai_key, args.dataset)
            else: 
                print("No OpenAI API key found!")

        case "train":
            if args.type == 'naive':
                naive.train(args.dataset, args.model)
            if args.type == 'classic':
                hmm.train(args.dataset, args.model) 
            if args.type == 'neural': 
                slm.train(args.dataset, args.model)

        case "test":
            if args.type == 'naive':
                naive.test(args.dataset)
            if args.type == 'classic':
                hmm.test(args.dataset) 
            if args.type == 'neural': 
                slm.test(args.dataset)

        case "deploy":            
            if hf_token: 
                deploy(args.model, hf_token, args.refresh)
            else: 
                print("No huggingface token found!")
        case _:
            parser.print_help()

if __name__ == "__main__": 
    router() 
