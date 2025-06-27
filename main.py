import argparse 
import os
import naive
import hmm 
import slm 

def build(): 
    pass

def deploy(): 
    pass

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

    parser = argparse.ArgumentParser()

    subparsers = parser.add_subparsers(dest="mode", required=True)

    # Build mode 
    build_parser = subparsers.add_parser("build") 
    build_parser.add_argument("--inputs", nargs="+", type=readable_dir, help="List of code directories to target", required=True)
    build_parser.add_argument("--dataset", type=readable_dir, help="Directory to write resulting dataset to", required=True)

    # Train mode 
    train_parser = subparsers.add_parser("train") 
    train_parser.add_argument("--dataset", type=readable_dir, help="Directory containing input dataset to train on ", required=True)
    train_parser.add_argument("--model", type=nonexistent_file, help="File to write resulting models to (each will get a different suffix)", required=True)
    train_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Test mode 
    test_parser = subparsers.add_parser("test") 
    test_parser.add_argument("--model", type=readable_file, help="File to load model from")
    test_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')

    # Deploy mode 
    deploy_parser = subparsers.add_parser("deploy")
    deploy_parser.add_argument("--token", help="Token to use to authenticate to HF spaces")
    deploy_parser.add_argument("--model", type=readable_file, help="Directory to load model from")
    deploy_parser.add_argument("--type", choices=['naive', 'classic', 'neural'], default='neural')
    
    args = parser.parse_args()
    
    match args.mode:     
        case "build":
            build(args.inputs, args.dataset)

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
            deploy(args.model, args.token)

        case _:
            parser.print_help()

if __name__ == "__main__": 
    router() 
