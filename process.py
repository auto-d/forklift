import subprocess

def run_subprocess(cmd, output=False): 
    """
    Shell out to run a provided command.

    NOTE: assist from chatgpt on syntax: https://chatgpt.com/share/685ef572-0b24-8013-99df-fab2155c6e80
    """
    try:
        result = subprocess.run(cmd,text=True, capture_output=output)
        return True, result.stdout
        
    except subprocess.CalledProcessError as e:
        return False, e.stderr
        