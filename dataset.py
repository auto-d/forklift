import tempfile 
import json 
import pandas as pd
from process import run_subprocess

def extract_symbols(input_dir): 
    """
    Given a directory (full of code we hope), extract all symbol information and return. 
    We could cache this output, but ctags is so fast it doesn't seem worth it given the 
    length of time each training run will take. 
    
    NOTE: this relies on the presence of a suitably-built ctags binary in the path, 
    see https://github.com/universal-ctags/ctags for build and install help 
    
    `ctags -R --output-format=json --fields=+nksSaf --extras=+q -o linux_kernel.ctags ../../../linux/kernel/`
    """

    symbols = None 

    with tempfile.NamedTemporaryFile() as tmp_file: 
        cmd = [
            'ctags',
            # Recursively process contents
            '-R',                    
            # Emit JSON (ensure this is compiled in if you are doing a manual ctags setup)
            '--output-format=json', 
            # Line number, kind/type, scope, signature/args, access, file scope
            '--fields=+nksSaf', 
            # Ensure scoping information is included
            '--extras=+q',
            # Write out pile of json to this file
            '-o', tmp_file, 
            # Dir full of stuff for ctags to work on... 
            input_dir
            ]

        result, _ = run_subprocess(cmd)

        # Weirdly the ctags json output is invalid as a document, but valid within each line... 
        # maybe an efficiency thing. Anyway, parse each document separately as as dict to 
        # streamline the ingest into a dataframe
        data = []
        with open(tmp_file, "r") as tag_file: 
            for line in tag_file: 
                data.append(json.loads(line))

        symbols = pd.DataFrame(data)
    
    return symbols 

def build(input_dirs, output_dir): 
    """
    Given a list of directories, build a fine-tuning dataset and emit to the provided
    dataset directory
    """
    
    if type(input_dirs) != list or  len(input_dirs == 0): 
        raise ValueError("Expecting non-empty directory list!")
    
    for input_dir in input_dirs: 
        symbols = extract_symbols(input_dir)
