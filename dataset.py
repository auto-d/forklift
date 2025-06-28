import tempfile 
import json 
import pandas as pd
from process import run_subprocess

def extract_symbol_defs(input_dir): 
    """
    Given a directory (full of code we hope), extract all symbol definitions and return. 
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

    # I'm sure there's a good use for ptags, but it's not jumping out at me on day one... for 
    # now leave them behind
    drop = symbols[symbols._type != 'tag'].index
    return symbols.drop(drop, inplace=True)

def extract_symbol_refs(input_dir): 
    """
    Recursively analyze a directory full of C/C++ source and extract all symbol references

    See https://cscope.sourceforge.net/large_projects.html

    NOTE: This function has a cscope dependency that isn't tracked elsewhere, the binary 
    will need to be accessible and in the path for this function to succeed. E.g.
    `sudo apt install cscope`    
    
    """

    # cscope wants a newline-separate list of files, oblige
    cmd = [
        'find',
        '.',
        input_dir,
        '-name',
        '"*.c"',   
        '-o', 
        '-name',
        '"*.h"',
        '-o', 
        '-name', 
        '"*.cpp"', 
        '-o', 
        '-name',
        '"*.hpp"']
    
    result, _ = run_subprocess(cmd )

    # We should now have a pair of indices, one for forward and one for reverse lookups... or
    # something like that. Looks like we can convince (maybe?) cscope to emit lines of the 
    # form: 
    # file_path, scope, line_no, source_line
    # 
    # For example: 
    # $ cscope -dL -0 mount_nodev_root 
    # ../../../linux/init/do_mounts.c mount_nodev_root 346 static int __init mount_nodev_root(char *root_device_name)
    # ../../../linux/init/do_mounts.c mount_root 403 mount_nodev_root(root_device_name) == 0)
    #
    # The above syntax "-L -<int>" is a cue to cscope to apply the function of the same index in the 
    # editor menu, here's the mapping (which can apparently vary between versions, yikes)
    FIND_SYMBOL       = 0 # Find this C symbol: 
    FIND_DEF          = 1 # Find this global definition:
    FIND_CHILD_FN     = 2 # Find functions called by this function:
    FIND_PARENT_FN    = 3 # Find functions calling this function:
    FIND_TEXT         = 4 # Find this text string:
    CHANGE_TEXT       = 5 # Change this text string:
    FIND_PATTERN      = 6 # Find this egrep pattern:
    FIND_FILE         = 7 # Find this file:
    FIND_INCLUDES     = 8 # Find files #including this file:
    FIND_ASSIGNMENTS  = 9 # Find assignments to this symbol:
    #
    # We could conceivable build a pretty awesome graph with just this information, 
    # though are we just working overtime to produce an AST-like structure? 
    


def build_symbol_map(defs, refs):
    """
    Given all symbol definitions and all symbol references, build a map of sorts to 
    inform the code decomposition and Q&A pair construction
    """ 
    pass

def build_prompt(): 
    pass

def build(input_dirs, output_dir): 
    """
    Given a list of directories, build a fine-tuning dataset and emit to the provided
    dataset directory
    """
    
    if type(input_dirs) != list or  len(input_dirs == 0): 
        raise ValueError("Expecting non-empty directory list!")
    
    for input_dir in input_dirs: 

        symbols = extract_symbol_defs(input_dir)
        for row in symbols.iterrows(): 
            print(row.name)
