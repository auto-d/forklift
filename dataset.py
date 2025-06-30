import os
import asyncio 
import tempfile 
import json 
import pandas as pd
import time 
from tqdm import tqdm
from openai import AsyncOpenAI
from process import run_subprocess

class CodebaseAnalyzer(): 

    # The cscope syntax "-L -<int>" is a cue to cscope to apply the function of the same index in the 
    # editor menu. Mapping follows (which can apparently vary between versions, yikes)... 
    CSCOPE_FIND_SYMBOL       = "-0" # Find this C symbol: 
    CSCOPE_FIND_DEF          = "-1" # Find this global definition:
    CSCOPE_FIND_CHILD_FN     = "-2" # Find functions called by this function:
    CSCOPE_FIND_PARENT_FN    = "-3" # Find functions calling this function:
    CSCOPE_FIND_TEXT         = "-4" # Find this text string:
    CSCOPE_CHANGE_TEXT       = "-5" # Change this text string:
    CSCOPE_FIND_PATTERN      = "-6" # Find this egrep pattern:
    CSCOPE_FIND_FILE         = "-7" # Find this file:
    CSCOPE_FIND_INCLUDES     = "-8" # Find files #including this file:
    CSCOPE_FIND_ASSIGNMENTS  = "-9" # Find assignments to this symbol:
    
    def __init__(self, input_dir, backends=[]): 
        """
        Create a new object scoped to the code in the provided input directory. Pass an OpenAI
        API key to hit open AI models, none to try Ollama locally
        """
        self.input_dir = input_dir
        self.defs = None
        self.cscope_indexed = False
        
        for backend in backends: 
            match backend['type']: 
                case "openai": 
                    backend['client'] = AsyncOpenAI(api_key=backend['api_key'])
                case "ollama":                    
                    backend['client'] = AsyncOpenAI(api_key="none", base_url=backend['url'])
                case _: 
                    raise ValueError("Unexpected backend type!")
                
        self.backends = backends
        self.usage = []
        

    def extract_symbol_defs(self, refresh=False): 
        """
        Given a directory (full of code we hope), extract all symbol definitions and return. 
        We could cache this output, but ctags is so fast it doesn't seem worth it given the 
        length of time each training run will take. 
        
        NOTE: this approach was adopted based on gpt-4o recommendation of ctags for symbol mining
        see conversation here: https://chatgpt.com/share/6860d0c4-363c-8013-b3cb-c6c413d10c46

        NOTE: this relies on the presence of a suitably-built ctags binary in the path, 
        see https://github.com/universal-ctags/ctags for build and install help 
        
        `ctags -R --output-format=json --fields=+nksSaf --extras=+q -o linux_kernel.ctags ../../../linux/kernel/`
        """

        if self.defs != None and refresh == False: 
            print("Found existing symbol definition table, aborting definition extraction!")
            return  

        print(f"Extracting symbols from {self.input_dir}..")

        ctags_file = tempfile.mkstemp()[1]
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
            '-o', ctags_file, 
            # Dir full of stuff for ctags to work on... 
            self.input_dir
            ]

        result, _ = run_subprocess(cmd)

        # Weirdly the ctags json output is invalid as a document, but valid within each line... 
        # maybe an efficiency thing. Anyway, parse each document separately as as dict to 
        # streamline the ingest into a dataframe
        data = []
        with open(ctags_file, "r") as tag_file: 
            for line in tag_file: 
                data.append(json.loads(line))

        os.remove(ctags_file)

        self.defs = pd.DataFrame(data)

        # I'm sure there's a good use for ptags, but it's not jumping out at me on day one... for 
        # now leave them behind
        drop = self.defs[self.defs._type != 'tag'].index
        self.defs.drop(drop, inplace=True)

        # Convert lines to ints and paste empty strings over the NaNs
        self.defs['line'] = self.defs['line'].fillna(0).astype(int)
        self.defs.fillna('', inplace=True)

        print (f"Extracted {len(self.defs)} definitions.")

    def parse_cscope_lines(self, lines): 
        """
        Parse a Cscope output line of the form "filename, symbol, line_no, line_text" and 
        return a dict with same info. 
        """
        line_dicts = []
        if not lines or len(lines) == 0: 
            return []
        
        for line in lines.split('\n'): 
            line_dict = {}
            
            if line == '': 
                continue
                
            fields = line.split(' ')
            if len(fields) >= 4: 
                line_dict['file'] = fields[0]
                line_dict['symbol'] = fields[1]
                line_dict['line'] = fields[2]
                line_dict['text'] = ''.join(fields[3:])
            else: 
                raise ValueError("Can't unpack all fields in cscope output!")
        
            line_dicts.append(line_dict)

        return line_dicts

    def print_cscope_dicts(self, dicts): 
        """
        Print a dict returned by parse_cscope_lines into a legible structure 
        """
        s = "" 
        for dict_ in dicts: 
            s+= f"file:{dict_['file']}\nsymbol:{dict_['symbol']}\nline:{dict_['line']}\ncode:{dict_['text']}"

        return s

    def extract_symbol_refs(self, symbol, refresh=False): 
        """
        Recursively analyze a directory full of C/C++ source and extract all symbol references

        See https://cscope.sourceforge.net/large_projects.html

        NOTE: This function has a cscope dependency that isn't tracked elsewhere, the binary 
        will need to be accessible and in the path for this function to succeed. E.g.
        `sudo apt install cscope`    
        """

        if self.cscope_indexed == False or refresh == True:             
        
            print (f"Extracting symbol references for {symbol}...")

            cscope_listing_file = tempfile.mkstemp()[1]

            # cscope wants a newline-separate list of files, oblige 
            print("Gathering input files...")
            cmd = [
                'find',
                self.input_dir,
                '-name',
                '*.c',   
                '-o', 
                '-name',
                '*.h',
                '-o', 
                '-name', 
                '*.cpp', 
                '-o', 
                '-name',
                '*.hpp', 
                ]
            result, files = run_subprocess(cmd, output_file=cscope_listing_file)

            # Now build the indices for all associated definitions    
            print("Preparing cscope indices... ")
            cmd = [
                'cscope', 
                # Build listing only 
                '-b',
                # Ignore /usr/include aka 'kernel' mode
                '-k', 
                # "Quick" index build
                '-q',
                # Recursive 
                '-R',
                # Build index on these files
                '-i', f'{cscope_listing_file}'
                ]
            run_subprocess(cmd)
            self.cscope_indexed = True

        # We should have a pair of indices, one for forward and one for reverse lookups... or
        # something like that. Cscope's line-oriented format produces each reference in the 
        # form (see cscope man page) below, with a single space between each field: 
        #         
        # file_path function_name line_number line_text
        # 
        # For example: 
        # $ cscope -dL -0 mount_nodev_root 
        # ../../../linux/init/do_mounts.c mount_nodev_root 346 static int __init mount_nodev_root(char *root_device_name)
        # ../../../linux/init/do_mounts.c mount_root 403 mount_nodev_root(root_device_name) == 0)

        result = {}
        result['symbol'] = symbol
        
        _, text = run_subprocess(["cscope", "-k", "-dL", self.CSCOPE_FIND_SYMBOL, symbol], output=True)
        result['refs'] = self.parse_cscope_lines(text)

        _, text = run_subprocess(["cscope", "-k", "-dL", self.CSCOPE_FIND_DEF, symbol], output=True)
        result['defs'] = self.parse_cscope_lines(text) 

        _, text = run_subprocess(["cscope", "-k", "-dL", self.CSCOPE_FIND_CHILD_FN, symbol], output=True)
        result['children'] = self.parse_cscope_lines(text) 

        _, text = run_subprocess(["cscope", "-k", "-dL", self.CSCOPE_FIND_PARENT_FN, symbol], output=True)
        result['parents'] = self.parse_cscope_lines(text) 

        _, text = run_subprocess(["cscope", "-k", "-dL", self.CSCOPE_FIND_ASSIGNMENTS, symbol], output=True)
        result['asnmts'] = self.parse_cscope_lines(text) 

        #print (f"Extracted {len(symbols)} definitions.")
        return {} if result['refs'] == '' else result
        
    def print_refs(self, refs): 
        if refs:                
            for key, value in refs.items(): 
                #if key == 'symbol': 
                #    print(f"{key}:{value}\n---------------------------")
                print(f"{key}:\n{value}")

    def build_symbol_map(self, defs, refs):
        """
        Given all symbol definitions and all symbol references, build a map of sorts to 
        inform the code decomposition and Q&A pair construction
        """ 
        pass

    async def complete(self, message, backend, context=None, system=None, temperature=0.5):
        """
        Push a chat completion request out to the void and collect the response ... we really need 
        to be able to run these in parallel given the volume of queries we have

        NOTE: with help from https://platform.openai.com/docs/guides/text?api-mode=chat&lang=python
        """

        usage = []
        start = time.time()

        messages = []
        if system: 
            messages.append({ "role": "system", "content": system })
        if context: 
            messages.append({ "role": "user", "content": f"source search results:\n{context}" })
        messages.append({ "role": "user","content": message })

        completion = await backend['client'].chat.completions.create(
            model=backend['model'],
            messages=messages,
            temperature=temperature
        )

        response = completion.choices[0].message.content

        in_tokens = completion.usage.prompt_tokens
        out_tokens = completion.usage.completion_tokens
    
        self.usage.append({'model':backend['model'],'in_tokens':in_tokens, 'out_tokens': out_tokens, 'time': time.time() - start})
        
        return response

    async def embellish_prompt_for_backend(self, x, context, backend):
        """
        Given a prompt about the codebase we're analyzing, and some hopefully useful
        context, permute a few different variations and harness the provided model 
        to build reasonable input/outputs to support SFT and DPO... return
        - x0: the motivating question/prompt (not intended to be used for training 
          but useful for comparison to generated sequences)
        - x: the question/prompt (SFT input sequence)
        - y: an authoritative answer (SFT target sequence)
        - x2: another framing of the question (DPO input sequence)
        - yp: another authoritative answer (not the same) (DPO preferred)
        - yu: a weak answer (DPO unpreferred)

        NOTE: Hasty prompt development with the help of gpt4o: https://chatgpt.com/share/6860bcae-7674-8013-9785-e99a78814b6d
        """
        y_system = "You are a Linux kernel expert. Provide a clear, CONCISE and rigorous answer using kernel-specific terminology and referencing relevant APIs, subsystems, or patterns. You will be provided with code search results from the Linux kernel which are authoritative."
        yp_system = "As a senior Linux kernel developer, provide a clear, CONCISE, and well-reasoned response using kernel idioms and terminology. You will be provided with code search results from the Linux kernel which are authoritative. "
        yu_system = "You are a Linux kernel specialist. Your responses should be CONCISE."

        set_ = {}

        # Generate a completion, giving the SFT and DPO+ response the advantage of a 
        # better system prompt, lower temperature (hypothesize that creativity=bad in 
        # code Q&A) and context to enrich the response... 
        set_['y'] = await self.complete(x, backend, system=y_system, context=context, temperature=0.3)
        set_['yp'] = await self.complete(x, backend, system=yp_system, context=context, temperature=0.3)
        set_['yu'] = await self.complete(x, backend, system=yu_system, temperature=0.7)

        # Generate a plausible question/prompt that induced the above completion to 
        # replace our stiff and unimaginative initial prompt. The variation here creates 
        # a much larger space of inputs to and will hopefully reduce overfit and retain 
        # more realistic style of human interaction with the resulting model.
        set_['x0'] = x
        bwd_prompt = f"We are building a Q&A dataset. Write a question about the Linux kernel that the following text would be an appropriate answer to.\n----\n{set_['y']}\n----\nAnswer ONLY with the question or prompt."
        set_['x'] = await self.complete(bwd_prompt, backend, system=y_system, context=context, temperature=0.3)
        set_['x2'] = await self.complete(bwd_prompt, backend, system=y_system, context=context, temperature=0.5) 

        return set_
    
    async def embellish_gatherer(self, tasks): 
        """
        Accumulate our prompt embellishment tasks - annoying but necessary 
        interlocutor if we don't want half our whole app to become async-aware
        """
        return await asyncio.gather(*tasks)
    
    def embellish_prompt(self, x, context):
        """
        Async task dispatcher to parallelize chat completions on multiple backends and 
        gather up the results ... we really need more parallelism than this, but the local 
        backends we're using aren't constrained by network latency or other queued requests, 
        it's more about available compute. We'd really only realize a gain with openAI
        """
        tasks = []
        for backend in self.backends: 
            tasks.append(self.embellish_prompt_for_backend(x, context, backend))

        sets = asyncio.run(self.embellish_gatherer(tasks))

        return sets

    def generate_symbol_prompts(self, defn, refs): 
        """
        Given a symbol definition, and all references to it, apply a series of heuristics 
        that are an amalgamation of canned text, symbol citations, and LLM completions for 
        the purpose of building associated input/output sequences to support training. 
        
        Because we want to support both SFT and DPO, we need a trio of prompts as follows: 
        - x:  a high quality input sequence that captures a facet of the underlying code
        - y:  a high quality output sequence  " " " 
        - x:  a second, closely related input sequence
        - y+: a high quality output sequence associated with x+ 
        - y-: a lower quality output sequence associated with x+ 
        
        It's tempting to reuse SFT pairs in DPO, but we should be concerned that identicaly 
        inputs will reinforce some overfit/memorization that happened during SFT. 

        We have the following data to use to build out our seqeuences: 

        Symbol defintion (from ctags): 

        Key                             Example Value   
        ----------------------------------------------------------------------------------
        name                            rd_load_image
        path                            ../../linux/init/do_mounts.h
        pattern                         /^static inline int rd_load_image(char *from) ...
        parserName                      NaN
        kindName                        NaN
        line                            33.0
        kind                            function
        typeref                         typename:int
        scope                           NaN
        scopeKind                       NaN
        file                            NaN
        access                          NaN
        signature                       (char * from)

        Symbol references (result of search on above symbol with cscope) : 
        Other than the symbol, each key may contain an arbitrary number of references, each 
        

        Key       Description and Example Value
        ----------------------------------------------------------------------------------
        symbol    symbol name, should always match above
                rd_load_image 

        refs      all references found to symbol, examples follow

            file path                             scope               line  line text
            -------------------------------------------------------------------------
            ../../linux/init/do_mounts.h          <global>            28    int __init rd_load_image(char *from);
            ../../linux/init/do_mounts.h          rd_load_image       33    static inline int rd_load_image(char *from) { return 0; }
            ../../linux/init/do_mounts_initrd.c   initrd_load         146   if (rd_load_image("/initrd.image") && ROOT_DEV != Root_RAM0) {
            ../../linux/init/do_mounts_rd.c       rd_load_image       186   int __init rd_load_image(char *from)
            ../../linux/init/do_mounts_rd.c       rd_load_disk        283   return rd_load_image("/dev/root");

        defs:      all definitions of symbol, examples follow

            file path                             scope               line  line text
            -------------------------------------------------------------------------
            ../../linux/init/do_mounts.h          rd_load_image       33    static inline int rd_load_image(char *from) { return 0; }
            ../../linux/init/do_mounts_rd.c       rd_load_image       186   int __init rd_load_image(char *from)

        children  functions called by this function, ...

            file path                             scope               line  line text
            -------------------------------------------------------------------------
            ../../linux/init/do_mounts_rd.c       defined             194   #if !defined(CONFIG_S390)
            ../../linux/init/do_mounts_rd.c       filp_open           198   out_file = filp_open("/dev/ram", O_RDWR, 0);
            ../../linux/init/do_mounts_rd.c       IS_ERR              199   if (IS_ERR(out_file))
            ../../linux/init/do_mounts_rd.c       filp_open           202   in_file = filp_open(from, O_RDONLY, 0);
            ../../linux/init/do_mounts_rd.c       IS_ERR              203   if (IS_ERR(in_file))
            ../../linux/init/do_mounts_rd.c       identify_ramdi...   207   nblocks = identify_ramdisk_image(in_file, in_pos, &decompressor);
            ../../linux/init/do_mounts_rd.c       crd_load            212   if (crd_load(decompressor) == 0)
            ../../linux/init/do_mounts_rd.c       nr_blocks           221   rd_blocks = nr_blocks(out_file);
            ../../linux/init/do_mounts_rd.c       printk              223   printk("RAMDISK: image too big! (%dKiB/%ldKiB)\n",
            ../../linux/init/do_mounts_rd.c       strcmp              231   if (strcmp(from, "/initrd.image") == 0)
            ../../linux/init/do_mounts_rd.c       nr_blocks           234   devblocks = nr_blocks(in_file);
            ..
            ../../linux/init/do_mounts_rd.c       kfree               274   kfree(buf);
            ../../linux/init/do_mounts_rd.c       init_unlink         275   init_unlink("/dev/ram");
        
        parents  functions thsi function calls

            file path                             scope               line  line text
            -------------------------------------------------------------------------
            ../../linux/init/do_mounts_initrd.c   initrd_load         146   if (rd_load_image("/initrd.image") && ROOT_DEV != Root_RAM0) {
            ../../linux/init/do_mounts_rd.c       rd_load_disk        283   return rd_load_image("/dev/root");

        asgnmts  assignments associated with this symbol

            file path                             scope               line  line text
            -------------------------------------------------------------------------
            ../../linux/init/initramfs.c          parse_header        209   rdev = new_encode_dev(MKDEV(parsed[9], parsed[10])); 
                            ^ example borrowed from the `rdev `symbol for completeness, 
                            these arrays can often be empty (no cscope match)

        """
        sets = []
        
        symbol = defn['name']
        kind = defn['kind']
        path = defn['path']
        line = defn['line']
        sig = defn['signature']

        # Volume of training data isn't going to be a problem, we can augment so many 
        # ways... temperature, weaker models, system prompt variation, etc... we really need 
        # a graph at some point, but for now rely on the model to make the connections internally
        # between the web of symbols and their links outlined here
        # TODO: go deeper on these questions once more source context is available... we really need to 
        # toss entire functions into the context for the embellishing
        # notably, implement the Q&A suggestions here: https://chatgpt.com/share/6860d282-ab3c-8013-89cc-7709cdf36bf7
        # and don't forget to reference the conversation: 
        # General Architecture and Design
        # What is the overall architecture and modular structure of the Linux kernel?
        # How are kernel subsystems organized in the source tree?
        # What are the roles of the main directories like kernel/, arch/, drivers/, fs/, and net/?
        # How does the Linux kernel handle hardware abstraction?
        # What are the core components responsible for process and memory management?
        # Build and Configuration
        # How is the Linux kernel configured and built?
        # What files control the kernel build process?
        # How do Kconfig and Makefiles work together to configure the kernel build?
        # How can one add or modify kernel configuration options?
        # Kernel Initialization
        # Where is the kernel’s entry point during system boot?
        # How does the Linux kernel initialize hardware and drivers at startup?
        # What is the role of the initramfs in kernel initialization?
        # Process Management and Scheduling
        # How does the kernel implement process scheduling?
        # Where is the scheduler code located?
        # What data structures represent processes or tasks?
        # How are system calls handled and dispatched?
        # Memory Management
        # How does the kernel manage physical and virtual memory?
        # What are the major components of the memory management subsystem?
        # Where are page tables and virtual memory managed in the source?
        # How does the kernel handle memory allocation and freeing?
        # Device Drivers
        # How are device drivers organized in the Linux kernel source?
        # What is the structure of a typical device driver?
        # How are drivers registered and initialized?
        # How does the kernel communicate with hardware devices?
        # Filesystems
        # Where is the implementation of the Virtual Filesystem (VFS)?
        # How are individual filesystem drivers implemented?
        # How does the kernel handle mounting and unmounting filesystems?
        # What mechanisms are used for caching and buffering filesystem data?
        # Networking
        # Where is the networking stack implemented?
        # How does the kernel handle network protocols?
        # How are network devices registered and managed?
        # What is the role of the socket interface in the kernel?
        # Synchronization and Concurrency
        # How does the kernel handle synchronization between processes and interrupts?
        # What locking primitives are available in the kernel?
        # How does the kernel prevent deadlocks and race conditions?
        # Debugging and Instrumentation
        # What tools or mechanisms exist in the kernel for debugging?
        # How are kernel logs and printk messages generated and managed?
        # Where is support for tracing and performance monitoring implemented?
        # Security
        # How does the kernel enforce security policies?
        # What subsystems are responsible for access control?
        # How are capabilities and privileges managed?
        # Kernel API and Internal Interfaces
        # What are the main internal APIs for kernel modules?
        # How can external modules interact with the kernel?
        # What conventions are used for coding style and documentation?
        # Versioning and Maintenance
        # How is versioning handled within the kernel source?
        # Where are changelogs or commit histories maintained?
        # How are patches and contributions managed?
        # Specific Questions about Source Files or Functions
        # What is the purpose of init/main.c?
        # How does schedule() function operate internally?
        # Where is the implementation of do_fork() located?
        # What is the role of the syscall_table?

        # Symbols    
        sets.extend(self.embellish_prompt(f"What file is {symbol} defined in?", context=f"{kind} {symbol} {sig} found in file:{path} @ line {line}.")) 
        #TODO: implement recursive summarzation block-> function-> file -> directory -> parent ... 
        #TODO: implement grepping for contextual lines beyond the single identified line here 
        #grep_context = retrieve_context(file, line, size=5)
        #prompt_set += self.embellish_prompt(f"What is the purpose of {symbol} ?", context=grep_context) 
        #prompt_set += self.embellish_prompt(f"What roles does {symbol} have in the system?", context=grep_context) 
        #prompt_set += self.embellish_prompt(f"How is {symbol} relevant to the system architecture?", context=grep_context) 
        #prompt_set += self.embellish_prompt(f"How is the file {symbole} resides in organized ?", context=file-context) 
        #prompt_set += self.embellish_prompt(f"What other files and symbols are important in {symbol}'s subsystem?", context=file-list_context) 
        #prompt_set += self.embellish_prompt(f"What subsystem does {symbol} belong to?", context=grep_context) 
        #prompt_set += self.embellish_prompt(f"Is memory being managed by {symbol}?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"What depenencies does {symbol} have?) 
        #prompt_set += self.embellish_prompt(f"What happens if {symbol} is corrupted?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"Does {symbol} process network traffic?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"Is {symbol} associated with security policies?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"Does {symbol} assist with threading?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"Can {symbol} interact with user mode processes?", context=grep_context_all_refs) 
        #prompt_set += self.embellish_prompt(f"What APIs might be associated with {symbol}?", context=grep_context_all_refs) 

        # Symbol References        
        for ref in refs['refs']: 
            # TODO: ref-specific prompts
            pass 

        sets.extend(self.embellish_prompt(f"Where is the {symbol} referenced?", context=self.print_cscope_dicts(refs['refs']))) 

        # Symbol Definitions
        for def_ in refs['defs']: 
            # TODO: definition-specific prompts            
            pass
    
        sets.extend(self.embellish_prompt(f"Where else is the {symbol} defined?", context=self.print_cscope_dicts(refs['defs']))) 

        # Functions called from associated function's scope (if any)
        for child in refs['children']:  
            # TODO: child-specific...           
            pass

        sets.extend(self.embellish_prompt(f"What functions does {symbol} call?", context=self.print_cscope_dicts(refs['children']))) 

        # Functions that call the associated function symbol (if any)
        for parent in refs['parents']:
            # TODO: parent specific prompts
            pass

        sets.extend(self.embellish_prompt(f"What functions is {symbol} called by?", context=self.print_cscope_dicts(refs['parents'])))
                                                                                                                        
        # Assignment operations (if any)
        for asnmt in refs['asnmts']:       
            # TODO: assignment prompts..      
            pass

        sets.extend(self.embellish_prompt(f"Where are assignments made to {symbol}?", context=self.print_cscope_dicts(refs['asnmts'])))

        return pd.DataFrame(sets)

    def generate_dataset(self): 
        """
        Generate prompts for all symbols in the provided input, incrementally writing
        progress to the provided file in parquet format
        """
        self.dataset = pd.DataFrame()

        self.extract_symbol_defs()
        for index, def_ in tqdm(self.defs.iterrows(), total=len(self.defs)): 
            
            # Extract all refernces, assignments, etc... to this symbol and proceed 
            # only if we've found one or more valid interactions w/ said symbol
            refs = self.extract_symbol_refs(symbol=def_['name'])
            if refs['refs']:
                self.dataset = pd.concat([self.dataset, self.generate_symbol_prompts(def_, refs)])
        
        print(f"Dataset generated ({len(self.dataset)} rows)!")
                
        usage = pd.DataFrame(self.usage).groupby('model').sum().reset_index()
        print(f"Model usage:\n{usage}")

        return self.dataset

def build(input_dirs, openai_key, output_dir): 
    """
    Given a list of directories, build a fine-tuning dataset and emit to the provided
    dataset directory
    
    Based on a conversation I had with ChatGPT about the Linux kernel source tree, 
    the following subsystems seem most relevant to mastery/sensemaking...

    high_value_dirs =  [
        'kernel', 
        'include', 
        'arch/x86', 
        'mm', 
        'fs',
        'init'     
    ]
    
    With these being less, but still, relevant... everything else feels like cruft 
    at this stage
    medium_value_dirs = [
        'drivers', 
        'ipc', 
        'security', 
        'block', 
        'lib' 
    ]
    """
    
    if type(input_dirs) != list or  len(input_dirs) == 0: 
        raise ValueError("Expecting non-empty directory list!")
    
    os.makedirs(output_dir, exist_ok=True)

    prompt_sets = []

    # We can iterate over a variety of input directories here, but the dataset 
    # we build will consider each separately. If we want symbols from one to 
    # be represented in another a common parent directory should be passed. 
    # This implies we should do some housekeeping in the passed directories if
    # we don't want some of the contained code/dirs included in this analysis. 
    for input_dir in tqdm(input_dirs): 

        backends = [
            {'type':'ollama', 'url': "http://localhost:11434/v1", 'model': 'llama3.1:8b'},
            # This guy's a little too slow to participate in the larger campaign... 
            #{'type':'ollama', 'url': "http://10.0.0.37:11435/v1", 'model': 'qwen2.5-coder:3b'},
            {'type':'openai', 'api_key': openai_key,'model': 'gpt-4.1-mini'}
            ]
        
        print("Generating dataset based on {input_dir}...")

        analyzer = CodebaseAnalyzer(input_dir, backends)
        dataset = analyzer.generate_dataset() 

        input_base = os.path.basename(input_dir)
        output_file = os.path.join(output_dir,f"{input_base}.parquet")
        dataset.to_parquet(output_file)
        
        print(f"Wrote dataset for {input_dir} to {output_file}.")
    
    print(f"Generation campaign complete!")

def load_dataset(path): 
    """
    Read our dataset from a file and return it
    """
    dataset = pd.read_parquet(path=path)
    return dataset 

def save_dataset(dataset, path): 
    """
    Write our dataset out to a file
    """
    if type(dataset) == pd.DataFrame: 
        dataset.to_parquet(path=path, index=True)