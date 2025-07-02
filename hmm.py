from tqdm import tqdm
import random 
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import string 
import spacy
import nltk
from naive import tokenize, clean, lemmatize, similarity

class HmmEstimator(BaseEstimator): 
    
    def __init__(self):
        """
        Set up an instance of our hidden markov model estimator 
        """
        self.model = None
        self.nlp = spacy.load("en_core_web_sm")

    def intersect_symbols(symbols, text):
        """
        Find kernel symbols that are present in provided text
        """
        words = text.split()
        matches = set()
        for symbol in symbols: 
            if symbol in words: 
                matches.add(symbol)

        return matches

    def fit(self, X, y): 
        """
        Fit our HMM estimator to supplied inputs/outputs

        NOTE: Help from: 
         - https://www.nltk.org/howto/probability.html
         - https://stackoverflow.com/questions/8941269/initialize-hiddenmarkovmodeltrainer-object
        """ 
        # We'll train our HMM to predict words across the sequences
        seqs = list(X + y)
        
        tqdm.write(f"Building reference data... ")
        data = []
        vocabulary = []
        for seq_in in tqdm(seqs): 
            seq_out = []
            for token in self.nlp(seq_in):                
                if not token.is_stop and not token.is_punct: 
                    vocabulary.append(token.text.lower())
                    seq_out.append((token.text.lower(), token.tag_))
            if len(seq_out) > 1: 
                data.append(seq_out)

        all_tags = self.nlp.get_pipe("tagger").labels

        tqdm.write(f"Fitting HMM model...")
        
        # Symbols int he HMM parlance is a set of our vocabulary 
        # States are the number of internal states which we can't 
        # see... hence the 'hidden' bit
        trainer = nltk.hmm.HiddenMarkovModelTrainer(
            all_tags, 
            symbols=nltk.unique_list(vocabulary))
        self.model = trainer.train_supervised(data)

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Generate an answer given a prompt/input/question
        """
        preds = []
        tqdm.write(f"Running predictions...")
        for x in tqdm(list(X)): 
            state = self.model.best_path(lemmatize(clean(tokenize(x))))
            tokens_n_tags = self.model.random_sample(random.Random(), 50)
            tokens = [a[0] for a in tokens_n_tags]
            preds.append(" ".join(tokens))
        
        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        tqdm.write(f"Scoring predictions...")
        for a, b in tqdm(zip(y, y_hat), total=len(y)): 
            scores.append(similarity(a, b)) 

        return scores
    
    def get_state(self): 
        model_data = {
            # "symbols": self.model._symbols,
            # "states": self.model._states,
            # "transitions": self.model._transitions,
            # "outputs": self.model._outputs,
            # "priors": self.model._priors,
        }
        return model_data
    
    def put_state(self, model_data):
        self.model = nltk.hmm.HiddenMarkovModelTagger(**model_data)

def load_dataset(file): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    df = pd.read_parquet(file)
    return df['x'], df['y']

def save_model(model, path):
    """
    Save the model to a file
    NOTE: needed gpt-4os help to figure out how to save the model since
    it's keeping someinternal lambda funcs we can't persist, see: 
    https://chatgpt.com/share/68658ba9-d938-8013-84d2-0d49cbdf5052
    """    
    filename = os.path.join(path, 'hmm.pkl')

    with open(filename, 'wb') as f:         
        state = model.get_state()
        pickle.dump(state, f)
    
    print (f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    NOTE: copy/pasta from vision project 
    """
    model = None

    filename = os.path.join(path, 'hmm.pkl')
    with open(filename, 'rb') as f: 
        model_data = pickle.load(f) 
        model = HmmEstimator()
        model.put_state(model_data)
    
    if type(model) != HmmEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(dataset, model_dir): 
    """
    'Train' the hmm model 
    """
    
    X, y  = load_dataset(dataset)
    model = HmmEstimator().fit(X, y)
    
    # NOTE: the NLTK code for the estimator is getting crusty, I can't figure out 
    # how to save the model... neither it nor it's critical properties are picklable, 
    # so essentially need to run testing concurrently (can't be separate CLI calls)
    #                                                  
    #save_model(model, model_dir)

    test(model, "data/ipc/ipc.parquet")

def test(model, dataset):
    """
    Test the hmm model 
    """
    X, y = load_dataset(dataset)
    # See comment in train() above
    # model = load_model(model_dir)    
    scores = model.score(X, y)
    tqdm.write(f"Hidden Markov model mean scores for the provided dataset: {np.mean(scores)}")