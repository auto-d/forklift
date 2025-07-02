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
        for seq in tqdm(seqs): 
            for token in self.nlp(seq):
                datum = []
                if not token.is_stop and not token.is_punct: 
                    vocabulary.append(token.text.lower())
                    datum.append([token.text.lower(), token.tag_])
                if len(datum) > 0: 
                    data.append(datum)

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
            preds.append(self.model.random_sample(random.Random(), 100))
        
        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        tqdm.write(f"Scoring predictions...")
        for a, b in tqdm(zip(y, y_hat), total=len(y)): 
            scores.append(self.similarity(a, b)) 

        return scores

def load_dataset(file): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    df = pd.read_parquet(file)
    return df['x'], df['y']

def save_model(model, path):
    """
    Save the model to a file
    NOTE: copy/pasta from vision project 
    """    
    filename = os.path.join(path, 'hmm.pkl')
    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
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
        model = pickle.load(f) 
    
    if type(model) != HmmEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(dataset, model_dir): 
    """
    'Train' the hmm model 
    """
    
    X, y  = load_dataset(dataset)
    model = HmmEstimator().fit(X, y)
    save_model(model, model_dir)

def test(model_dir, dataset):
    """
    Test the hmm model 
    """
    X, y = load_dataset(dataset)
    model = load_model(model_dir)    
    scores = model.score(X, y)
    tqdm.write(f"Hidden Markov model mean scores for the provided dataset: {np.mean(scores)}")