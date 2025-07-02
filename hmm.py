import random 
import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import string 
import spacy
from nltk.hmm import HiddenMarkovModelTrainer
from naive import tokenize, clean, lemmatize

class HmmEstimator(BaseEstimator): 
    
    def __init__(self):
        """
        Set up an instance of our hidden markov model estimator 
        """
        self.model = None

    def similarity(a, b): 
        """
        Return pairwise similarity between elements in passed arrays
        """
        similarity_matrix = cosine_similarity(a,b)
        pairwise_similarity = np.diag(similarity_matrix)

        return pairwise_similarity
    
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
        """ 
        # We'll train our HMM to predict words across the sequences
        seqs = list(df['x'] + df['y'])
        
        data = []
        for seq in seqs: 
           data.append(lemmatize(clean(tokenize(seq))))

        self.model = nltk.hmm.HiddenMarkovModelTrainer().train(unlabeled_sequences=data)

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Generate an answer given a prompt/input/question
        """
        for x in X: 
            state = self.model.best_path(lemmatize(clean(tokenize(x))))
            self.model.random_sample(random.Random(), 100)
        
        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        for a, b in zip (y, y_hat): 
            scores.append(self.similarity(a, b)) 

    # Evaluate the tagger on the test data
    test_accuracy = hmm_tagger.evaluate(test_data)

    print(f"Test accuracy: {test_accuracy:.2f}")
    return test_accuracy

        return scores

def load_dataset(file): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    df = pd.read_parquet(file)
    return df['x'], df['y']

def save_model(model:NaiveEstimator, path):
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
    
    if type(model) != NaiveEstimator: 
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
    X, _ = load_dataset(dataset)
    model = load_model(model_dir)    
    score = model.score(X)