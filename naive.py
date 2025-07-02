import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import pickle
import string 
import spacy
import nltk
from dataset import CodebaseAnalyzer
from openai import OpenAI

def tokenize(text):
    """
    Split a sequence on word boundaries
    """
    tokens = nltk.word_tokenize(text)
    return tokens 

def clean(tokens): 
    """
    Clean stopwords and punctuation from a sequence of

    Adapted from https://github.com/AIPI540/AIPI540-Deep-Learning-Applications/blob/main/3_nlp/text_preprocessing.ipynb
    """
    stop_words = set(nltk.corpus.stopwords.words('english'))
    punctuations = string.punctuation

    tokens = [w for w in tokens if w.lower() not in stop_words and w not in punctuations]
    return tokens

def lemmatize(tokens): 
    """
    Try to reduce words to their roots and make the task of matching easier for the 
    "algorithm" 
    """
    lemmatizer = nltk.stem.WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word).lower().strip() for word in tokens]

    return tokens 

# We just leave this floating in space to avoid creating the client a thousand times
# each eval. Should migrate to a class, but here we are. 💃
client = OpenAI(api_key="none", base_url="http://localhost:11434/v1")

def embed(text): 
    """
    Generate a text embedding for the input sequence using our trusty ollama 
    nomin embedding model 

    NOTE: snippet from https://platform.openai.com/docs/guides/embeddings
    """    
    response = client.embeddings.create(model="nomic-embed-text:latest", input=text)
    return response.data[0].embedding

def similarity(a, b): 
    """
    Return pairwise similarity between elements in passed arrays
    """
    similarity_matrix = cosine_similarity([embed(a)],[embed(b)])
    pairwise_similarity = np.diag(similarity_matrix)

    return pairwise_similarity

class NaiveEstimator(BaseEstimator): 
    
    def __init__(self):
        """
        Set up an instance of our naive estimator 
        """
        self.model = None
    
    def intersect_symbols(self, symbols, tokens):
        """
        Find kernel symbols that are present in provided text
        """
        matches = set()
        for symbol in symbols: 
            if symbol.lower() in tokens: 
                matches.add(symbol)

        return matches

    def fit(self, X, y): 
        """
        Fit our naive estimator to a set of prompts
        """ 
        self.analyzer = CodebaseAnalyzer('linux/init')
        self.analyzer.extract_symbol_defs()
        
        symbols = set(self.analyzer.defs['name']) 

        self.model = { 
            "symbols": symbols, 
            "mapping" : []
            }
        
        for x, y in zip(X, y):
            ins = self.intersect_symbols(symbols, lemmatize(clean(tokenize(x)))) 
            outs = self.intersect_symbols(symbols, lemmatize(clean(tokenize(y))))
            self.model["mapping"].append((ins,outs))
            break 

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Generate an answer given a prompt/input/question
        """
        preds = []
        for x in X: 
            for ins, outs in self.model["mapping"]: 
                symbols = self.intersect_symbols(self.model["symbols"], x)
                preds.append("".join(outs) if symbols == ins else "")

        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """        
        y_hat = self.predict(X)

        scores = []
        for a, b in zip (y, y_hat): 
            scores.append(similarity(a, b)) 

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
    filename = os.path.join(path, 'naive.pkl')

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

    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f) 
    
    if type(model) != NaiveEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model

def train(dataset, model_dir): 
    """
    'Train' the naive model 
    """
    
    X, y  = load_dataset(dataset)
    model = NaiveEstimator().fit(X, y)
    save_model(model, model_dir)

def test(model_dir, dataset):
    """
    Test the naive model 
    """
    X, y = load_dataset(dataset)
    model = load_model(model_dir)    
    scores = model.score(X, y)
    print("Naive model scores for the provided dataset: ", scores)