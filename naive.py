import os
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator 
import matplotlib.pyplot as plt
import pickle

class NaiveEstimator(BaseEstimator): 

    def __init__(self, bins=256):
        """
        Set up an instance of our naive histogram-based estimator. 
        """
    
    def fit(self, X, y=None): 
        """
        Fit our naive estimator to a training set (of images) 
        """
      

        return self

    def predict(self, X) -> np.ndarray: 
        """
        Predict classes for a set of inputs (images) based on a prior fit
        """
        preds = np.zeros(len(X))

        return preds 
    
    def score(self, X, y):
        """
        Sklearn expectation for CV scoring 
        """
        return np.mean(self.predict(X) == y)

def load_dataset(annotations, image_dir): 
    """
    Load and return a compatible dataset for the naive classifier
    """
    X = []
    y = list(annotations.label)

    return X, y

def save_model(model:NaiveEstimator, path):
    """
    Save the model to a file
    """    
    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'wb') as f: 
        pickle.dump(model, f)
    
    print (f"Model saved to {path}")

    return filename

def load_model(path): 
    """
    Load our naive model off disk
    """
    model = None

    filename = os.path.join(path, 'naive.pkl')
    with open(filename, 'rb') as f: 
        model = pickle.load(f) 
    
    if type(model) != NaiveEstimator: 
        raise ValueError(f"Unexpected type {type(model)} found in {filename}")

    return model


def train(dataset, dir): 
    pass

def test(dataset):
    pass