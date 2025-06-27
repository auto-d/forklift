import nltk 

def train(dataset, dir): 
    
    # Load the Penn Treebank dataset
    corpus = nltk.corpus.treebank.tagged_sents()

    # Split the dataset into training and test sets
    train_data = corpus[:3000]
    test_data = corpus[3000:]

    # Train an HMM POS tagger
    hmm_tagger = nltk.hmm.HiddenMarkovModelTrainer().train_supervised(train_data)

def test(dataset):

    # Evaluate the tagger on the test data
    test_accuracy = hmm_tagger.evaluate(test_data)

    print(f"Test accuracy: {test_accuracy:.2f}")
    return test_accuracy