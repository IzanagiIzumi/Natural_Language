import nltk
from nltk.corpus import movie_reviews
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize

nltk.download('movie_reviews')
nltk.download('punkt')

documents = [(list(movie_reviews.words(fileid)), category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]
import random
random.shuffle(documents)

def extract_features(words):
    return {word: True for word in words}

feature_sets = [(extract_features(d), c) for (d, c) in documents]

train_set, test_set = feature_sets[:1900], feature_sets[1900:]

classifier = NaiveBayesClassifier.train(train_set)

print("Accuracy:", nltk.classify.accuracy(classifier, test_set))

classifier.show_most_informative_features(10)
