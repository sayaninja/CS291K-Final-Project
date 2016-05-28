import json
import re
import itertools as it
from collections import Counter
import numpy as np

RESTAURANTS_OUTPUT_FILE = "./dataset/restaurants.json"
REVIEWS_OUTPUT_FILE = "./dataset/reviews.json"
filename = "./dataset/temp.json"

def get_reviews():
    reviews = []
    stars = []
    file = open(filename, "r")
    for line in file:
        review = json.loads(line)
        del review["review_id"]
        del review["business_id"]
        reviews.append(clean_str(review["text"].strip()))
        stars.append(review["stars"] - 1)
    reviews = [s.split(" ") for s in reviews]
    return reviews, stars

def get_restaurants():
    restaurants = []
    file = open(REVIEWS_OUTPUT_FILE, "r")
    for line in file:
        restaurant = json.loads(line)
        restaurants.append(restaurant)
    return restaurants

def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def pad_reviews(reviews, padding_word="<pad/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    review_length = max(len(x) for x in reviews)
    reviews_padded = []
    for i in range(len(reviews)):
        review = reviews[i]
        num_padding = review_length - len(review)
        new_sentence = review + [padding_word] * num_padding
        reviews_padded.append(new_sentence)
    return reviews_padded

def build_vocab(sentences):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    # Build vocabulary
    word_counts = Counter(it.chain(*sentences))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv]

def build_input_data(sentences, labels, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    x = np.array([[vocabulary[word] for word in sentence] for sentence in sentences])
    y = np.array(labels)
    return [x, y]

def load_data():
    reviews, stars = get_reviews()
    reviews_padded = pad_reviews(reviews)
    vocabulary, vocabulary_inv = build_vocab(reviews_padded)
    x, y = build_input_data(reviews_padded, stars, vocabulary)
    return x, y, vocabulary, vocabulary_inv