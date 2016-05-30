import json
import re
import itertools as it
from collections import Counter
import numpy as np

RESTAURANTS_OUTPUT_FILE = "./dataset/restaurants.json"
REVIEWS_OUTPUT_FILE = "./dataset/reviews.json"
filename = "./dataset/temp.json"

def get_business_ids():
    '''
    Scans through data file for all business IDs
    '''
    ids = []
    file = open(filename, "r")
    for line in file:
        review = json.loads(line)
        ids.append(review["business_id"])
    file.close()

    # Remove duplicates
    ids = list(set(ids))
    return np.array(ids)

def get_reviews_and_stars(ids):
    '''
    Returns all reviews corresponding ratings for one restaurant
    '''
    reviews = []
    stars = []
    file = open(filename, "r")
    for line in file:
        review = json.loads(line)
        if review["business_id"] in ids:
            del review["review_id"]
            # del review["business_id"]
            reviews.append(clean_str(review["text"].strip()))
            stars.append(review["stars"] - 1)  # Make 0-4
    file.close()
    reviews = [s.split(" ") for s in reviews]
    stars = np.array(stars)
    return reviews, stars

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

def pad_reviews(reviews, review_length, padding_word="<pad/>"):
    """
    Pads all sentences to the same length. The length is defined by the longest sentence.
    Returns padded sentences.
    """
    reviews_padded = []
    for i in range(len(reviews)):
        review = reviews[i]
        num_padding = review_length - len(review)
        new_sentence = review + [padding_word] * num_padding
        reviews_padded.append(new_sentence)
    return reviews_padded

def build_vocab():
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
    reviews = []
    file = open(filename, "r")
    for line in file:
        review = json.loads(line)
        del review["review_id"]
        reviews.append(clean_str(review["text"].strip()))
    file.close()
    reviews = [s.split(" ") for s in reviews]
    review_length = max(len(x) for x in reviews)

    reviews_padded = pad_reviews(reviews, review_length)
    # Build vocabulary
    word_counts = Counter(it.chain(*reviews_padded))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    vocabulary_inv = list(sorted(vocabulary_inv))
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return [vocabulary, vocabulary_inv, review_length]

def build_input_data(reviews_padded, vocabulary):
    """
    Maps sentences and labels to vectors based on a vocabulary.
    """
    reviews = np.array([[vocabulary[word] for word in review] for review in reviews_padded])
    return reviews

def round_rating(rating):
    '''
    Yelp's overall business ratings are in increments of 0.5 stars.
    This returns a rating rounded to the nearest half star.
    '''
    return round(rating / 0.5) * 0.5