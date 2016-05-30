import json
import re
import itertools as it
from collections import Counter
import numpy as np

RESTAURANTS_OUTPUT_FILE = "../dataset/restaurants.json"
REVIEWS_OUTPUT_FILE = "../dataset/reviews.json"
reviews_filename = "../dataset/temp5reviews.json"
rests_filename = "../dataset/temp2rests.json"

def get_data():
    reviews = []
    stars = []
    ids  = []
    file = open(reviews_filename, "r")
    for line in file:
        review = json.loads(line)
        del review["review_id"]
        reviews.append(clean_str(review["text"].strip()))
        stars.append(review["stars"] - 1)
        ids.append(review["business_id"])
    reviews = [s.split(" ") for s in reviews]
    print reviews[0]
    return reviews, stars, ids

def get_reviews_data():
    reviews_data = {}
    file = open('../dataset/temp.json', "r")
    for line in file:
        review = json.loads(line)
        del review["review_id"]
        del review["stars"]
        review["text"] = clean_str(review["text"].strip())
        if review["business_id"] in reviews_data.keys():
            reviews_data[review["business_id"]] = reviews_data[review["business_id"]] + review["text"].split(" ")
        else:
            reviews_data[review["business_id"]] = list()
            reviews_data[review["business_id"]] = review["text"].split(" ")
    return reviews_data

def get_restaurants_data():
    stars_converter = {0: 0, 0.5: 1, 1: 2,
                       1.5: 3, 2: 4, 2.5: 5,
                       3: 6, 3.5: 7, 4: 8,
                       4.5: 9, 5: 10}
    stars = {}
    file = open(rests_filename, "r")
    for line in file:
        restaurant = json.loads(line)
        del restaurant["review_count"]
        stars[restaurant["business_id"]] = stars_converter[restaurant["stars"]]
    return stars

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

def build_input_data(reviews_padded, stars, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    reviews = np.array([[vocabulary[word] for word in review] for review in reviews_padded])
    stars = np.array(stars)
    return [reviews, stars]

def load_data():
    reviews, stars, ids = get_data()
    reviews_padded = pad_reviews(reviews)
    vocabulary, vocabulary_inv = build_vocab(reviews_padded)
    reviews, stars, ids = build_input_data(reviews_padded, stars, ids, vocabulary)
    return reviews, stars, ids, vocabulary, vocabulary_inv

def load_data_based_ids():
    reviews_based_ids = get_reviews_data()
    rests = get_restaurants_data()
    reviews = []
    stars = []
    for key, value in reviews_based_ids.iteritems():
        if key in rests.keys():
            reviews.append(value)
            stars.append(rests[key])

    reviews_padded = pad_reviews(reviews)
    vocabulary, vocabulary_inv = build_vocab(reviews_padded)
    reviews, stars = build_input_data(reviews_padded, stars, vocabulary)
    return reviews, stars, vocabulary, vocabulary_inv