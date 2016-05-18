import json
import re

RESTAURANTS_OUTPUT_FILE = "./dataset/restaurants.json"
REVIEWS_OUTPUT_FILE = "./dataset/reviews.json"

def get_reviews():
    reviews = []
    stars = []
    file = open(REVIEWS_OUTPUT_FILE, "r")
    for line in file:
        review = json.loads(line)
        del review["review_id"]
        del review["business_id"]
        reviews.append(clean_str(review["text"]))
        stars.append(review["stars"])
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