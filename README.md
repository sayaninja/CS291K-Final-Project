Check out the GitHub repository here:
https://github.com/zhansaya19/CS291K-Final-Project

# Yelp Review Classification
Yelp has an ongoing data mining challenge where they have provided datasets on businesses, reviews, users, check-ins, and tips. We are only interested in features of the first two datasets for now: businesses and user reviews. Each dataset is nicely formatted in JSON, rich in features, and composed of thousands of records. In this project, we developed a convolutional neural network to predict star ratings of individual text reviews, as well as overall restaurant ratings.

The features from the business dataset that we are interested in include:
* Unique business ID code
* Attributes: We will only use records that have the “restaurant” attribute
* Overall star rating: From 1 to 5, in __half__-star increments

The features from the reviews dataset that we are interested in include:
* Business ID code with which this review is associated
* Rating: 1 to 5 stars, in __whole__-star increments
* Text content of the review

## Prerequisites

To run this neural network, you will need the following:

1. Python 2
2. Numpy
3. TensorFlow
4. Yelp Data from Jeffrey's Dropbox: 
   * https://www.dropbox.com/sh/agrb46jkr87q9t4/AADcm5XJdhEgiwRgg-2Ti1vPa?dl=0
   * Note: Create a folder at the project root called `dataset` and put these files in there.

## How to Run

As discussed in our presentation and report, our project consists of three experiments outlined here:

1. Experiment A: Train on individual reviews; predict individual review ratings.
2. Experiment B: Train on individual reviews; average individual predicted ratings to form a single predicted overall rating.
3. Experiment C: For each restaurant, combine all of its reviews into one super-review and train on that; predict overall rating.

Each experiment is in its own git branch: experiment_a, experiment_b, and experiment_c. To run, simply checkout the desired branch and type `python train.py` in a command prompt. Importing and preprocessing the full dataset of 2.2 million reviews takes 40 minutes, so we created a subset of 20,000 reviews in the `temp.json` file for development purposes. This is the default data file used in the `data_utils.py` script.

## Project Files

Here is a description of the main files in this repository.

* `data_utils.py` - Contains functions for importing and preprocessing the JSON data files.
* `text_cnn.py` - The convolutional neural network class.
* `train.py` - The main function for running the neural network.
* `Presentation.pptx` - The PowerPoint of our presentation. Made in Google Slides.
* `Report.pdf` - Paper report of our project.
