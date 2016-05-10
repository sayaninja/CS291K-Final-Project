###Project Proposal###

__Problem Definition__

Enormous amounts of data are generated and collected in all shapes and sizes. User-submitted text reviews are among the most difficult to interpret due to hidden contexts and intentions. Any single given review may contain a number of opinions on various aspects of a product or service. Programmatically extracting the overall sentiment of a review would allow each piece to be more easily quantified in an efficient manner without the need of human examination. This will be incredibly helpful to consumers who seek to find the best product or service with minimal searching on their own part. Currently, overall product ratings are a simple average of all of the user-submitted ratings. By developing an algorithm that can determine the underlying attitude of a text review and weigh it against other factors such as location, overall ratings can become a more accurate representation than a simple average.
Dataset Description
Yelp is currently holding a data mining challenge where they have provided datasets on businesses, reviews, users, check-ins, and tips. We are only interested in features of the first two datasets for now: businesses and user reviews. Each dataset is nicely formatted in JSON, rich in features, and composed of thousands of records. This saves us the trouble of having to do extensive cleaning and sorting, allowing more time to work on the algorithm instead.

The features from the business dataset that we are interested in include:
* Unique business ID code
* Attributes: We will only use records that have the “restaurant” attribute
* Location: city, state, as well as longitudinal and latitudinal coordinates
* Overall star rating: From 1 to 5, in half-star increments

The features from the reviews dataset that we are interested in include:
* Business ID code with which this review is associated
* Rating given by this review
* Text content of the review: tokenize into a bag of words
* Votes: the number of users that found this review to be useful, funny, or cool


__Task Description__

The ultimate goal of this project is to build a model that predicts the overall rating for a restaurant given a set of features. Initially, we will predict if the restaurant has positive or negative rating overall using only the review text. A positive rating is anything from 3 to 5 stars, and a negative rating will be anything from 1 to 2.5 stars. Then the next step would be to incorporate additional features such as location and usefulness votes on individual reviews as weights in predicting an overall rating. Finally, we will have our model predict a specific star rating and compare it to the true rating. This would be a better measurement of the precision of our model. In analyzing each review to train the model, the first step would be to tokenize the text into a bag of words. A number of statistical strategies can be used, such as Bayesian networks and maximum a posteriori estimation, to determine if the words collectively produce a positive or negative review. Features such as location and ratings of the reviews will be used as the ground truth in calculating loss and measuring performance. So this will be a supervised learning task.

__Plan Outline__

While there has been a lot of past work in this type of prediction, not many have employed Neural Networks. First, we must research mathematical and statistical strategies that have been used before in interpreting text. We are planning to implement our model using a Recurrent Neural Network (RNN) algorithm, but since we are not entirely certain, we might use a combination of different algorithms to try and achieve better performance. Our programming language of choice will be Python combined with the TensorFlow library.
