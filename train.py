from data_utils import *
from text_cnn import *
import time
from datetime import datetime

# Hyper parameters
num_validation = 1000 # 1000
num_test = 100 # 1000
num_classes = 5
num_epochs = 1000 # 5000
batch_size = 100 # 150
embedding_size = 128
filter_sizes = [3, 4, 5]
num_filters = 128
learning_rate = 1e-3
keep_prob = 0.5 # 0.5
l2_reg_lambda = 0.5

print("Importing data...")
# Get all business IDs
start_time = time.time()
business_ids = get_business_ids()
print("Found " + str(len(business_ids)) + " restaurants.")

# Subset businesses
shuffle_indices = np.random.permutation(np.arange(len(business_ids)))
business_ids = business_ids[shuffle_indices]
train_ids, val_ids, test_ids = \
    business_ids[:-(num_validation + num_test)], \
    business_ids[-(num_validation + num_test): -num_test], \
    business_ids[-num_test:]

# Get reviews from file
x_train, y_train = get_reviews_and_stars(train_ids)
x_val, y_val = get_reviews_and_stars(val_ids)
duration = time.time() - start_time

print("Importing finished in {:.2f} seconds".format(duration))
print
print("Preprocessing data...")
start_time = time.time()

# Build vocabulary
vocabulary, vocabulary_inv, review_length = build_vocab()

# Pad training and validation reviews
x_train_padded = pad_reviews(x_train, review_length)
x_val_padded = pad_reviews(x_val, review_length)
x_train = build_input_data(x_train_padded, vocabulary)
x_val = build_input_data(x_val_padded, vocabulary)

duration = time.time() - start_time

print("Prepocessing finished in {:.2f} seconds".format(duration))
print
print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Val/Test split: {:d}/{:d}/{:d}".format(
    len(y_train), num_validation, num_test))

# Build graph
with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        text_cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                embedding_size=embedding_size,
                filter_sizes=filter_sizes,
                num_filters=num_filters,
                l2_reg_lambda=l2_reg_lambda)

        # Setup training op
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(learning_rate)
        grads_and_vars = optimizer.compute_gradients(text_cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            start_time = time.time()
            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: y_batch,
                text_cnn.dropout: keep_prob
            }
            _, step, loss, accuracy = sess.run(
                    [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
                    feed_dict=feed_dict)
            if step % 10 == 0:
                duration = time.time() - start_time
                print("Step {}, loss: {:g}, acc: {:g}, dur: {:.2f} sec".format(
                    step, loss, accuracy, duration))

        def val_step(x_batch, y_batch):
            start_time = time.time()
            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: y_batch,
                text_cnn.dropout: 1
            }
            _, step, loss, accuracy = sess.run(
                    [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
                    feed_dict=feed_dict)
            duration = time.time() - start_time
            print("Step {}, loss: {:g}, acc: {:g}, dur: {:.2f} sec".format(
                step, loss, accuracy, duration))


        # Start the training loop
        print "\nTraining..."
        step = 1
        for step in range(num_epochs):
            offset = (step * batch_size) % (y_train.shape[0] - batch_size)

            # Generate a minibatch
            x_batch = x_train[offset:(offset + batch_size), :]
            y_batch = y_train[offset:(offset + batch_size)]

            batch_labels_one_hot = np.zeros((batch_size, num_classes))
            batch_labels_one_hot[np.arange(batch_size), y_batch] = 1
            train_step(x_batch, batch_labels_one_hot)

        # Evaluate validation set
        print "\nValidation Data Evaluation..."
        step = 1
        for step in range(num_validation/batch_size):
            offset = (step * batch_size) % (y_val.shape[0] - batch_size)

            # Generate a minibatch
            x_batch = x_val[offset:(offset + batch_size), :]
            y_batch = y_val[offset:(offset + batch_size)]

            batch_labels_one_hot = np.zeros((batch_size, num_classes))
            batch_labels_one_hot[np.arange(batch_size), y_batch] = 1
            val_step(x_batch, batch_labels_one_hot)

        # Evaluate test set
        print "\nTest Data Evaluation..."
        step = 1
        correct = 0
        differences = []
        start_time = time.time()
        for restaurant in test_ids:
            # Import reviews from file and preprocess
            x_test, y_test = get_reviews_and_stars(restaurant)
            x_test_padded = pad_reviews(x_test, review_length)
            x_batch = build_input_data(x_test_padded, vocabulary)

            # Build labels
            batch_labels_one_hot = np.zeros((len(x_test), num_classes))
            batch_labels_one_hot[np.arange(len(y_test)), y_test] = 1

            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: batch_labels_one_hot,
                text_cnn.dropout: 1
            }
            # Predict individual ratings of each review
            batch_predictions = sess.run(text_cnn.predictions,
                                         feed_dict=feed_dict)

            # Predict overall rating by averaging individual rating predictions
            overall_prediction = np.mean(batch_predictions)

            # Get true overall rating for this restaurant
            true_overall = np.mean(y_test)

            # Round ratings to nearest 0.5 star
            overall_prediction = round_rating(overall_prediction)
            true_overall = round_rating(true_overall)

            differences.append(np.absolute(true_overall - overall_prediction))

            if overall_prediction == true_overall:
                correct += 1

            print("Step {}, predicted: {}, truth: {}".format(
                step, overall_prediction, true_overall))
            step += 1

        # Take averages of the star predictions (average residual)
        duration = time.time() - start_time
        accuracy = (float(correct) / float(num_test)) * 100
        print
        print("Testing completed in {:.2f} seconds".format(duration))
        print("Average residual: " + str(np.mean(differences)))
        print("Accuracy: {:.2f}%".format(accuracy))