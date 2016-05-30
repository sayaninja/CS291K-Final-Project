from data_utils import *
from text_cnn import *

print("Importing data...")
x, y, vocabulary, vocabulary_inv = load_data_based_ids()

print("Done importing.")
# Hyper parameters
num_validation = 1000
num_test = 1000
num_classes = 11
num_epochs = 100
batch_size = 5
embedding_size = 128
filter_sizes = [3, 4, 5]
num_filters = 128
adam_opt = 1e-3
keep_prob = 0.5
l2_reg_lambda = 0.5

# Get data

x_train = x
y_train = y
# x_train, x_val, x_test = x_shuffled[:-(num_validation + num_test)], \
#                          x_shuffled[-(num_validation + num_test): -num_test], \
#                          x_shuffled[-num_test:]
# y_train, y_val, y_test = y_shuffled[:-(num_validation + num_test)], \
#                          y_shuffled[-(num_validation + num_test): -num_test], \
#                          y_shuffled[-num_test:]
#
print("Vocabulary Size: {:d}".format(len(vocabulary)))
# print("Train/Dev/Test split: {:d}/{:d}/{:d}".format(len(y_train), num_validation, num_test))
#
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

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(adam_opt)
        grads_and_vars = optimizer.compute_gradients(text_cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.initialize_all_variables())


        def train_step(x_batch, y_batch):
            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: y_batch,
                text_cnn.dropout: keep_prob
            }
            _, step, loss, accuracy = sess.run(
                    [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
                    feed_dict=feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))


#         def test_step(x_batch, y_batch):
#             feed_dict = {
#                 text_cnn.review_placeholder: x_batch,
#                 text_cnn.stars_placeholder: y_batch,
#                 text_cnn.dropout: 1
#             }
#             _, step, loss, accuracy = sess.run(
#                     [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
#                     feed_dict=feed_dict)
#             print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))
#
#
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
#         print "\nValidation Data Evaluation..."
#         step = 1
#         for step in range(num_validation/batch_size):
#             offset = (step * batch_size) % (y_val.shape[0] - batch_size)
#
#             # Generate a minibatch
#             x_batch = x_val[offset:(offset + batch_size), :]
#             y_batch = y_val[offset:(offset + batch_size)]
#
#             batch_labels_one_hot = np.zeros((batch_size, num_classes))
#             batch_labels_one_hot[np.arange(batch_size), y_batch] = 1
#             test_step(x_batch, batch_labels_one_hot)
#
#         # Evaluate test set
#         print "\nTest Data Evaluation..."
#         step = 1
#         for step in range(num_test/batch_size):
#             offset = (step * batch_size) % (y_val.shape[0] - batch_size)
#
#             # Generate a minibatch
#             x_batch = x_test[offset:(offset + batch_size), :]
#             y_batch = y_test[offset:(offset + batch_size)]
#
#             batch_labels_one_hot = np.zeros((batch_size, num_classes))
#             batch_labels_one_hot[np.arange(batch_size), y_batch] = 1
#             test_step(x_batch, batch_labels_one_hot)