from data_utils import *
from text_cnn import *

print("Importing data...")
x, y, vocabulary, vocabulary_inv = load_data()
print("Done importing.")

shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]

x_train, x_dev = x_shuffled[:-1000], x_shuffled[-1000:]
y_train, y_dev = y_shuffled[:-1000], y_shuffled[-1000:]

print("Vocabulary Size: {:d}".format(len(vocabulary)))
print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

# Hyper parameters
num_classes=5
num_epochs = 200
batch_size = 50

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        text_cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=num_classes,
                vocab_size=len(vocabulary),
                embedding_size=128,
                filter_sizes=[3, 4, 5],
                num_filters=128)

        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        grads_and_vars = optimizer.compute_gradients(text_cnn.loss)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.initialize_all_variables())

        def train_step(x_batch, y_batch):
            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: y_batch,
                text_cnn.dropout: 0.5
            }
            _, step, loss, accuracy = sess.run(
                    [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
                    feed_dict=feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        def test_step(x_batch, y_batch):
            feed_dict = {
                text_cnn.review_placeholder: x_batch,
                text_cnn.stars_placeholder: y_batch,
                text_cnn.dropout: 1
            }
            _, step, loss, accuracy = sess.run(
                    [train_op, global_step, text_cnn.loss, text_cnn.accuracy],
                    feed_dict=feed_dict)
            print("step {}, loss {:g}, acc {:g}".format(step, loss, accuracy))

        step = 1
        # Start the training loop
        print "\nTraining..."
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
        # TODO

        # Evaluate test set
        print "\nTest Data Evaluation..."
        # TODO
