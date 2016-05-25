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

with tf.Graph().as_default():
    sess = tf.Session()
    with sess.as_default():
        text_cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=2,
                vocab_size=len(vocabulary),
                embedding_size=128,
                filter_sizes=[3,4,5],
                num_filters=128)
