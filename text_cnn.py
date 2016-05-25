import tensorflow as tf
import numpy as np

class TextCNN(object):
    def __init__(self, sequence_length, num_classes, vocab_size,
                 embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Initialize placeholders
        self.x = tf.placeholder(tf.int32, [None, sequence_length], name="x")
        self.y = tf.placeholder(tf.float32, [None, num_classes], name="y")
        self.dropout = tf.placeholder(tf.float32, name="dropout")
        print "num_filters: ", num_filters
        # Initialize of L2 regularization
        l2_loss = tf.constant(0.0)

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # Randomly initialize W2 from uniform distribution
            W = tf.Variable(tf.random_uniform([vocab_size, embedding_size],
                                              -1.0, 1.0), name="W")
            self.embedded_chars = tf.nn.embedding_lookup(W, self.x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Forward pass through hidden layers
        outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("foward_pass-%s" % filter_size):
                # Convolution layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                print filter_shape
                # Initialize W2 ad b2
                W = tf.Variable(tf.truncated_normal(filter_shape,
                                                    stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]),
                                name="b")
                conv = tf.nn.conv2d(self.embedded_chars_expanded,
                                    W,
                                    strides=[1, 1, 1, 1],
                                    padding="VALID",
                                    name="conv")

                # ReLU activation layer
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")

                # Maxpooling layer
                ksize = [1, sequence_length - filter_size + 1, 1, 1]
                pool = tf.nn.max_pool(h,
                                      ksize=ksize,
                                      strides=[1, 1, 1, 1],
                                      padding="VALID",
                                      name="pool"
                                      )
                outputs.append(pool)

            # Fully connected layer 1
            # num_filters_total = num_filters * len(filter_sizes)
            # self.h_pool = tf.concat(3, outputs)
            # self.h_pool_flat = tf.reshape(self.h_pool[-1, num_filters_total])

            # # Perform dropout
            # with tf.name_scope("dropout"):
            #     self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout)

            # # Fully connected output layer 2
            # with tf.name_scope("output"):
            #     # Initialize W3
            #     W = tf.get_variable("W",
            #                         shape=[num_filters_total, num_classes],
            #                         initializer=tf.contrib.layers.xavier_initializer())
            #     b = tf.Variable(tf.constant(0.1))
            #
            #     # L2 loss
            #     l2_loss += tf.nn.l2_loss(W)
            #     l2_loss += tf.nn.l2_loss(b)
            #     self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            #     self.predictions = tf.argmax(self.scores, 1, name="predictions")
            #
            # # Calculate cross-entropy loss
            # with tf.name_scope("loss"):
            #     losses = tf.nn.softmax_cross_entropy_with_logits(self.scores,
            #                                                      self.y)
            #     self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss
            #
            # # Accuracy
            # with tf.name_scope("accuracy"):
            #     correct_predictions = tf.equal(self.predictions,
            #                                    tf.argmax(self.input_y, 1))
            #     self.accuracy = tf.reduce_mean(tf.cast(correct_predictions,
            #                                            "float"),
            #                                    name="accuracy")
