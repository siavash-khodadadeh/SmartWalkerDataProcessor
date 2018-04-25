import tensorflow as tf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tensorflow.contrib.metrics.python.metrics.classification import accuracy
from tensorflow.python.training.adam import AdamOptimizer

from simple_fully_connected.datasets import Dataset
from simple_fully_connected.preprocessor import get_data_and_labels

from simple_fully_connected.settings import DATA_ADDRESS, PREPROCESSED_FILE_ADDRESS


font = {'family': 'normal',
        'weight': 'normal',
        'size': 22}

matplotlib.rc('font', **font)


INPUT_SIZE = 4
WIDTH = 25
N_TRAIN_SAMPLE = 1200
N_ITERATION = 5000
LOG_DIRECTORY = './tf_logs/4/'


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_precision_and_recall(y, labels):
    # True positive
    tp = np.sum(y * labels)
    # False positive
    fp = np.sum(y * (1 - labels))
    # True negative
    tn = np.sum((1 - y) * (1 - labels))
    # False negative
    fn = np.sum((1 - y) * labels)
    if tp + fp != 0:
        precision = tp / (tp + fp)
    else:
        precision = 0

    if tp + fn != 0:
        recall = tp / (tp + fn)
    else:
        recall = 0
    return precision, recall


def precision_recall_curve(logits, labels):
    precisions = []
    recalls = []

    for threshold in [i/100. for i in range(0, 101, 5)]:
        y = np.apply_along_axis(softmax, axis=1, arr=logits)[:, 1]
        y[np.where(y >= threshold)] = 1
        y[np.where(y < threshold)] = 0
        y = y.reshape(-1, 1)

        precision, recall = get_precision_and_recall(y, labels)

        precisions.append(precision)
        recalls.append(recall)

    plt.xlabel('Recall')
    plt.ylabel('Precision')

    plt.plot(recalls, precisions, color='black')
    plt.show()


train_data, train_labels, validation_data, validation_labels = get_data_and_labels(
    PREPROCESSED_FILE_ADDRESS,
    step_size=1,
    width=WIDTH,
    # data_address=DATA_ADDRESS,
    data_address=None,
)


train_data = train_data.reshape(-1, INPUT_SIZE * WIDTH)
validation_data = validation_data.reshape(-1, INPUT_SIZE * WIDTH)

train_dataset = Dataset(train_data, train_labels)
validation_dataset = Dataset(validation_data, validation_labels, shuffle=False)


x = tf.placeholder(dtype=tf.float32, shape=(None, INPUT_SIZE * WIDTH))
y = tf.placeholder(dtype=tf.float32, shape=(None, 1))
y_one_hot = tf.reshape(tf.one_hot(tf.cast(y, tf.int32), depth=2), shape=(-1, 2))


with tf.variable_scope('network'):
    layer1 = tf.layers.dense(x, units=100, activation=tf.nn.relu, name='layer1')
    layer2 = tf.layers.dense(layer1, units=50, activation=tf.nn.relu, name='layer2')
    out = tf.layers.dense(layer2, units=2, name='out')


with tf.variable_scope('loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_one_hot, logits=out, name='loss'))
    loss_summary = tf.summary.scalar('loss', loss)

with tf.variable_scope('accuracy'):
    predictions = tf.reshape(tf.argmax(out, axis=1), (-1, 1))
    tf_accuracy = accuracy(tf.cast(predictions, tf.int32), tf.cast(y, tf.int32))
    accuracy_summary = tf.summary.scalar('accuracy', tf_accuracy)

    baseline_accuracy = accuracy(tf.cast(tf.zeros_like(out), tf.int32), tf.cast(y, tf.int32))
    tf.summary.scalar('base line', baseline_accuracy)


optimizer = AdamOptimizer(learning_rate=0.001)
train_op = optimizer.minimize(loss)
merge_op = tf.summary.merge_all()
init_op = tf.initialize_all_variables()
init_global_op = tf.global_variables_initializer()
init_local_op = tf.initialize_local_variables()

sess = tf.Session()

train_writer = tf.summary.FileWriter(LOG_DIRECTORY + 'train/', sess.graph)
train_writer.add_graph(sess.graph)
test_writer = tf.summary.FileWriter(LOG_DIRECTORY + 'test/')
sess.run(init_op)
sess.run(init_local_op)
sess.run(init_global_op)


for it in range(N_ITERATION + 1):
    x_batch, y_batch = train_dataset.next_batch()
    sess.run(train_op, feed_dict={x: x_batch, y: y_batch})

    if it % 100 == 0:
        merged_summary = sess.run(merge_op, feed_dict={x: x_batch, y: y_batch})
        train_writer.add_summary(merged_summary, it)
        print(it)

    if it % 1000 == 0:
        x_valid_batch, y_valid_batch = validation_dataset.next_batch(len(validation_dataset))
        merged_validation_summary = sess.run(merge_op, feed_dict={x: x_valid_batch, y: y_valid_batch})
        test_writer.add_summary(merged_validation_summary, it)

        y_output = sess.run(out, feed_dict={x: x_valid_batch, y: y_valid_batch})
        # precision_recall_curve(y_output, y_valid_batch)


train_writer.close()
