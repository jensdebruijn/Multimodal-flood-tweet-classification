import numpy as np
import pandas as pd
import json
import os
from collections import defaultdict
import tensorflow as tf
from sklearn.metrics import precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight


def gen_batch(data, batch_size, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = list(data)
    data = np.array(data)
    data_size = data.shape[0]
    if shuffle:
        shuffle_indices = np.random.permutation(np.arange(data_size))
        shuffled_data = data[shuffle_indices]
    else:
        shuffled_data = data
    num_batches_per_epoch = (len(data) - 1) // batch_size + 1
    for batch_num in range(num_batches_per_epoch):
        start_index = batch_num * batch_size
        end_index = min((batch_num + 1) * batch_size, data_size)
        yield shuffled_data[start_index:end_index]


def calculate_scores(actual_batches, predictions_batches, epoch, logger=None, logger_prefix=None):
    # No positive samples at all
    if sum(predictions_batches) == 0:
        precision = 0
        recall = 0
    else:
        precision = precision_score(actual_batches, predictions_batches)
        recall = recall_score(actual_batches, predictions_batches)

    index = ['precision', 'recall']
    if logger:
        try:
            assert logger_prefix is not None
        except AssertionError:
            raise AssertionError('logger_prefix must be declared if logger')
        logger.log_scalar(f'{logger_prefix}_precision', precision, epoch)
        logger.log_scalar(f'{logger_prefix}_recall', recall, epoch)

    return pd.DataFrame([precision, recall], columns=['flood'], index=index)

def get_confusion_matrix(sess, actual, predictions, labels):
    confusion_matrix = tf.confusion_matrix(
        actual,
        predictions,
        num_classes=len(labels),
        dtype=tf.int32,
        name="confusion_matrix",
        weights=None
    )
    confusion_matrix = tf.Tensor.eval(
        confusion_matrix,
        feed_dict=None,
        session=sess
    )
    return pd.DataFrame(
        confusion_matrix,
        columns=labels,
        index=labels
    )

def get_class_weights(y, kind='balanced', rebalance=None):
    categorical_labels = np.argmax(y, axis=1)
    classes = np.unique(categorical_labels)
    weights = compute_class_weight(kind, classes, categorical_labels)
    if rebalance:
        assert len(weights) == len(rebalance)
        weights = weights * rebalance
    return weights

def do_step(
    loss,
    all_loss,
    actual,
    predictions,
    print_tensor,
    print_tensor2,
    optimizer,
    sess,
    x,
    y,
    dropout,
    batch,
    use_context,
    train
):
    if train:
        dropout_rate = 0.5
    else:
        dropout_rate = 0.0
    to_run_output = [
        loss,
        all_loss,
        actual,
        predictions,
        print_tensor,
        print_tensor2
    ]
    if train:
        to_run = [optimizer] + to_run_output
    else:
        to_run = to_run_output

    if use_context:
        batch_text, batch_context, batch_id, batch_y = zip(*batch)
        x_text, x_context = x
        feed_dict = {
            x_text: batch_text,
            x_context: batch_context,
            y: batch_y,
            dropout: dropout_rate,
        }
    else:
        batch_text, batch_id, batch_y = zip(*batch)
        feed_dict = {
            x: batch_text,
            y: batch_y,
            dropout: dropout_rate,
        }

    *_, c, c_all, batch_actual, batch_predictions, t1, t2 = sess.run(
        to_run,
        feed_dict=feed_dict
    )
    if not np.array_equal(t1, np.bool_(False)):
        print(f't1 {t1.shape}:')
        print(t1)
    if not np.array_equal(t2, np.bool_(False)):
        print(f't2 {t2.shape}:')
        print(t2)
    # print(cs)

    batch_loss = c * len(batch_y)
    batch_loss_all = c_all * len(batch_y)
    return list(batch_actual), list(batch_predictions), list(batch_id), batch_loss, batch_loss_all

def learning_rate_multiplier(alpha):
    @tf.custom_gradient
    def _lr_mult(x):
        def grad(dy):
            return dy * alpha * tf.ones_like(x)
        return x, grad
    return _lr_mult

def get_run_log_folder(name, log_folder):
    last_run_txt = os.path.join(log_folder, 'lastrun.txt')
    if not os.path.exists(last_run_txt):
        run_n = 0
        with open(last_run_txt, 'w') as f:
            f.write(str(run_n))
    else:
        with open(last_run_txt, 'r+') as f:
            run_n = f.read()
            run_n = int(run_n)
            run_n += 1
            f.seek(0)
            f.truncate()
            f.write(str(run_n))
    return os.path.join(log_folder, f"{run_n}_{name}")

def get_positive_weight(y):
    positive = np.sum(y)
    negative = y.shape[0] - positive
    return negative / positive  # 1 / (positive / negative)