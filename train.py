import numpy as np
import os
import tensorflow as tf
import pandas as pd

from model import model
from config import (
    EXPORT_DIR,
    SEQUENCE_LENGTH,
    EMBEDDING_DIM,
)
from tf_helpers import (
    gen_batch,
    calculate_scores,
    do_step,
    get_run_log_folder,
    get_positive_weight
)
from logger import Logger

np.set_printoptions(suppress=True)

PATIENCE = None


def export_labels_f(
    batch_ids_val,
    predictions_batches_val,
    actual_batches_val,
    run_name,
):
    data = []
    for ID, prediction, actual in zip(batch_ids_val, predictions_batches_val, actual_batches_val):
        row = (ID, actual, prediction)
        data.append(row)

    df = pd.DataFrame(data, columns=['ID', 'actual', 'prediction'])
    folder = os.path.join(EXPORT_DIR, 'labels')
    try:
        os.makedirs(folder)
    except OSError:
        pass
    fn = os.path.join(folder, f'{run_name}.xlsx')
    while True:
        try:
            df.to_excel(fn, index=False)
            break
        except PermissionError:
            print()
            input(f"Please close {fn} and press ENTER")


def train(
    run_data,
    run_name,
    pos_weight=1,
    n_epochs=100,
    learning_rate=0.0005,
    use_context=True,
    batch_size=128,
    context_labels=None,
    test_model=False,
    save_model_path=True,
    verbose=False,
    log=True,
    return_restore_dir=False,
    export_labels=False
):
    if use_context:
        assert context_labels

    if use_context:
        x_train_id, x_val_id, y_train, y_val, x_train_text, x_val_text, x_train_context, x_val_context = run_data
    else:
        x_train_id, x_val_id, y_train, y_val, x_train_text, x_val_text = run_data

    tf.reset_default_graph()

    (
        x,
        y,
        actual,
        logits,
        predictions,
        dropout,
        l2_loss,
        print_tensor,
        print_tensor2
    ) = model(
        use_context=use_context,
        lang_dim=x_train_text.shape[1],
        context_dim=x_train_context.shape[1] if use_context else None,
        embedding_dim=EMBEDDING_DIM,
        sequence_length=SEQUENCE_LENGTH,
        kind='CNN',
        test_model=test_model
    )

    with tf.Session() as sess:        
        if log:
            log_folder = 'logs'
            logger = Logger(
                get_run_log_folder(run_name, log_folder),
                sess.graph
            )
        else:
            logger = None
        
        saver = tf.train.Saver(max_to_keep=1)

        # Sigmoid cross entropy is typically used for binary classification.
        # It can handle multiple labels, 
        # but sigmoid cross entropy basically makes a (binary) decision on each of them
        losses = tf.nn.weighted_cross_entropy_with_logits(
            logits=logits,
            targets=y,
            pos_weight=get_positive_weight(y_train) * pos_weight
        )
        loss = tf.reduce_mean(losses)
        all_loss = loss + l2_loss
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta2=0.999).minimize(all_loss)

        sess.run(tf.global_variables_initializer())

        validation_losses_per_epoch = []

        # for the validation batches, no shuffling is required as no optimization
        # is performed during the validation stages
        val_batches = list(
            gen_batch(
                zip(x_val_text, x_val_context, x_val_id, y_val) if use_context else zip(x_val_text, x_val_id, y_val),
                batch_size,
                shuffle=False
            )
        )

        for global_epoch in range(1, n_epochs+1):
            # TRAINING PHASE
            # Reshuffle training data for each epoch
            train_batches = gen_batch(
                zip(x_train_text, x_train_context, x_train_id, y_train) if use_context else zip(x_train_text, x_train_id, y_train),
                batch_size,
                shuffle=True
            )
            train_loss = 0.
            train_loss_all = 0.
            actual_batches_train = []
            predictions_batches_train = []
            batch_ids_train = []
            batch_context_train = []
            for train_batch in train_batches:
                batch_actual, batch_predictions, _, batch_loss, batch_loss_all = do_step(
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
                    train_batch,
                    use_context=use_context,
                    train=True, 
                )
                train_loss += batch_loss
                train_loss_all += batch_loss_all

                actual_batches_train.extend(batch_actual)
                predictions_batches_train.extend(batch_predictions)
                batch_ids_train.extend(train_batch[:,2])
                batch_context_train.extend(train_batch[:,1])

            train_scores = calculate_scores(
                actual_batches_train,
                predictions_batches_train,
                global_epoch,
                logger=logger,
                logger_prefix="train"
            )
            
            avg_train_loss = train_loss / len(y_train)
            if logger:
                logger.log_scalar('train_loss', avg_train_loss, global_epoch)
            
            if verbose:
                print("Global epoch:", '%04d' % (global_epoch), "training loss = {:.9f}".format(avg_train_loss))

            # VALIDATION PHASE
            val_loss = 0.
            val_loss_all = 0.
            actual_batches_val = []
            predictions_batches_val = []
            batch_ids_val = []
            for val_batch in val_batches:
                batch_actual, batch_predictions, batch_id, batch_loss, batch_loss_all = do_step(
                    loss,
                    all_loss,
                    actual,
                    predictions,
                    print_tensor,
                    print_tensor2,
                    None,
                    sess,
                    x,
                    y,
                    dropout,
                    val_batch,
                    use_context=use_context,
                    train=False,
                )
                
                val_loss += batch_loss
                val_loss_all += batch_loss_all
                
                actual_batches_val.extend(batch_actual)
                predictions_batches_val.extend(batch_predictions)
                batch_ids_val.extend(batch_id)

            val_scores = calculate_scores(
                actual_batches_val,
                predictions_batches_val,
                global_epoch,
                logger=logger,
                logger_prefix='val'
            )

            avg_val_loss = val_loss / len(y_val)
            if logger:
                logger.log_scalar('val_loss', avg_val_loss, global_epoch)
            
            validation_losses_per_epoch.append(avg_val_loss)
            best_model_idx = np.argmin(validation_losses_per_epoch)
            
            if verbose:
                print(val_scores)
                print("Epoch:", '%04d' % (global_epoch))
                print("Epoch:", '%04d' % (global_epoch), "validation loss = {:.9f}".format(avg_val_loss))
                print('Best model so far:', best_model_idx + 1)

            if best_model_idx + 1 == global_epoch:
                best_model_val_score = val_scores
                if export_labels:
                    best_model_export_for_error_label = batch_ids_val, predictions_batches_val, actual_batches_val
                best_val_loss = avg_val_loss

                if save_model_path:
                    saver.save(sess, save_model_path, write_meta_graph=True)
            
            if PATIENCE and best_model_idx < len(validation_losses_per_epoch) - PATIENCE:
                if verbose:
                    print(f"Stopping early")
                    print()
                break
            if verbose:
                print()
        
        best_model_epoch = best_model_idx + 1

        if export_labels:
            assert len(best_model_export_for_error_label) == 3
            export_labels_f(
                *best_model_export_for_error_label,
                run_name,
            )

        if verbose:
            print()
            print(f"best model:", best_model_epoch, "validation loss = {:.9f}".format(best_val_loss))
            print(best_model_val_score)

        if return_restore_dir:
            restore_dir = os.path.join(EXPORT_DIR, f'{best_model_epoch}.ckpt')
            return best_model_val_score, best_val_loss, restore_dir
        else:
            return best_model_val_score, best_val_loss
