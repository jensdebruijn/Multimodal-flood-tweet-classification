import pandas as pd
import json
import numpy as np
import os
import pickle
import tensorflow as tf

from sklearn.model_selection import KFold, StratifiedKFold


class DataLoader:
    def __init__(self, fps, embedded_text=True, includes_context=False, includes_labels=False):
        self.includes_context = includes_context
        self.includes_labels = includes_labels
        self.load(fps, embedded_text)

    def __len__(self):
        if hasattr(self, 'length'):
            return self.length
        else:
            raise ValueError('Data not initalized - run create_data')

    def load(self, fps, embedded_text):
        if not isinstance(fps, (list, tuple)):
            fps = [fps]

        all_ids = []
        all_text = []
        all_languages = []
        if self.includes_context:
            all_context = []
        if self.includes_labels:
            all_labels = []
        for fp in fps:
            with open(fp, 'rb') as p:
                res = pickle.load(p)
            data = res['data']
            ids = data['ids']
            all_languages.extend(data['languages'])
            all_ids.extend(ids)
            if embedded_text:
                embedded_sequences = self.lookup_word_embeddings(
                    data['text_sequences'],
                    res['embedding_matrix']
                )
                all_text.extend(embedded_sequences)
            else:
                all_text.extend(data['sentences'])
            if self.includes_context:
                context_labels = res['context_labels']
                context = data['context']
                all_context.extend(context.tolist())
            if self.includes_labels:
                labels = data['labels']
                all_labels.extend(labels)

        if self.includes_labels:
            y = np.array(all_labels)
                
            self.y = y
        if self.includes_context:
            self.x_context = np.array(all_context) / 100
            assert (self.x_context <= 1).all() and (self.x_context >= 0).all()
            self.context_labels = context_labels

        self.languages = np.array(all_languages)
        self.x_text = np.array(all_text)
        self.ids = np.array(all_ids)


    def kfold(self, n_folds, ids, y, x_text, x_context=None, statified=True):
        all_data = []
        if statified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True)
        for train_index, test_index in kf.split(X=np.zeros(len(y)), y=y):  # x 
            data = [
                    ids[train_index],
                    ids[test_index],
                    y[train_index],
                    y[test_index],
                    x_text[train_index],
                    x_text[test_index],
                ]
            if self.includes_context:
                data.extend([
                    x_context[train_index],
                    x_context[test_index],
                ])
            all_data.append(data)
        return all_data

    def lookup_word_embeddings(self, text_sequences, embedding_matrix):
        with tf.Session() as sess:
            embedding_matrix = tf.Variable(embedding_matrix, dtype=tf.float32)
            text_sequences = tf.nn.embedding_lookup(embedding_matrix, text_sequences)
            sess.run(tf.global_variables_initializer())
            text_sequences = text_sequences.eval()
        tf.reset_default_graph()
        return text_sequences

    def set_data(self, n_folds, iterations, split):
        self.data = self.create_data(n_folds, iterations, split)

    def k_fold_single_set(self, n_folds, ids, y, x_text, x_context, stratified=True):
        if stratified:
            kf = StratifiedKFold(n_splits=n_folds, shuffle=True)
        else:
            kf = KFold(n_splits=n_folds, shuffle=True)
        output = []
        for index, _ in kf.split(X=np.zeros(len(y)), y=y):
            output.append([
                ids[index],
                y[index],
                x_text[index],
                x_context[index]
            ])
        return output    

    def create_data(self, n_folds, iterations, split):
        self.length = iterations * n_folds
        for iteration_n in range(iterations):
            if split == 'random':
                if self.includes_context:
                    all_data = self.kfold(n_folds, self.ids, self.y, self.x_text, x_context=self.x_context)
                else:
                    all_data = self.kfold(n_folds, self.ids, self.y, self.x_text)

            elif split.startswith('only-'):
                language = split[len('only-'):]
                lang_set = np.where(self.languages == language)
                if self.includes_context:
                    all_data = self.kfold(n_folds, self.ids[lang_set], self.y[lang_set], self.x_text[lang_set], x_context=self.x_context[lang_set])
                else:
                    all_data = self.kfold(n_folds, self.ids[lang_set], self.y[lang_set], self.x_text[lang_set])
                    
            elif split.startswith('exclude-'):
                if self.includes_context:
                    language = split[len('exclude-'):]
                    train_set = np.where(self.languages != language)
                    val_set = np.where(self.languages == language)
                    train_data = self.k_fold_single_set(n_folds, self.ids[train_set], self.y[train_set], self.x_text[train_set], self.x_context[train_set], stratified=True)
                    all_data = []
                    for train in train_data:
                        all_data.append([
                            train[0], self.ids[val_set],
                            train[1], self.y[val_set],
                            train[2], self.x_text[val_set],
                            train[3], self.x_context[val_set],
                        ])
                else:
                    raise NotImplementedError
            elif split.startswith('evaluate-'):
                if self.includes_context:
                    language = split[len('evaluate-'):]
                    languages_not_evaluated = np.where(self.languages != language)
                    languages_evaluated = np.where(self.languages == language)

                    training_sample_lang, (
                        validation_sample_ids,
                        validation_sample_y,
                        validation_sample_x_text,
                        validation_sample_x_context
                    ) = self.k_fold_single_set(
                        2,
                        self.ids[languages_evaluated],
                        self.y[languages_evaluated],
                        self.x_text[languages_evaluated],
                        self.x_context[languages_evaluated],
                        stratified=True
                    )

                    training_sample_ids = np.concatenate([self.ids[languages_not_evaluated], training_sample_lang[0]], axis=0)
                    training_sample_y = np.concatenate([self.y[languages_not_evaluated], training_sample_lang[1]], axis=0)
                    training_sample_x_text = np.concatenate([self.x_text[languages_not_evaluated], training_sample_lang[2]], axis=0)
                    training_sample_x_context = np.concatenate([self.x_context[languages_not_evaluated], training_sample_lang[3]], axis=0)

                    train_data = self.k_fold_single_set(n_folds, training_sample_ids, training_sample_y, training_sample_x_text, training_sample_x_context, stratified=True)
                    all_data = []
                    for train in train_data:
                        all_data.append([
                            train[0], validation_sample_ids,
                            train[1], validation_sample_y,
                            train[2], validation_sample_x_text,
                            train[3], validation_sample_x_context,
                        ])

                else:
                    raise NotImplementedError
            else:
                raise ValueError(f'{split} splitting not available')

            for fold_n, data in enumerate(all_data):
                _, _, _, y_val, *_ = data
                percent_positive_validation = y_val.sum() / y_val.size * 100
                yield percent_positive_validation, data, iteration_n, fold_n