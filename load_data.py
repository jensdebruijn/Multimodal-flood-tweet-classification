import json
import numpy as np
import os
import pickle
from collections import Counter, defaultdict
import tensorflow as tf
import random


class DataLoader:
    def __init__(self, fps, embedded_text=True, includes_context=False, includes_groups=False, includes_labels=False):
        self.includes_context = includes_context
        self.includes_labels = includes_labels
        self.includes_groups = includes_groups
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
        if self.includes_groups:
            all_group_ids = []
        for fp in fps:
            with open(fp, 'rb') as p:
                res = pickle.load(p)
            data = res['data']
            ids = data['ids']
            all_languages.extend(data['languages'])
            all_ids.extend(ids)
            if self.includes_groups:
                all_group_ids.extend(data['event_ids'])
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
        if self.includes_groups:
            self.group_ids = np.array(all_group_ids)
        else:
            self.group_ids = np.arange(0, self.ids.size, 1)


    def stratified_group_k_fold(self, y, groups, k, seed=None):
        labels_num = np.max(y) + 1
        y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
        y_distr = Counter()
        for label, g in zip(y, groups):
            y_counts_per_group[g][label] += 1
            y_distr[label] += 1

        y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
        groups_per_fold = defaultdict(set)

        def eval_y_counts_per_fold(y_counts, fold):
            y_counts_per_fold[fold] += y_counts
            std_per_label = []
            for label in range(labels_num):
                label_std = np.std([y_counts_per_fold[i][label] / y_distr[label] for i in range(k)])
                std_per_label.append(label_std)
            y_counts_per_fold[fold] -= y_counts
            return np.mean(std_per_label)
        
        groups_and_y_counts = list(y_counts_per_group.items())
        random.Random(seed).shuffle(groups_and_y_counts)

        for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
            best_fold = None
            min_eval = None
            for i in range(k):
                fold_eval = eval_y_counts_per_fold(y_counts, i)
                if min_eval is None or fold_eval < min_eval:
                    min_eval = fold_eval
                    best_fold = i
            y_counts_per_fold[best_fold] += y_counts
            groups_per_fold[best_fold].add(g)

        all_groups = set(groups)
        for i in range(k):
            train_groups = all_groups - groups_per_fold[i]
            test_groups = groups_per_fold[i]

            train_indices = [i for i, g in enumerate(groups) if g in train_groups]
            test_indices = [i for i, g in enumerate(groups) if g in test_groups]

            yield train_indices, test_indices


    def kfold(self, n_folds, ids, group_ids, y, x_text, x_context=None, statified=False):
        all_data = []
        folds = self.stratified_group_k_fold(y=y, groups=group_ids, k=n_folds)
        for train_index, test_index in folds:  # x 
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

    def k_fold_single_set(self, n_folds, ids, group_ids, y, x_text, x_context):
        output = []
        folds = self.stratified_group_k_fold(y=y, groups=group_ids, k=n_folds)
        for index, _ in folds:
            output.append([
                ids[index],
                group_ids[index],
                y[index],
                x_text[index],
                x_context[index]
            ])
        return output    

    def get_data(self, n_folds, iterations, split):
        self.length = iterations * n_folds
        for iteration_n in range(iterations):
            if split == 'random':
                if self.includes_context:
                    all_data = self.kfold(n_folds, self.ids, self.group_ids, self.y, self.x_text, x_context=self.x_context)
                else:
                    all_data = self.kfold(n_folds, self.ids, self.group_ids, self.y, self.x_text)

            elif split.startswith('only-'):
                language = split[len('only-'):]
                lang_set = np.where(self.languages == language)
                if self.includes_context:
                    all_data = self.kfold(n_folds, self.ids[lang_set], self.group_ids[lang_set], self.y[lang_set], self.x_text[lang_set], x_context=self.x_context[lang_set])
                else:
                    all_data = self.kfold(n_folds, self.ids[lang_set], self.group_ids[lang_set],self.y[lang_set], self.x_text[lang_set])
                    
            elif split.startswith('exclude-'):
                if self.includes_context:
                    language = split[len('exclude-'):]
                    train_set = np.where(self.languages != language)
                    val_set = np.where(self.languages == language)
                    train_data = self.k_fold_single_set(n_folds, self.ids[train_set], self.group_ids[train_set], self.y[train_set], self.x_text[train_set], self.x_context[train_set])
                    all_data = []
                    for train in train_data:
                        all_data.append([
                            train[0], self.ids[val_set],
                            train[2], self.y[val_set],
                            train[3], self.x_text[val_set],
                            train[4], self.x_context[val_set],
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
                        validation_sample_group_ids,
                        validation_sample_y,
                        validation_sample_x_text,
                        validation_sample_x_context
                    ) = self.k_fold_single_set(
                        2,
                        self.ids[languages_evaluated],
                        self.group_ids[languages_evaluated],
                        self.y[languages_evaluated],
                        self.x_text[languages_evaluated],
                        self.x_context[languages_evaluated],
                    )

                    training_sample_ids = np.concatenate([self.ids[languages_not_evaluated], training_sample_lang[0]], axis=0)
                    training_sample_group_ids = np.concatenate([self.group_ids[languages_not_evaluated], training_sample_lang[1]], axis=0)
                    training_sample_y = np.concatenate([self.y[languages_not_evaluated], training_sample_lang[2]], axis=0)
                    training_sample_x_text = np.concatenate([self.x_text[languages_not_evaluated], training_sample_lang[3]], axis=0)
                    training_sample_x_context = np.concatenate([self.x_context[languages_not_evaluated], training_sample_lang[4]], axis=0)

                    train_data = self.k_fold_single_set(n_folds, training_sample_ids, training_sample_group_ids, training_sample_y, training_sample_x_text, training_sample_x_context)
                    all_data = []
                    for train in train_data:
                        all_data.append([
                            train[0], validation_sample_ids,
                            train[2], validation_sample_y,
                            train[3], validation_sample_x_text,
                            train[4], validation_sample_x_context,
                        ])

                else:
                    raise NotImplementedError
            else:
                raise ValueError(f'{split} splitting not available')

            for fold_n, data in enumerate(all_data):
                _, _, _, y_val, *_ = data
                percent_positive_validation = y_val.sum() / y_val.size * 100
                yield percent_positive_validation, data, iteration_n, fold_n



if __name__ == '__main__':
    import pandas as pd

    for split in ('random', 'exclude-en', 'evaluate-en'):
        SAMPLE_SETS = ['en', 'id', 'fr', 'es']
        settings = {
            "split": split,
            "rainfall_dataset": 'GSMaP',
            "discard_below_or_equal_to_value": 0,
            "correct_rainfall": True,
            "replace_locations": True
        }
        fps = [
            os.path.join(
                'data',
                'input',
                f"data_{sample_set}_correct_rainfall_{settings['correct_rainfall']}_discard_below_or_equal_to_value_{settings['discard_below_or_equal_to_value']}_{settings['rainfall_dataset']}.pickle"
            ) for sample_set in SAMPLE_SETS
        ]

        data_loader = DataLoader(
            fps,
            includes_context=True,
            includes_labels=True
        )

        df = pd.read_csv('data/labeled_data_hydrated.csv')
        group_ids = df.set_index('ID')['event_id'].to_dict()
        group_ids = {
            't-' + str(ID): event_id
            for ID, event_id in group_ids.items()
        }

        data_loader.set_data(split=settings['split'], n_folds=5, iterations=10)
        for percent_positive_validation, run_data, iteration_n, fold_n in data_loader.data:
            x_train_id, x_val_id, y_train, y_val, x_train_text, x_val_text, x_train_context, x_val_context = run_data

            print(y_train.sum() / y_train.size)
            print(y_val.sum() / y_val.size)

            train_group_ids = set([group_ids[ID] for ID in x_train_id])
            val_group_ids = set([group_ids[ID] for ID in x_val_id])

            assert not train_group_ids & val_group_ids
