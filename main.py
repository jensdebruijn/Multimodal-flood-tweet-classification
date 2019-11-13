import pandas as pd
import os
import itertools
import numpy as np
from datetime import datetime

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf

from load_data import DataLoader
from train import train
from config import SAVE_DIR, SAMPLE_SETS

# Do not display info messages for Tensorflow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'


def create_run_name(experiment, iterations):
    run_name = ""
    for parameter, value in experiment.items():
        value = str(value)
        run_name += f'{parameter}_{value}_'
    run_name += str(iterations)
    return run_name

def main(experiment_type, iterations, n_folds, verbose=False):
    hyper_parameters = {
        'n_epochs': [300],
        'discard_below_or_equal_to_value': [0],  # Discard zeros works better than not discard. Test with discard lower than x..
        'correct_rainfall': [True]
    }

    assert experiment_type in ('learning_rate', 'base_test', 'compare_pos_weight', 'compare_rainfall', 'export_normal', 'batch_size') or experiment_type.startswith('exclude-') or experiment_type.startswith('evaluate-')

    if experiment_type == 'compare_pos_weight':
        POS_WEIGHTS = [0.50, 1.00, 2.00, 3.00, 5.00]
    else:
        POS_WEIGHTS = [2.00]

    if experiment_type == 'compare_rainfall':
        hyper_parameters['rainfall_dataset'] = ['GSMaP', 'PERSIANN']
    else:
        hyper_parameters['rainfall_dataset'] = ['GSMaP']

    if experiment_type.startswith('exclude-') or experiment_type.startswith('evaluate-'):
        language = experiment_type[-2:]
        if experiment_type.startswith('exclude-'):
            hyper_parameters['split'] = [f'exclude-{language}']
        else:
            hyper_parameters['split'] = [f'evaluate-{language}']            
    else:
        hyper_parameters['split'] = ['random']

    if experiment_type == 'learning_rate':
        hyper_parameters['learning_rate'] = [0.0005, 0.0001, 0.00005]
    else:
        hyper_parameters['learning_rate'] = [0.0005]

    if experiment_type == 'batch_size':
        hyper_parameters['batch_size'] = [32, 64, 128, 256, 512]      
    else:
        hyper_parameters['batch_size'] = [128]      
          
    if experiment_type == 'export_normal':
        save_model = True
    else:
        save_model = False

    # assert all hyperparmaters are in list for
    assert all(isinstance(value, list) for value in hyper_parameters.values())

    output_folder = os.path.join('results')
    try:
        os.makedirs(output_folder)
    except OSError:
        pass

    output_fn = os.path.join(output_folder, f'{experiment_type}.xlsx')
    if os.path.exists(output_fn):
        print('already finished')
        return None

    df_columns = ['iteration_n', 'fold_n', 'n_folds', '% pos val', 'test_model', 'use_hydrology', 'pos_weight'] + list(hyper_parameters.keys()) + ['precision', 'recall', 'loss']
    output_df = pd.DataFrame(columns=df_columns)

    keys, values = zip(*hyper_parameters.items())
    experiments = list(itertools.product(*values))
    n_experiments = len(experiments)
    
    for experiment_n, v in enumerate(experiments, start=1):
        settings = dict(zip(keys, v))
        t0 = datetime.now()
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
        all_data = data_loader.get_data(split=settings['split'], n_folds=n_folds, iterations=iterations)

        experiment_iteration = 0
        for percent_positive_validation, run_data, iteration_n, fold_n in all_data:
            experiment_iteration += 1
            for test_model in (False, ):
                for pos_weight in POS_WEIGHTS:
                    for use_hydrology in (True, False):
                        run_name = f'{pos_weight}_{use_hydrology}_{test_model}_{iteration_n}_{fold_n}'
                        if save_model:
                            save_model_path = os.path.join(SAVE_DIR, f'best_model_{run_name}.ckpt')
                        else:
                            save_model_path = None
                        if use_hydrology:
                            run_data_sel = run_data
                        else:
                            run_data_sel = run_data[:-2]
                        best_val_score, best_val_loss = train(
                            run_data_sel,
                            run_name=run_name,
                            log=False,
                            verbose=verbose,
                            pos_weight=pos_weight,
                            n_epochs=settings['n_epochs'],
                            learning_rate=settings['learning_rate'],
                            batch_size=settings['batch_size'],
                            test_model=test_model,
                            use_context=use_hydrology,
                            context_labels=data_loader.context_labels,
                            save_model_path=save_model_path
                        )
                        output_df = output_df.append(pd.Series(
                                [iteration_n, fold_n, n_folds, percent_positive_validation, test_model, use_hydrology, pos_weight] + \
                                list(settings.values()) + \
                                [
                                    best_val_score.loc['precision', 'flood'],
                                    best_val_score.loc['recall', 'flood'],
                                    best_val_loss
                                ],
                                index=df_columns
                        ), ignore_index=True)
                        print(output_df)
            print(f'Experiment {experiment_n}/{n_experiments} finished {experiment_iteration}/{len(data_loader)} iterations', end='\r')
        t1 = datetime.now()
        print(f'Experiment {experiment_n}/{n_experiments} finished {len(data_loader)}/{len(data_loader)} iterations in {t1 - t0}')

        while True:
            try:
                output_df.to_excel(output_fn, index=False)
                break
            except PermissionError:
                print()
                input(f"Please close {output_fn} and press ENTER")
                print('OK')




if __name__ == '__main__':
    EXPERIMENT_TYPE = 'base_test'
    main(EXPERIMENT_TYPE, 1, 5)