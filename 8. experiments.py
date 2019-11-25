from main import main
from config import SAMPLE_SETS

ITERATIONS = 10
N_FOLDS = 5

experiments = [
    'export_normal',
    'compare_pos_weight',
]

for sample_set in SAMPLE_SETS:
    experiments.append(f'exclude-{sample_set}')
    experiments.append(f'evaluate-{sample_set}')

for experiment in experiments:
    main(experiment, iterations=ITERATIONS, n_folds=N_FOLDS)
 