from main import main
from config import SAMPLE_SETS

experiments = [
    'compare_pos_weight',
]

for sample_set in SAMPLE_SETS:
    experiments.append(f'exclude-{sample_set}')
    experiments.append(f'evaluate-{sample_set}')

for experiment in experiments[::-1]:
    main(experiment)
 