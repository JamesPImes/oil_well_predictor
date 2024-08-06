import json

__all__ = [
    'json_to_exp_regress_params',
    'json_to_nearest_wells_params',
]


def json_to_exp_regress_params(fp):
    with open(fp, 'r') as json_file:
        model_params = json.load(json_file)
    kwarg_sets = {}  # For use of params as kwargs.
    for param_set in model_params:
        # key: 'model_name'
        # other params: ['min_months', 'max_months',  'discard_gaps', 'weight_power']
        kwarg_sets[param_set.pop('model_name')] = param_set
    return kwarg_sets


def json_to_nearest_wells_params(fp):
    with open(fp, 'r') as json_file:
        model_params = json.load(json_file)
    kwarg_sets = {}
    for param_set in model_params:
        model_name = param_set['model_name']
        # Convert raw params to kwargs for nearest wells parameters
        nearest_wells_params = {
            'k': param_set['k'],
            'distance_weighting': param_set['distance_weighting'],
        }
        kwarg_sets[model_name] = nearest_wells_params
    return kwarg_sets
