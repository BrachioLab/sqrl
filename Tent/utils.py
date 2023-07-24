
import numpy as np
from scipy.stats import bootstrap

import pickle
def calculate_confidence_interval(stat_ls):
    stat_array = np.array(stat_ls)
    rng = np.random.default_rng()
    # res = bootstrap((stat_array,), np.mean, confidence_level=0.9,
    #             random_state=rng)
    # f1_low, f1_high = bootstrap([stat_array, ], np.median, confidence_level=0.99, method='percentile').confidence_interval

    # f1_low_low, f1_low_high = bootstrap([stat_array, ], np_percentile_lower, confidence_level=0.99, method='percentile', random_state=rng).confidence_interval

    # f1_high_low, f1_high_high = bootstrap([stat_array, ], np_percentile_upper, confidence_level=0.99, method='percentile', random_state=rng).confidence_interval

    f1_low_low, f1_low_high = bootstrap([stat_array, ], np.min, confidence_level=0.99, method='percentile').confidence_interval

    f1_high_low, f1_high_high = bootstrap([stat_array, ], np.max, confidence_level=0.99, method='percentile').confidence_interval


    return f1_low_low, f1_low_high, f1_high_low, f1_high_high

def save_objs(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)

def load_objs(file):
    with open(file, "rb") as f:
        obj = pickle.load(f)
    return obj