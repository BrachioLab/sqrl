import numpy as np

class discrete_dist:
    def __init__(self, array):
        self.array = array

    def calculate_sampling_probs(self):
        elem_array, count_array = np.unique(self.array, return_counts=True)
        self.prob_array = count_array/np.sum(count_array)
        self.elem_array = elem_array

    def sample(self, n_samples=1, random_state=None):
        return np.random.choice(self.elem_array, n_samples, replace=True, p=self.prob_array)