import numpy as np

def to_knary_one_hot(data, k):
    temp = np.eye(k + 2)[data][:, 1:-1]
    return temp.reshape(-1, len(data), k)

def max_norm(data):
    return (data - data.min()) / (data.max() - data.min())