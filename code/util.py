import tensorflow as tf

# sampling methods

def sample(probs):
    return tf.nn.relu(tf.sign(probs - tf.random.uniform(tf.shape(probs))))


# evaluation metrics
def mse(x, x_):
    return tf.reduce_mean(tf.square(x - x_))

def mae(x, x_):
    return tf.reduce_mean(tf.abs(x - x_))


