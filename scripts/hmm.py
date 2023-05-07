import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

rng = np.random.default_rng()

monthly_data, yearly_data = utils.load_dataset()
area = "city of london"
area_data = utils.get_area(monthly_data, area)

num_states = 2

# Package together the observation variables
average_price = np.array(area_data["average_price"])
average_price_d1 = np.array(area_data["average_price_d1"])
observations = np.stack([average_price, average_price_d1]).T
num_observations = len(observations)

# Randomly initialize the initial state distribution
initial_logits = tf.Variable(rng.random([num_states]),
                             name="initial_logits",
                             dtype=tf.float32)
initial_distribution = tfd.Categorical(logits=initial_logits)

# Randomly initialize transition logits
transition_logits = tf.Variable(rng.random([num_states, num_states]),
                                name="transition_logits",
                                dtype=tf.float32)

# Create state-dependent observation distribution
mu_avg_p = tf.Variable(np.zeros(num_states), name="mu_avg_p", dtype=np.float32)
std_avg_p = tf.Variable(np.ones(num_states), name="std_avg_p", dtype=np.float32)
mu_avg_p_d1 = tf.Variable(np.zeros(num_states), name="mu_avg_p_d1", dtype=np.float32)
std_avg_p_d1 = tf.Variable(np.ones(num_states), name="std_avg_p_d1", dtype=np.float32)
dists = [
    tfd.Normal(loc=mu_avg_p, scale=std_avg_p),
    tfd.Normal(loc=mu_avg_p_d1, scale=std_avg_p_d1)
]
joint_dists = tfd.Blockwise(dists)
hmm = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution,
    transition_distribution = tfd.Categorical(logits=transition_logits),
    observation_distribution = joint_dists,
    num_steps = num_observations
)

def compute_loss():
    return -tf.reduce_logsumexp(hmm.log_prob(observations))

def trace_fn(traceable_quantities):
    # Update progress bar
    # t.update()
    return traceable_quantities.loss

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)

print('Training the hmm:')

loss_history = tfp.math.minimize(loss_fn=compute_loss,
                                 num_steps=100,
                                 optimizer=optimizer,
                                 convergence_criterion=criterion)
                                #  trace_fn=trace_fn)

print('Training finished')

plt.plot(loss_history)
plt.show()
