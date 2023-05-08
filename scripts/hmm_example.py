import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

from matplotlib import pyplot as plt
import scipy.stats

true_rates = [40, 3, 20, 50]
true_durations = [10, 20, 5, 35]

true_states = tf.concat([
    [rate for _ in range(num_steps)]
        for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

observed_counts = tf.concat(
    [tfd.Poisson(rate).sample(num_steps)
        for (rate, num_steps) in zip(true_rates, true_durations)], axis=0)

print(observed_counts)
plt.plot(true_states)
plt.plot(observed_counts)
plt.show()

num_states = 4
initial_state_logits = tf.zeros([num_states]) # uniform distribution
daily_change_prob = 0.05
transition_probs = daily_change_prob / (num_states - 1) * np.ones(
    [num_states, num_states], dtype=np.float32)
np.fill_diagonal(transition_probs, 1 - daily_change_prob)

print("Initial state logits:\n{}".format(initial_state_logits))
print("Transition matrix:\n{}".format(transition_probs))

trainable_log_rates = tf.Variable(
    np.log(np.mean(observed_counts)) + tf.random.normal([num_states]),
    name="log_rates")

hmm = tfd.HiddenMarkovModel(
    initial_distribution = tfd.Categorical(
        logits = initial_state_logits),
    transition_distribution = tfd.Categorical(probs=transition_probs),
    observation_distribution = tfd.Poisson(log_rate=trainable_log_rates),
    num_steps = len(observed_counts))

rate_prior = tfd.LogNormal(5, 5)

def log_prob():
    # return (tf.reduce_sum(rate_prior.log_prob(tf.math.exp(trainable_log_rates))) + hmm.log_prob(observed_counts))
    return hmm.log_prob(observed_counts)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)

@tf.function(autograph=False)
def train_op():
    with tf.GradientTape() as tape:
        neg_log_prob = -log_prob()
    grads = tape.gradient(neg_log_prob, [trainable_log_rates])[0]
    optimizer.apply_gradients([(grads, trainable_log_rates)])
    return neg_log_prob, tf.math.exp(trainable_log_rates)

for step in range(201):
    loss, rates = [t.numpy() for t in train_op()]
    if step % 20 == 0:
        print("step {}: log prob {} rates {}".format(step, -loss, rates))

print("Inferred rates: {}".format(rates))
print("True rates: {}".format(true_rates))
