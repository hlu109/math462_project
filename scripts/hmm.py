import utils
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd

def plot_state_posterior(ax, state_posterior_probs, observed_data, title="", label="", ylabel=""):
    '''Plot state posterior distributions.'''
    ln1 = ax.plot(state_posterior_probs, c="blue", lw=3, label="p(state | obs)")
    ax.set_ylim(0., 1.1)
    ax.set_ylabel("posterior probability")
    ax2 = ax.twinx()
    ln2 = ax2.plot(observed_data, c="black", alpha=0.3, label=label)
    ax2.set_title(title)
    ax2.set_ylabel(ylabel)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color="white")
    ax2.grid(False)
    return ax2

def plot_transition_probs(ax, transition_probs, covariate_data, title="", label="", ylabel=""):
    '''Plot state transition probabilities'''
    ln1 = ax.plot(transition_probs, c="blue", lw=3, label="prob")
    ax.set_ylim(0., 1.1)
    ax.set_ylabel("transition probability")
    ax2 = ax.twinx()
    ln2 = ax2.plot(covariate_data, c="black", alpha=0.3, label=label)
    ax2.set_title(title)
    ax2.set_ylabel(ylabel)
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    ax.grid(True, color="white")
    ax2.grid(False)
    return ax2

monthly_data = utils.load_interpolated_data()
utils.add_column_derivative(monthly_data, "median_salary", "median_salary_d1")
area = "city of london"
data = utils.get_area(monthly_data, area)

true_locs = [5, 10, 2]
true_scales = [1, 1, 1]
true_durations = [100, 100, len(data) - 200]
data["test"] = tf.concat(
    [tfd.Normal(loc, scale).sample(num_steps)
        for (loc, scale, num_steps) in zip(true_locs, true_scales, true_durations)], axis=0)

num_states = 2
observation_types = ["average_price_d1"]
covariate_types = ["median_salary_d1"]
rng = np.random.default_rng()
training_steps = 200

# def fit_hmm(data, num_states, observation_types=[], covariate_types=[], rng=np.random.default_rng()):
# Package together the observation variables

def normalize(data):
    mean = np.mean(data)
    std = np.std(data)
    return (data - mean) / std

observations = np.stack([
    normalize(np.array(data[observation_type]))
        for observation_type in observation_types]).T
num_observations = observations.shape[0]
num_obs_types = observations.shape[1]

# Package together the covariates
num_cov_types = len(covariate_types)
if num_cov_types != 0:
    covariates = np.stack([
        normalize(np.array(data[covariate_type][1:]))
            for covariate_type in covariate_types]).reshape(-1, num_cov_types, 1, 1)

# Randomly initialize the regression coefficients
if num_cov_types != 0:
    regression_weights = tf.Variable(
        np.ones([num_cov_types, num_states, num_states]) * (1 - np.diag([1] * num_states)),
        name="regression_weights",
        dtype=tf.float32)
    regression_intercepts = tf.Variable(
        np.ones([1, num_states, num_states]) * (1 - np.diag([1] * num_states)),
        name="regression_intercepts",
        dtype=tf.float32)

# If there are no covariates, randomly initialize and cache the transition logits
if num_cov_types == 0:
    transition_logits = tf.Variable(rng.random([num_states, num_states]),
                                    name="transition_logits",
                                    dtype=tf.float32)
    def get_transition_logits():
        return transition_logits
else:
    def get_transition_logits():
        return tf.reduce_sum(regression_weights * covariates, axis=1) + regression_intercepts

# Randomly initialize the initial state distribution
initial_logits = tf.Variable(rng.random([num_states]),
                            name="initial_logits",
                            dtype=tf.float32)
initial_distribution = tfd.Categorical(logits=initial_logits)

# Create state-dependent observation distribution
dists = []
for observation_type in observation_types:
    mu = tf.Variable(np.zeros(num_states), name=f"mu_{observation_type}", dtype=np.float32)
    std = tf.Variable(np.ones(num_states), name=f"std_{observation_type}", dtype=np.float32)
    dists.append(tfd.Normal(loc=mu, scale=std))
joint_dists = tfd.Blockwise(dists)

# If there are no covariates, cache the HMM
if num_cov_types == 0:
    hmm = tfd.HiddenMarkovModel(
        initial_distribution = initial_distribution,
        transition_distribution = tfd.Categorical(logits=get_transition_logits()),
        observation_distribution = joint_dists,
        num_steps = num_observations,
        time_varying_transition_distribution = False)

    def get_hmm():
        return hmm
else:
    def get_hmm():
        return tfd.HiddenMarkovModel(
            initial_distribution = initial_distribution,
            transition_distribution = tfd.Categorical(logits=get_transition_logits()),
            observation_distribution = joint_dists,
            num_steps = num_observations,
            time_varying_transition_distribution = True)

def compute_loss():
    hmm = get_hmm()
    return -tf.reduce_logsumexp(hmm.log_prob(observations))

def trace_fn(traceable_quantities):
    # Update progress bar
    # t.update()
    if num_cov_types != 0:
        regression_weights.assign(regression_weights * (1 - np.diag([1] * num_states)))
        regression_intercepts.assign(regression_intercepts * (1 - np.diag([1] * num_states)))
    return traceable_quantities.loss

print('Training the hmm:')
optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)
loss_history = tfp.math.minimize(
    loss_fn=compute_loss,
    num_steps=training_steps,
    optimizer=optimizer,
    convergence_criterion=criterion,
    trace_fn=trace_fn)
print('Training finished')

hmm = get_hmm()

plt.plot(loss_history)
plt.xlabel("training steps")
plt.ylabel("Loss (negative log likelihood)")
plt.show()

# Plot emissions
fig, axs = plt.subplots(1, num_obs_types, figsize=(5 * num_obs_types, 5))
if type(axs) != np.ndarray:
    axs = np.array([axs])
num = 1001
for (j, (obs_dist, ax, observation_type)) in enumerate(zip(hmm.observation_distribution.distributions, axs, observation_types)):
    max_scale = np.max(obs_dist.scale)
    min_loc, max_loc = np.min(obs_dist.loc), np.max(obs_dist.loc)
    x = np.linspace(min_loc - 3 * max_scale, max_loc + 3 * max_scale, num).reshape(-1, 1)
    y = obs_dist.prob(x).numpy()
    for i in range(y.shape[1]):
        label = f"state {i}, loc={obs_dist.loc[i]:.2f}, scale={obs_dist.scale[i]:.2f}"
        ax.plot(x[:, 0], y[:, i], label=label)
        ax.set_title(f"{observation_type} distributions")
        ax.legend(loc="upper right")
plt.show()

# Infer the posterior distributions
posterior_dists = hmm.posterior_marginals(observations)
posterior_probs = posterior_dists.probs_parameter().numpy()
most_likely_states = np.argmax(posterior_probs, axis=1)

# Plot posterior probabilities
fig, axs = plt.subplots(num_states, num_obs_types, figsize=(7 * num_obs_types, 5 * num_states), sharex=True)
axs = axs.reshape(num_states, num_obs_types)
for state, ax_row in enumerate(axs):
    for i, (ax, obs_dist) in enumerate(zip(ax_row, hmm.observation_distribution.distributions)):
        ax2 = plot_state_posterior(
            ax =                    ax,
            state_posterior_probs = posterior_probs[:, state],
            observed_data =         observations[:, i],
            title =                 f"State {state}",
            label =                 f"{observation_types[i]}",
            ylabel =                f"normalized {observation_types[i]}")
        colors = [utils.colors[i] for i in most_likely_states]
        n = np.arange(num_observations)
        locs = obs_dist.loc.numpy()[most_likely_states]
        scales = obs_dist.scale.numpy()[most_likely_states]
        ax2.fill_between(n, -2 * scales + locs, 2 * scales + locs, alpha=0.4)
        ax2.scatter(n, locs, c=colors, marker='.')
plt.show()

if num_cov_types != 0:
    transition_logits = get_transition_logits()
    transition_probs = tf.exp(transition_logits) / tf.reshape(tf.reduce_sum(tf.exp(transition_logits), axis=2), [num_observations - 1, 2, 1])
    for k, covariate_type in enumerate(covariate_types):
        fig, axs = plt.subplots(num_states, num_states, figsize=(7 * num_states, 5 * num_states), sharex=True, sharey=True)
        axs = axs.reshape(num_states, num_states)
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                plot_transition_probs(
                    ax = ax,
                    transition_probs = transition_probs[:, i, j],
                    covariate_data = covariates[:, k, 0, 0],
                    title = f"Probability S_t = {j} given S_t-1 = {i}",
                    label = covariate_type,
                    ylabel = covariate_type)
plt.show()
        
# if __name__ == "__main__":
#     monthly_data = utils.load_interpolated_data()
#     area = "city of london"
#     area_data = utils.get_area(monthly_data, area)
#     hmm = fit_hmm(area_data, 2,
#         ["average_price_d1"],
#         ["median_salary"])