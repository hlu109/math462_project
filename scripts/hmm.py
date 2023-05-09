import utils
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from collections import namedtuple
from sklearn.preprocessing import StandardScaler

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

def reduce_logsumexp(x, axis=None):
    '''
    Reduce with logsumexp along some axis
    '''
    m = np.max(x, axis=axis, keepdims=bool(axis))
    result = np.log(np.sum(np.exp(x - m), axis=axis))
    if axis is not None:
        m = m.reshape(m.shape[:1] + m.shape[2:])
    return m + result

def forward_backward_alg(observations, initial_probs, transition_probs, joint_dists):
    '''
    Given some observations and the transition and initial probabilities, and the joint distribution, compute the posterior state distributions at each time step
    '''
    num_observations = len(observations)

    # Create transition matrix for each transition if necessary
    if len(transition_probs.shape) == 2:
        transition_probs = [transition_probs] * (num_observations - 1)
    transition_log_probs = np.log(transition_probs)

    # Compute probability of observations
    obs_probs = joint_dists.prob(observations.reshape(-1, 1, 1)).numpy()
    obs_log_probs = np.log(obs_probs)

    # Perform forward pass
    forward_log_probs = [obs_log_probs[0] * np.log(initial_probs)]  # Compute base case
    forward_log_probs[-1] -= reduce_logsumexp(forward_log_probs[-1])
    for t in range(1, num_observations):
        forward_log_probs.append(
            obs_log_probs[t] + reduce_logsumexp(
                transition_log_probs[t - 1] + forward_log_probs[-1].reshape(-1, 1),
                axis=0))  # Compute recursive case
        forward_log_probs[-1] -= reduce_logsumexp(forward_log_probs[-1])
    forward_log_probs = np.array(forward_log_probs)

    # Perform backward pass
    backward_log_probs = [reduce_logsumexp(obs_log_probs[-1] + transition_log_probs[-1], axis=1)]  # Compute base case
    for t in reversed(range(num_observations - 2)):
        backward_log_probs.insert(0,
            reduce_logsumexp(backward_log_probs[0] + obs_log_probs[t + 1] + transition_log_probs[t], axis=1))
    backward_log_probs = np.array(backward_log_probs)

    # Compute posterior probabilities from forward and backward log probs
    posterior_log_probs = forward_log_probs[:-1] + backward_log_probs
    posterior_log_probs -= reduce_logsumexp(posterior_log_probs, axis=1).reshape(-1, 1)
    posterior_probs = np.exp(np.vstack([posterior_log_probs, forward_log_probs[-1]]))
    return posterior_probs / np.sum(posterior_probs, axis=1).reshape(-1, 1)

def get_dataset(data, observation_types, covariate_types=None):
    '''
    Extract observations and covariates from the dataframe
    '''
    observations = get_observations(data, observation_types)
    if covariate_types is not None:
        covariates = get_covariates(data, covariate_types)
    else:
        covariates = None

    return (observations, covariates)

def normalize(data, means=None, stds=None):
    '''
    Normalize data so that it has mean 0 and std 1
    '''
    mean = np.mean(data)
    std = np.std(data)
    if means is not None:
        means.append(mean)
    if stds is not None:
        stds.append(stds)
    return (data - mean) / std

def get_observations(data, observation_types):
    '''
    Package together observations from the dataframe
    '''
    observations = np.stack([
        normalize(np.array(data[observation_type]))
            for observation_type in observation_types]).T
    return observations

def get_covariates(data, covariate_types):
    '''
    Package together covariates from the dataframe
    '''
    covariates = np.stack([
        normalize(np.array(data[covariate_type][1:]))
            for covariate_type in covariate_types]).reshape(-1, len(covariate_types), 1, 1)
    return covariates

def split_dataset(dataset, split=0.8):
    '''
    Split dataset into training and testing datasets
    '''
    observations, covariates = dataset
    n = int(len(observations) * split)
    if covariates is None:
        return (observations[:n], None), (observations[n:], None)
    else:
        return (observations[:n], covariates[:n - 1]), (observations[n:], covariates[n - 1:])

def get_counts(dataset):
    '''
    Get various counts
    '''
    observations, covariates = dataset
    num_observations = observations.shape[0]
    num_obs_types = observations.shape[1]
    if covariates is None:
        num_cov_types = 0
    else:
        num_cov_types = covariates.shape[1]
    return num_observations, num_obs_types, num_cov_types

def fit_hmm(dataset, num_states=2, steps=500, rng=np.random.default_rng()):
    '''
    Given a dataset consisting of observations and covariates (optional), fit a hidden markov model with the given number of states
    '''
    observations, covariates = dataset
    num_observations, num_obs_types, num_cov_types = get_counts(dataset)

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
        
        def get_transition_probs():
            exp_transition_logits = tf.exp(transition_logits)
            return exp_transition_logits / tf.reduce_sum(exp_transition_logits, axis=1)
    else:
        def get_transition_logits():
            return tf.reduce_sum(regression_weights * covariates, axis=1) + regression_intercepts
        
        def get_transition_probs():
            exp_transition_logits = tf.exp(get_transition_logits())
            return exp_transition_logits / tf.reshape(tf.reduce_sum(exp_transition_logits, axis=2), [num_observations - 1, num_states, 1])

    # Randomly initialize the initial state distribution
    initial_logits = tf.Variable(rng.random([num_states]),
                                name="initial_logits",
                                dtype=tf.float32)
    initial_distribution = tfd.Categorical(logits=initial_logits)

    # Create state-dependent observation distribution
    dists = []
    for i in range(num_obs_types):
        mu = tf.Variable(np.zeros(num_states), name=f"mu_{i}", dtype=np.float32)
        std = tf.Variable(np.ones(num_states), name=f"std_{i}", dtype=np.float32)
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

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.1)
    criterion = tfp.optimizer.convergence_criteria.LossNotDecreasing(rtol=0.001)
    loss_history = tfp.math.minimize(
        loss_fn=compute_loss,
        num_steps=steps,
        optimizer=optimizer,
        convergence_criterion=criterion,
        trace_fn=trace_fn)

    hmm = get_hmm()
    initial_probs = np.exp(initial_logits) / np.sum(np.exp(initial_logits))
    return loss_history, hmm, ((initial_logits, initial_probs), (get_transition_logits(), get_transition_probs()))

def plot_emissions(hmm, observation_types=None):
    joint_dists = hmm.observation_distribution
    num_obs_types = len(joint_dists.distributions)
    if observation_types is None:
        observation_types = range(num_obs_types)
    
    fig, axs = plt.subplots(1, num_obs_types, figsize=(5 * num_obs_types, 5))
    if type(axs) != np.ndarray:
        axs = np.array([axs])
    num = 1001
    for obs_dist, ax, observation_type in zip(joint_dists.distributions, axs, observation_types):
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

def plot_posterior_probs(dataset, hmm, params):
    observations, _ = dataset
    num_observations, num_obs_types, _ = get_counts(dataset)
    _, (_, transition_probs) = params
    num_states = transition_probs.shape[-1]

    # Infer the posterior distributions
    posterior_dists = hmm.posterior_marginals(observations)
    posterior_probs = posterior_dists.probs_parameter().numpy()
    states = np.argmax(posterior_probs, axis=1)

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
            colors = [utils.colors[i] for i in states]
            n = np.arange(num_observations)
            locs = obs_dist.loc.numpy()[states]
            scales = obs_dist.scale.numpy()[states]
            ax2.fill_between(n, -2 * scales + locs, 2 * scales + locs, alpha=0.4)
            ax2.scatter(n, locs, c=colors, marker='.')
    plt.show()

def plot_covariate_probs(dataset, params):
    _, covariates = dataset
    _, _, num_cov_types = get_counts(dataset)
    _, (_, transition_probs) = params
    num_states = transition_probs.shape[-1]
    if num_cov_types != 0:
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

def predict(dataset, hmm, params):
    '''
    Predict states and observations
    '''
    observations, _ = dataset
    num_observations, _, num_cov_types = get_counts(dataset)
    joint_dists = hmm.observation_distribution
    (_, initial_probs), (_, transition_probs) = params
    predicted_states = [np.argmax(initial_probs)]
    for t in range(2, num_observations + 1):
        if num_cov_types == 0:
            states = np.argmax(forward_backward_alg(
                observations[:t],
                initial_probs,
                transition_probs,
                joint_dists), axis=1)
            predicted_states.append(np.argmax(transition_probs[states[-1]]))
        else:
            states = np.argmax(forward_backward_alg(
                observations[:t],
                initial_probs,
                transition_probs[:t],
                joint_dists), axis=1)
            predicted_states.append(np.argmax(transition_probs[t-2, states[-1]]))
    predicted_states = np.array(predicted_states)

    predicted_observations = np.array([
        [dist.loc[predicted_states[t]].numpy() for dist in joint_dists.distributions]
            for t in range(num_observations)])

    return predicted_states, predicted_observations

def compute_error(observations, predicted_observations):
    return np.mean(np.square(predicted_observations - observations))

if __name__ == "__main__":
    # Get dataframe
    monthly_data = utils.load_interpolated_data()
    utils.add_column_derivative(monthly_data, "median_salary", "median_salary_d1")
    area = "city of london"
    data = utils.get_area(monthly_data, area)

    observation_types = ["average_price_d1"]
    covariate_types = ["median_salary_d1"]
    dataset = get_dataset(data, observation_types, covariate_types)
    train, test = split_dataset(dataset, split=0.8)

    print("Fitting the hmm")
    loss_history, hmm, params = fit_hmm(train, num_states=10)
    print("Fitting finished")

    # plt.plot(loss_history)
    # plt.xlabel("training steps")
    # plt.ylabel("Loss (negative log likelihood)")
    # plt.show()

    # plot_emissions(hmm, observation_types)
    # plot_posterior_probs(train, hmm, params)
    # plot_covariate_probs(dataset, params)

    # Predict and plot predictions
    _, train_predict = predict(train, hmm, params)
    _, test_predict = predict(test, hmm, params)
    x_all = list(range(len(dataset[0])))
    x_train = list(range(len(train[0]))) 
    x_test = list(range(len(train[0]), len(train[0]) + len(test[0])))

    print(f"Train error: {compute_error(train[0], train_predict)}")
    print(f"Test error: {compute_error(test[0], test_predict)}")

    fig, (ax1, ax2) = plt.subplots(2)
    cum_obs = np.cumsum(dataset[0])
    ax1.plot(x_all, cum_obs, label="Truth")
    ax1.plot(x_train, cum_obs[0] + np.cumsum(train_predict), label="Prediction train")
    ax1.plot(x_test, cum_obs[len(train[0])] + np.cumsum(test_predict), label="Prediction test")
    ax1.set_ylabel("Average price ($)")
    ax1.legend()
    ax2.plot(x_all, dataset[0], label="Truth")
    ax2.plot(x_train, train_predict, label="Prediction train")
    ax2.plot(x_test, test_predict, label="Prediction test")
    ax2.set_ylabel("Change in average price ($/month)")
    ax2.legend()
    plt.show()