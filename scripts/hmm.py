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
    first_idx = 0
    last_idx = len(data)
    observations = []
    for col_name in observation_types:
        column = data[col_name]
        observations.append(column)
        first_idx = max(first_idx, column.first_valid_index())
        last_idx = min(last_idx, column.last_valid_index())

    if covariate_types is not None:
        covariates = []
        for col_name in covariate_types:
            column = data[col_name]
            covariates.append(column)
            first_idx = max(first_idx, column.first_valid_index())
            last_idx = min(last_idx, column.last_valid_index())
        covariates = np.expand_dims(np.stack(covariates).T, (2, 3))[first_idx:last_idx]
        # covariates = np.stack(covariates).reshape(-1, len(covariate_types), 1, 1)[first_idx:last_idx]
    else:
        covariates = None

    observations = np.stack(observations).T[first_idx:last_idx]

    return observations, covariates

def get_dataset_normalization_params(dataset):
    observations, covariates = dataset
    obs_means, obs_stds = np.mean(observations, axis=0), np.std(observations, axis=0)
    if covariates is not None:
        cov_means, cov_stds = np.mean(covariates, axis=0), np.std(covariates, axis=0)
    else:
        cov_means = cov_stds = None
    return obs_means, obs_stds, cov_means, cov_stds

def normalize(dataset, normps):
    '''
    Normalize data so that it has mean 0 and std 1
    '''
    observations, covariates = dataset
    obs_means, obs_stds, cov_means, cov_stds = normps
    normed_obs = (observations - obs_means) / obs_stds
    if covariates is not None:
        normed_covs = (covariates - cov_means) / cov_stds
    else:
        normed_covs = None
    return normed_obs, normed_covs

def denormalize(dataset, normps):
    '''
    Denormalize data using the normalization parameters filled in by get_dataset
    '''
    observations, covariates = dataset
    obs_means, obs_stds, cov_means, cov_stds = normps
    denormed_obs = obs_stds * observations + obs_means
    if covariates is not None:
        denormed_covs = cov_stds * covariates + cov_means
    else:
        denormed_covs = None
    return denormed_obs, denormed_covs

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

def fit_hmm(dataset, normps, num_states=2, steps=500, rng=np.random.default_rng()):
    '''
    Given a dataset consisting of observations and covariates (optional), fit a hidden markov model with the given number of states
    '''
    observations, covariates = normalize(dataset, normps)
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

def plot_emissions(hmm, observation_types=None, save=False, savedir=None):
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
    if save:
        utils.savefig(fig, f"{savedir}/emissions.png")

def plot_posterior_probs(dataset, normps, hmm, hmm_params, unified=False, save=False, savedir=None):
    observations, _ = dataset
    normed_obs, _ = normalize(dataset, normps)
    obs_means, obs_stds, _, _ = normps
    num_observations, num_obs_types, _ = get_counts(dataset)
    _, (_, transition_probs) = hmm_params
    num_states = transition_probs.shape[-1]

    # Infer the posterior distributions
    posterior_dists = hmm.posterior_marginals(normed_obs)
    posterior_probs = posterior_dists.probs_parameter().numpy()
    states = np.argmax(posterior_probs, axis=1)

    # Plot posterior probabilities
    if unified:
        fig, axs = plt.subplots(1, num_obs_types, figsize=(7 * num_obs_types, 5))
        axs = np.array(axs).reshape(num_obs_types)
        for i, (ax, obs_dist) in enumerate(zip(axs, hmm.observation_distribution.distributions)):
            ax.plot(observations[:, i], c="black", alpha=0.3, label=observation_types[i])
            ax.legend(loc=4)
            ax.grid(True, color="white")

            colors = [utils.colors[i] for i in states]
            n = np.arange(num_observations)
            locs = obs_dist.loc.numpy()[states] * obs_stds[i] + obs_means[i]
            scales = obs_dist.scale.numpy()[states] * obs_stds[i]
            ax.fill_between(n, -2 * scales + locs, 2 * scales + locs, alpha=0.4)
            ax.scatter(n, locs, c=colors, marker='.')
    else:
        fig, axs = plt.subplots(num_states, num_obs_types, figsize=(7 * num_obs_types, 5 * num_states), sharex=True)
        axs = np.array(axs).reshape(num_states, num_obs_types)
        for state, ax_row in enumerate(axs):
            for i, (ax, obs_dist) in enumerate(zip(ax_row, hmm.observation_distribution.distributions)):
                ax2 = plot_state_posterior(
                    ax =                    ax,
                    state_posterior_probs = posterior_probs[:, state],
                    observed_data =         observations[:, i],
                    title =                 f"State {state}",
                    label =                 observation_types[i],
                    ylabel =                observation_types[i])
                colors = [utils.colors[i] for i in states]
                n = np.arange(num_observations)
                locs = obs_dist.loc.numpy()[states] * obs_stds[i] + obs_means[i]
                scales = obs_dist.scale.numpy()[states] * obs_stds[i]
                ax2.fill_between(n, -2 * scales + locs, 2 * scales + locs, alpha=0.4)
                ax2.scatter(n, locs, c=colors, marker='.')
    plt.show()
    if save:
        utils.savefig(fig, f"{savedir}/posterior_probs.png")

def plot_covariate_probs(dataset, hmm_params, covariate_types=None, unified=False, save=False, savedir=None):
    _, covariates = dataset
    _, _, num_cov_types = get_counts(dataset)

    if num_cov_types == 0:
        return

    if covariate_types is None:
        covariate_types = range(num_cov_types)

    _, (_, transition_probs) = hmm_params
    num_states = transition_probs.shape[-1]
    if unified:
        fig, axs = plt.subplots(num_states, num_states, figsize=(7 * num_states, 5 * num_states), sharex=True, sharey=True)
        axs = np.array(axs).reshape(num_states, num_states)
        for i, ax_row in enumerate(axs):
            for j, ax in enumerate(ax_row):
                ax.plot(transition_probs[:, i, j], c="blue", lw=3)
                ax.set_ylim(0., 1.1)
                ax.grid(True, color="white")
        plt.show()
        if save:
            utils.savefig(fig, f"{savedir}/transition_probs.png")
    else:
        for k, covariate_type in enumerate(covariate_types):
            fig, axs = plt.subplots(num_states, num_states, figsize=(7 * num_states, 5 * num_states), sharex=True, sharey=True)
            axs = np.array(axs).reshape(num_states, num_states)
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
            if save:
                utils.savefig(fig, f"{savedir}/transition_probs_vs_{covariate_type}.png")

def predict(dataset, normps, hmm, hmm_params):
    '''
    Predict states and observations
    '''
    observations, _ = normalize(dataset, normps)
    num_observations, _, num_cov_types = get_counts(dataset)
    joint_dists = hmm.observation_distribution
    (_, initial_probs), (_, transition_probs) = hmm_params
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

    return predicted_states, denormalize((predicted_observations, None), normps)[0]

def rmse(observations, predicted_observations):
    return np.sqrt(np.mean(np.square(observations - predicted_observations)))

if __name__ == "__main__":
    save = True
    savedir = "../plots/city_of_london"

    # Get dataframe
    monthly_data = utils.load_interpolated_data()
    utils.add_column_derivative(monthly_data, "median_salary", "median_salary_d1")
    area = "city of london"
    data = utils.get_area(monthly_data, area)
    split = 0.7

    # Create dataset and split it into training and testing datasets
    observation_types = ["average_price", "average_price_d1"]
    covariate_types = ["median_salary_d1", "population_size", "number_of_jobs", "no_of_houses"]
    dataset = get_dataset(data, observation_types, covariate_types)
    train, test = split_dataset(dataset, split)

    # Get true average price data and split it
    avg_price = dataset[0][:, 0].reshape(-1, 1)
    avg_price_train = train[0][:, 0].reshape(-1, 1)
    avg_price_test = test[0][:, 0].reshape(-1, 1)

    # Remove average price from observations
    dataset = (dataset[0][:, 1:], dataset[1])
    train = (train[0][:, 1:], train[1])
    test = (test[0][:, 1:], test[1])
    observation_types = observation_types[1:]

    # Fit the hmm and plot the loss history
    print("Fitting the hmm")
    normps = get_dataset_normalization_params(train)  # Necessary to denormalize later
    loss_history, hmm, hmm_params = fit_hmm(train, normps, num_states=10, steps=1000)
    print("Fitting finished")
    plt.plot(loss_history)
    plt.xlabel("training steps")
    plt.ylabel("Loss (negative log likelihood)")
    fig = plt.gcf()
    plt.show()
    if save:
        utils.savefig(fig, f"{savedir}/loss_history.png")

    # Plot various information about the hmm
    plot_emissions(hmm, observation_types, save=save, savedir=savedir)
    plot_posterior_probs(train, normps, hmm, hmm_params, unified=True, save=save, savedir=savedir)
    plot_covariate_probs(train, hmm_params, covariate_types, unified=True, save=save, savedir=savedir)

    # Predict and plot predictions
    _, train_predict = predict(train, normps, hmm, hmm_params)
    _, test_predict = predict(test, normps, hmm, hmm_params)
    x_all = list(range(len(dataset[0])))
    x_train = list(range(len(train[0]))) 
    x_test = list(range(len(train[0]), len(train[0]) + len(test[0])))

    print(f"Train error avg_p_d1: {rmse(train[0], train_predict)}")
    print(f"Test error avg_p_d1: {rmse(test[0], test_predict)}")
    plt.plot(x_all, dataset[0])
    plt.plot(x_train, train_predict, label="train")
    plt.plot(x_test, test_predict, label="val")
    plt.ylabel("Change in average price (\xA3/month)")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    if save:
        utils.savefig(fig, f"{savedir}/obs_vs_pred_obs.png")

    y_train = avg_price[0] + np.cumsum(train_predict)
    y_test = avg_price[len(train[0])] + np.cumsum(test_predict)
    print(f"Train error avg_p_cum: {rmse(avg_price_train, y_train)}")
    print(f"Test error avg_p_cum: {rmse(avg_price_test, y_test)}")
    plt.plot(x_all, avg_price)
    plt.plot(x_train, y_train, label="train")
    plt.plot(x_test, y_test, label="val")
    plt.ylabel("Average price (\xA3)")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    if save:
        utils.savefig(fig, f"{savedir}/forecasted_avg_price_cum.png")

    y_train = avg_price_train[:-1] + train_predict[1:]
    y_test = avg_price_test[:-1] + test_predict[1:]
    print(f"Train error avg_p: {rmse(avg_price_train[:-1], y_train)}")
    print(f"Test error avg_p: {rmse(avg_price_test[:-1], y_test)}")
    plt.plot(x_all, avg_price)
    plt.plot(x_train[1:], y_train, label="train")
    plt.plot(x_test[1:], y_test, label="val")
    plt.ylabel("Average price (\xA3)")
    plt.legend()
    fig = plt.gcf()
    plt.show()
    if save:
        utils.savefig(fig, f"{savedir}/forecasted_avg_price.png")