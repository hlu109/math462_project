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

# Randomly initialize the initial state distribution
initial_logits = tf.Variable(rng.random([num_states]),
                             name="initial_logits",
                             dtype=tf.float32)
initial_distribution = tfd.Categorical(logits=initial_logits)

# Create state-dependent observation distribution