
import edward.models as edm
import edward as ed
import numpy as np
import tensorflow as tf
from scipy.stats import norm, gamma

from constants import PHASES, PHASE_SHIFT_DAYS, N, SAMPLES_WEIGHTS


def get_shift_dist(n, phase_mean, phase_stddev, days=7.):
    """
    Get the parameters of a distribution that describes how many blueberries are expected to shift out of that phase in the next d days.
    :param n: number of blueberries to consider
    :param phase_mean: Average number of days that blueberry will spend in this phase
    :param phase_stddev: Standard deviation of days that blueberry will spend in this phase
    :param days: Number of days
    :return: expected value for the number of blueberries that will shift in phase
    """
    p = norm.cdf(days, phase_mean, phase_stddev)
    return n * p.astype('float32')


def calc_accuracy(samples, predictions):
    """
    Calculate the accuracy by counting the number of samples which fall in the prediction interval.
    :param samples: Ground truth
    :param predictions: Model predictions
    :return: Accuracy
    """
    return ((samples > predictions[:, 0]) & (samples < predictions[:, 2])).sum() / len(samples) * 100.


def build_model():
    """
    Build the model.
    :return: Tensors for the inputs, outputs, growths and shifts in each phase
    """

    # Initialize
    inputs = [0] * PHASES  # Number of blueberries in each phase at time t
    shifts = [0] * PHASES  # Number of blueberries that will leave the the phase by time t + 1
    grow = [0] * PHASES  # Number of blueberries that will arrive at phase by time t + 1
    outputs = [0] * PHASES # Number of blueberries in each phase at time t + 1
    for i in range(PHASES):
        inputs[i] = edm.Normal(20., 7., sample_shape=[N])
        shift_mean = get_shift_dist(inputs[i], PHASE_SHIFT_DAYS[i], PHASE_SHIFT_DAYS[i]/2.)
        shifts[i] = edm.Normal(shift_mean, PHASE_SHIFT_DAYS[i]/2.)
        grow[i] = shifts[i - 1] if i > 0 else edm.Normal(20., 7., sample_shape=[N])
        outputs[i] = edm.Normal(inputs[i] - shifts[i] + grow[i], inputs[i].scale + shifts[i].scale + grow[i].scale)
    return inputs, shifts, grow, outputs


def predict(samples, outputs, latent_var_dict, input_ph):
    """

    :param samples: Data to score
    :param outputs: Tensor that represents the outputs
    :param latent_var_dict: Dictionary that contains the latent variables in the model.
    :param input_ph: Placeholder for the inputs
    :return: Predictions
    """
    x_post = ed.copy(outputs[-1], latent_var_dict)
    sess = ed.get_session()
    predictions = np.zeros((samples.shape[0], 3))
    for i in range(0, samples.shape[0]):
        feed_dict = {}
        feed_dict.update({key: [value] for key, value in zip(input_ph, samples[i, :])})
        quantile_1, quantile_2, mean = sess.run([x_post.quantile(0.025), x_post.quantile(0.975), x_post.mean()],
                                                feed_dict=feed_dict)
        predictions[i, :] = [quantile_1, mean, quantile_2]

    return predictions


def predict_weights(samples, gamma_params, outputs, latent_var_dict, input_ph):
    """

    :param samples: Data to score
    :param gamma_params: Parameters of the gamma distribution associated with the berry weight distribution
    :param outputs: Tensor that represents the outputs
    :param latent_var_dict: Dictionary that contains the latent variables in the model.
    :param input_ph: Placeholder for the inputs
    :return:
    """
    predictions = np.zeros((samples.shape[0], SAMPLES_WEIGHTS))
    for i in range(0, SAMPLES_WEIGHTS):
        print('Sampling from the blueberry number distribution, iteration:', i)
        predictions[:, i] = predict(samples, outputs, latent_var_dict, input_ph)[:, 1]
    weight_samples = gamma(gamma_params[0], gamma_params[1], gamma_params[2]).rvs((samples.shape[0], SAMPLES_WEIGHTS))
    weight_predictions = predictions * weight_samples
    mean_weights = np.mean(weight_predictions, axis=1)
    stddev_weights = np.std(weight_predictions, axis=1)
    return mean_weights, stddev_weights


def rebuild_model(graph):
    """
    Rebuild the model from a tensorflow graph
    :param graph: Tensorflow graph
    :return: Tensor that represents the outputs, Dictionary that contains the latent variables in the model, Placeholder for the inputs
    """
    # Create model
    inputs, shifts, grow, outputs = build_model()

    # populate with reloaded data
    q_grow = edm.Normal(loc=graph.get_tensor_by_name('posterior/Normal/loc:0'),
                        scale=graph.get_tensor_by_name('posterior/Normal/scale:0'), sample_shape=[N])
    latent_vars = [0] * PHASES
    input_ph = [0] * PHASES
    for i in range(PHASES):
        input_ph[i] = tf.placeholder(tf.float32, shape=[N])
        latent_vars[i] = edm.Normal(
            loc=graph.get_tensor_by_name('posterior/Normal_' + str(i + 1) + '/loc:0'),
            scale=graph.get_tensor_by_name(
                'posterior/Normal_' + str(i + 1) + '/scale:0'), sample_shape=[N])

    latent_var_dict = {grow[0]: q_grow}
    latent_var_dict.update({key: value for key, value in zip(shifts, latent_vars)})

    return outputs, latent_var_dict, input_ph
