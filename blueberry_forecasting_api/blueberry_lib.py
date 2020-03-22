
import edward.models as edm
import edward as ed
import numpy as np
from scipy.stats import norm

from constants import PHASES, PHASE_SHIFT_DAYS, N


def get_shift_dist(n, phase_mean, phase_stddev, days=7.):
    p = norm.cdf(days, phase_mean, phase_stddev)
    return n * p.astype('float32')


def calc_accuracy(samples, predictions):
    return ((samples > predictions[:, 0]) & (samples < predictions[:, 2])).sum() / len(samples) * 100.


def build_model():

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
    x_post = ed.copy(outputs[-1], latent_var_dict)
    sess = ed.get_session()
    print('in the predict method')
    print(samples)
    print(samples.shape)
    predictions = np.zeros((samples.shape[0], 3))
    for i in range(0, samples.shape[0]):
        feed_dict = {}
        feed_dict.update({key: [value] for key, value in zip(input_ph, samples[i, :])})
        quantile_1, quantile_2, mean = sess.run([x_post.quantile(0.025), x_post.quantile(0.975), x_post.mean()],
                                                feed_dict=feed_dict)
        predictions[i, :] = [quantile_1, mean, quantile_2]

    return predictions
