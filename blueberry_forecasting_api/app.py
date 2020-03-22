"""
Main API entry point.

This file also contains the definition of the flask application that handles Http requests.
"""
# Pylint does not recognize the predict() function as being used - so switch this check off in this module.
# pylint: disable=unused-variable

# System imports
import logging

# Third-party imports
from flask import Flask, json, jsonify, request, make_response
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
import tensorflow as tf
import edward.models as edm
import numpy as np

from blueberry_lib import build_model, predict
from constants import PHASES, N

LOGGER = logging.getLogger(__name__)

app = Flask(__name__)
api = Api(app)
CORS(app)


@app.route("/health")
def health():
    """
    Expose an endpoint that can be used to check the health of the API.
    """
    return jsonify({
        "status": {
            "code": 200,
            "status": "SUCCESS!"
        }
    })


class PredictBlueberry(Resource):

    @staticmethod
    def post():

        parse = reqparse.RequestParser()
        parse.add_argument('samples')

        args = parse.parse_args()
        samples = np.asarray(args['samples'])

        samples = np.asarray(request.get_json()['samples'])

        with tf.Session() as new_sess:
            # Restore variables from disk.
            fname = './posterior.ckpt'
            loader = tf.train.import_meta_graph(fname + '.meta')
            loader.restore(new_sess, fname)
            graph = tf.get_default_graph()

            # Create model
            inputs, shifts, grow, outputs = build_model()

            # populate with reloaded data
            q_grow = edm.Normal(loc=graph.get_tensor_by_name('posterior/Normal/loc:0'),
                                scale=graph.get_tensor_by_name('posterior/Normal/scale:0'), sample_shape=[N])
            latent_vars = [0] * PHASES
            input_ph = [0] * PHASES
            for i in range(PHASES):
                input_ph[i] = tf.placeholder(tf.float32, shape=[N])
                latent_vars[i] = edm.Normal(loc=graph.get_tensor_by_name('posterior/Normal_' + str(i + 1) + '/loc:0'),
                                            scale=graph.get_tensor_by_name(
                                                'posterior/Normal_' + str(i + 1) + '/scale:0'), sample_shape=[N])

            latent_var_dict = {grow[0]: q_grow}
            latent_var_dict.update({key: value for key, value in zip(shifts, latent_vars)})

            predictions = predict(samples, outputs, latent_var_dict, input_ph)

        return make_response(jsonify({'predictions': predictions.tolist()}))


api.add_resource(PredictBlueberry, '/predict')

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=3500)
