"""
Main API entry point.

This file also contains the definition of the flask application that handles Http requests.
"""

# System imports
import logging

# Third-party imports
from flask import Flask, json, jsonify, request, make_response
from flask_cors import CORS
from flask_restful import Resource, Api, reqparse
import tensorflow as tf
import numpy as np

from blueberry_lib import predict, predict_weights, rebuild_model
from constants import GAMMA_PARAMS, MODEL_FILENAME

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


class PredictBlueberryNumbers(Resource):
    """
    Predict the number of blueberries after 7 days that will be ready to harvest.
    """

    @staticmethod
    def post():

        parse = reqparse.RequestParser()
        parse.add_argument('samples')
        tf.set_random_seed(10)
        samples = np.asarray(request.get_json()['samples'])

        with tf.Session() as new_sess:
            # Restore variables from disk.
            loader = tf.train.import_meta_graph(MODEL_FILENAME + '.meta')
            loader.restore(new_sess, MODEL_FILENAME)
            graph = tf.get_default_graph()

            outputs, latent_var_dict, input_ph = rebuild_model(graph)

            predictions = predict(samples, outputs, latent_var_dict, input_ph)

        return make_response(jsonify({'predictions': predictions.tolist()}))


class PredictBlueberryWeight(Resource):
    """
    Predict the weights of the blueberries that will be ready to harvest in 7 days.
    """

    @staticmethod
    def post():
        parse = reqparse.RequestParser()
        parse.add_argument('samples')
        tf.set_random_seed(10)
        samples = np.asarray(request.get_json()['samples'])

        with tf.Session() as new_sess:
            # Restore variables from disk.
            fname = './test.cpkt'
            loader = tf.train.import_meta_graph(MODEL_FILENAME + '.meta')
            loader.restore(new_sess, MODEL_FILENAME)
            graph = tf.get_default_graph()

            outputs, latent_var_dict, input_ph = rebuild_model(graph)

            predictions = predict_weights(samples, GAMMA_PARAMS, outputs, latent_var_dict, input_ph)

        return make_response(jsonify({'mean_predictions': predictions[0].tolist(),
                                      'stddev_predictions': predictions[1].tolist()}))


api.add_resource(PredictBlueberryNumbers, '/predict_numbers')
api.add_resource(PredictBlueberryWeight, '/predict_weights')

if __name__ == '__main__':
     app.run(host='0.0.0.0', port=3500)
