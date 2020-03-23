Blueberry forecasting API
====================

Background
----------
<Business context for the model>

Description
-----------
<Description of the inputs, outputs, type of your model etc.>

Getting started
---------------
(NOTE: The following commands assume you are developing locally using using a docker environment.)

Build the publishable docker image by running
```bash
docker-compose -f docker/docker-compose.yml build
```

And then run it with 
```bash
docker-compose -f docker/docker-compose.yml up
```

This creates a container that exposes the application through port 3500. Check that the API is running by doing
We can test that it runs by running 
```bash
curl --request GET   http://0.0.0.0:3500/health
```

```
which should return
```json
{"status":{"code":200,"status":"SUCCESS!"}}
```

For predictions you may use
```bash
curl --header "Content-Type: application/json"   --request POST   --data '{ 
                "samples": [[7.0, 4.0, 7.0, 3.0, 1.0, 4.0],[10.0, 4.0, 7.0, 3.0, 5.0, 6.0]]
          }'   http://0.0.0.0:3500/predict_numbers
```
and expect to receive
```json
{"predictions":[[3.717885971069336,20.47557830810547,37.2332763671875],[5.390043258666992,22.147735595703125,38.905433654785156]]}
```
one row for each sample to predict how many blueberries will be ready to harvest 7 days in the future. The first entry is the value of the 2.5% quantile, the middle is the mean and the last value is the 97.5% quantile.
To find the predicted weights for the samples, execute
```bash
curl --header "Content-Type: application/json"   --request POST   --data '{ 
                "samples": [[7.0, 4.0, 7.0, 3.0, 1.0, 4.0],[10.0, 4.0, 7.0, 3.0, 5.0, 6.0]]
          }'   http://0.0.0.0:3500/predict_weights
```
and expect
```json
{"mean_predictions":[1.1529998407560966,0.9650088971670148],
"stddev_predictions":[1.0225498969839848,0.8030288980399618]}
```
in return.


An explanation of the modelling choices can be found in `Model training.ipynb`, while the final model is trained in `Final model training.ipynb`