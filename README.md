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
