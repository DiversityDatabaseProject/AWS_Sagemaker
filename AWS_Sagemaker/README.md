This document describes the deployment of our face detection model with AWS Sagemaker.

# AWS Sagemaker intro

Amazon SageMaker makes use of endpoints. Endpoints are a fully managed service that allows you to make real-time inferences via a REST API. With AWS Sagemaker, there is no need to run your own EC2 instance.

# AWS Sagemaker notebook setup

The Sagemaker notebook was created so it performs the following steps:

1. Import some original pictures stored in S3, together with the Tensorflow trained .pb model
2. Run the inference
3. Returns labelled picture and confidence score

It makes use of a number of files:
- environment.yml: it is the file that creates the virtual environment and import all packages necessary to running the inference.
- sample_utils.py: this file enables the conversion from the picture format to tensor format so it can subsequently be used by Tensorflow.
- inference.py: it is the file that runs the model and performs the inference

The notebook was mainly created with demo in mind, which is why it displays the picture before and after inference.
