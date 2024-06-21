import os
import sys
import json
import boto3
import pickle
import numpy as np
import pandas as pd
from src.models.efficient_coding_model import run_efficient_coding_model, run_separate_sigma_model
from src.models.choice_model import run_choice_model
from src.utils.utils import *

S3_BUCKET = "efficient-coding-model"


class MockModelRunner:
    def __init__(self, choice_data_json, rating_data_json, job_name="default_job", run_choice=False, run_name="default_run"):
        self.choice_data_json = choice_data_json
        self.rating_data_json = rating_data_json
        self.job_name = job_name
        self.run_choice = run_choice

        self.rating_data = pd.read_json(self.rating_data_json, orient='split')
        self.choice_data = pd.read_json(self.choice_data_json, orient='split')
        self.emotion = self.rating_data['EMOTION_NAME'].iloc[0]
        self.duration = self.rating_data['DURATION'].iloc[0]

        self.paths = {
            "posterior_distributions": f"{run_name}/{self.job_name}/{self.emotion}_{self.duration}_posterior_distributions.p",
            "choice_model_outputs": f"{run_name}/{self.job_name}/{self.emotion}_choice_probs.p"
        }

        self.login_aws()

        mock_data = pd.DataFrame()
        self.upload_to_s3(mock_data, self.paths["posterior_distributions"])
        self.upload_to_s3(mock_data, self.paths["choice_model_outputs"])

    def upload_to_s3(self, data, key):
        pickle_data = pickle.dumps(data)
        self.s3_client.put_object(Bucket=S3_BUCKET, Key=key, Body=pickle_data)

    def login_aws(self):
        print("Login to AWS S3")
        self.s3_client = boto3.client(
            service_name="s3",
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
        print("AWS S3 login successful")


if __name__ == "__main__":
    # Fetch environment variables
    combinations = json.loads(os.getenv("COMBINATIONS"))
    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    combination = combinations[array_index]

    model_runner = MockModelRunner(
        choice_data_json=combination['choice_data'],
        rating_data_json=combination['rating_data'],
        job_name=combination['job_name'],
        run_name=combination['run_name']
    )

