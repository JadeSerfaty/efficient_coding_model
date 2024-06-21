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
from utils.s3_client import S3Client

S3_BUCKET = "efficient-coding-model"

class MockModelRunner:
    def __init__(self, task_config):
        self.choice_data_json = task_config['choice_data']
        self.rating_data_json = task_config['rating_data']
        self.job_name = task_config['job_name']
        self.run_choice = task_config.get('run_choice', False)
        self.run_name = task_config.get('run_name', "default_run")

        self.rating_data = pd.read_json(self.rating_data_json, orient='split')
        self.choice_data = pd.read_json(self.choice_data_json, orient='split')
        self.emotion = self.rating_data['EMOTION_NAME'].iloc[0]
        self.duration = self.rating_data['DURATION'].iloc[0]

        self.paths = {
            "posterior_distributions": f"{self.run_name}/{self.job_name}/{self.emotion}_{self.duration}_posterior_distributions.p",
            "choice_model_outputs": f"{self.run_name}/{self.job_name}/{self.emotion}_choice_probs.p"
        }

        self.s3_client = S3Client()

        mock_data = pd.DataFrame()
        self.s3_client.upload_to_s3(mock_data, self.paths["posterior_distributions"])
        self.s3_client.upload_to_s3(mock_data, self.paths["choice_model_outputs"])


if __name__ == "__main__":
    # Fetch environment variables
    task_configs = json.loads(os.getenv("COMBINATIONS"))
    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    task_config = task_configs[array_index]

    model_runner = MockModelRunner(task_config)