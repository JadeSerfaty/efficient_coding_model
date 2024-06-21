import json
import os
import sys
import boto3
import pickle
import numpy as np
import pandas as pd
import concurrent.futures
from multiprocessing import Manager
from src.models.efficient_coding_model import run_efficient_coding_model, run_separate_sigma_model
from src.models.choice_model import run_choice_model
from src.utils.utils import *

S3_BUCKET = "efficient-coding-model"
AWS_ACCESS_KEY_ID="AKIAZI2LH2U5KTZBDBVB"
AWS_SECRET_ACCESS_KEY="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"

class ModelRunner:
    def __init__(self, choice_data_json, rating_data_json, job_name="default_job", run_choice=False, run_name="default_run"):
        self.choice_data_json = choice_data_json
        self.rating_data_json = rating_data_json
        self.job_name = job_name
        self.run_choice = run_choice

        self.rating_data = pd.read_csv(self.rating_data_json)
        self.choice_data = pd.read_csv(self.choice_data_json)
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

    def run_parallel_models(self, model_func, *model_args):
        all_participant_ids = np.unique(self.rating_data["SUBJECT_ID"])
        with Manager() as manager:
            results = manager.dict()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(model_func, results, participant_id, *model_args)
                    for participant_id in all_participant_ids
                ]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")
            return dict(results)

    def run_efficient_coding_models(self):
        posterior_distributions_all_participants = self.run_parallel_models(
            run_separate_sigma_model, self.rating_data
        )
        self.upload_to_s3(posterior_distributions_all_participants, self.paths["posterior_distributions"])
        print(f"Processing posterior distributions for {self.emotion} and {self.duration} completed and results saved successfully.")
        return posterior_distributions_all_participants

    def run_choice_models(self, posterior_distributions_all_participants):
        choice_results = self.run_parallel_models(
            run_choice_model, self.rating_data, self.choice_data, posterior_distributions_all_participants
        )
        self.upload_to_s3(choice_results, self.paths["choice_model_outputs"])
        print(f"Processing choice model for {self.emotion} and {self.duration} completed and results saved successfully.")

    def run(self):
        posterior_distributions_all_participants = self.run_efficient_coding_models()
        if self.run_choice:
            self.run_choice_models(posterior_distributions_all_participants)
        else:
            print(f"Efficient coding model for {self.emotion} and {self.duration} has been processed. Choice model run was skipped.")

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
    combinations = json.loads(os.getenv("COMBINATIONS"))
    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    combination = combinations[array_index]

    model_runner = ModelRunner(
        choice_data_json=combination['choice_data'],
        rating_data_json=combination['rating_data'],
        job_name=combination['job_name'],
        run_name=combination['run_name']
    )
