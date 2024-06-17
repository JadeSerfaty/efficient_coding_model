import os
import boto3

import numpy as np
import pandas as pd
import concurrent.futures
from models.efficient_coding_model import run_efficient_coding_model
from models.choice_model import run_choice_model
from utils.utils import *

DURATION_MAPPING_DICT = {"short": 900, "long": 2600}
S3_BUCKET = "efficient-coding-model"

class ModelRunner:
    def __init__(self, emotion="ANXIETY", duration="LONG", use_mock_data=False, run_choice=False, iteration="v3"):
        self.emotion = emotion.upper()
        self.duration = DURATION_MAPPING_DICT.get(duration.lower(), duration)
        self.use_mock_data = use_mock_data
        self.run_choice = run_choice
        self.iteration = iteration

        base_path = "mock" if use_mock_data else f"main_study/{self.iteration}"

        self.paths = {
            "rating_data": f"data/{base_path}/rating_data.csv",
            "choice_data": f"data/{base_path}/choice_data.csv",
            "ec_model": f"model_outputs/{base_path}/ec_model/",
            "choice_model": f"model_outputs/{base_path}/choice_model/",
            "posterior_distributions": f"{base_path}_{emotion}_{duration}_posterior_distributions.p",
            "choice_model_outputs": f"{base_path}_{emotion}_choice_probs.p"
        }

        self.rating_data = None
        self.choice_data = None
        self.load_data()

        self.login_aws()

    def load_data(self):
        self.rating_data = pd.read_csv(self.paths["rating_data"])
        self.choice_data = pd.read_csv(self.paths["choice_data"])
        self.filter_data()

    def filter_data(self):
        self.rating_data = self.rating_data[self.rating_data["EMOTION_NAME"] == self.emotion].copy()
        self.choice_data = self.choice_data[self.choice_data["EMOTION_NAME"] == self.emotion].copy()

        if self.duration != "both":
            self.rating_data = self.rating_data[self.rating_data['DURATION'] == self.duration]
        else:
            self.rating_data['DURATION_SHORT'] = (self.rating_data['DURATION'] == 900).astype(int)

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
            run_efficient_coding_model, self.rating_data, self.use_mock_data
        )
        with open(os.path.join(self.paths["ec_outputs"], self.paths["posterior_distributions"]), 'wb') as fp:
            pickle.dump(posterior_distributions_all_participants, fp)
        print("Processing posterior distributions completed and results saved successfully.")
        return posterior_distributions_all_participants

    def run_choice_models(self, posterior_distributions_all_participants):
        choice_results = self.run_parallel_models(
            run_choice_model, self.rating_data, self.choice_data, posterior_distributions_all_participants
        )
        with open(os.path.join(self.paths["choice_outputs"], self.paths["choice_model_outputs"]), 'wb') as fp:
            pickle.dump(choice_results, fp)
        print("Processing choice model completed and results saved successfully.")

    def run(self):
        posterior_distributions_all_participants = self.run_efficient_coding_models()
        if self.run_choice:
            self.run_choice_models(posterior_distributions_all_participants)
        else:
            print("Efficient coding model has been processed. Choice model run was skipped.")

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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--run_choice", action="store_true", help="Run the choice model after the efficient coding model")
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data instead of real data")
    parser.add_argument("--emotion", type=str, default="ANXIETY", help="Emotion to filter the data")
    parser.add_argument("--duration", type=str, choices=["short", "long", "both"], default="LONG", help="Duration to filter the data")
    parser.add_argument("--iteration", type=str, default="iter3", help="Iteration of the data collection")
    args = parser.parse_args()

    model_runner = ModelRunner(emotion=args.emotion, duration=args.uration, use_mock_data=args.use_mock_data, run_choice=args.run_choice,
                               iteration=args.iteration)
    model_runner.run()
