import os
import json
import numpy as np
import pandas as pd
import concurrent.futures
from multiprocessing import Manager
from src.models.efficient_coding_model import run_efficient_coding_model, run_separate_sigma_model
from src.models.choice_model import run_choice_model
from src.utils.s3_client import S3Client


class ModelRunner:
    def __init__(self, task_config):
        print("Starting model runner")
        print(task_config)
        self.rating_data_json = task_config['rating_data']
        self.job_name = task_config['job_name']
        self.run_choice = task_config['run_choice']
        self.run_name = task_config.get('run_name', "default_run")

        self.choice_data_json = task_config['choice_data']
        if self.choice_data_json:
            self.choice_data = pd.read_json(self.choice_data_json, orient='split')

        self.rating_data = pd.read_json(self.rating_data_json, orient='split')
        self.emotion = self.rating_data['EMOTION_NAME'].iloc[0]
        # self.duration = self.rating_data['DURATION'].unique()
        self.duration = np.array([task_config["duration"]])
        print(self.duration)
        self.subject_id = self.rating_data['SUBJECT_ID'].unique()
        if self.subject_id.size > 1:
            raise ValueError("Only one subject_id is supported at the moment.")

        if self.duration.size > 1:
            title_duration = "all"
        else:
            title_duration = f"{self.duration[0]}"
        self.paths = {
            "posterior_distributions": f"{self.run_name}/{self.job_name}/{self.emotion}_{title_duration}_posterior_distributions.p",
            "choice_model_outputs": f"{self.run_name}/{self.job_name}/{self.emotion}_choice_probs.p"
        }

        self.s3_client = S3Client()

        print(f"model started for {task_config['job_name']}, {self.emotion}, {self.duration} and {self.run_name}")

    def run_efficient_coding_models(self):
        if self.duration.size > 1:
            print("Running separate sigma model")
            posterior_distributions = run_separate_sigma_model(rating_data=self.rating_data, duration=self.duration)
        else:
            print("Running efficient coding model")
            posterior_distributions = run_efficient_coding_model(rating_data=self.rating_data, duration=self.duration)
        self.s3_client.upload_pickle_to_s3(posterior_distributions, self.paths["posterior_distributions"])
        print(
            f"Processing posterior distributions for participant: {self.subject_id} and {self.emotion} and {self.duration} completed and results saved successfully.")
        return posterior_distributions

    def run_choice_models(self, posterior_distributions):
        choice_results = run_choice_model(self.rating_data, self.choice_data, posterior_distributions)
        self.s3_client.upload_pickle_to_s3(choice_results, self.paths["choice_model_outputs"])
        print(f"Processing choice model for {self.emotion} and {self.duration} completed and results saved successfully.")

    def run(self):
        posterior_distributions_all_participants = self.run_efficient_coding_models()
        if self.run_choice:
            self.run_choice_models(posterior_distributions_all_participants)
        else:
            print(
                f"Efficient coding model for {self.emotion} and {self.duration} has been processed. Choice model run was skipped.")


if __name__ == "__main__":
    # Fetch environment variables
    s3_client = S3Client()
    task_configs = s3_client.get_json_file_from_s3(os.getenv("COMBINATIONS_KEY_S3"))
    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    task_config = task_configs[array_index]

    model_runner = ModelRunner(task_config)
    model_runner.run()
