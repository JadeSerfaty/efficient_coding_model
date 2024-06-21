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
        self.s3_client.upload_to_s3(posterior_distributions_all_participants, self.paths["posterior_distributions"])
        print(f"Processing posterior distributions for {self.emotion} and {self.duration} completed and results saved successfully.")
        return posterior_distributions_all_participants

    def run_choice_models(self, posterior_distributions_all_participants):
        choice_results = self.run_parallel_models(
            run_choice_model, self.rating_data, self.choice_data, posterior_distributions_all_participants
        )
        self.s3_client.upload_to_s3(choice_results, self.paths["choice_model_outputs"])
        print(f"Processing choice model for {self.emotion} and {self.duration} completed and results saved successfully.")

    def run(self):
        posterior_distributions_all_participants = self.run_efficient_coding_models()
        if self.run_choice:
            self.run_choice_models(posterior_distributions_all_participants)
        else:
            print(f"Efficient coding model for {self.emotion} and {self.duration} has been processed. Choice model run was skipped.")

if __name__ == "__main__":
    # Fetch environment variables
    task_configs = json.loads(os.getenv("COMBINATIONS"))
    array_index = int(os.getenv("AWS_BATCH_JOB_ARRAY_INDEX", 0))

    task_config = task_configs[array_index]

    model_runner = ModelRunner(task_config)
    model_runner.run()
