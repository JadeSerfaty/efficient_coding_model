import pandas as pd
import pickle
import concurrent.futures
from efficient_coding_model import run_efficient_coding_model
from choice_model import run_choice_model
from multiprocessing import Manager
from utils import *

class ModelRunner:
    def __init__(self, emotion="anxiety", duration="long", use_mock_data=False, run_choice=False):
        self.emotion = emotion
        self.duration = duration
        self.use_mock_data = use_mock_data
        self.run_choice = run_choice
        self.main_path_rating_data = "data_collection/main_study/v1/rating_data.csv"
        self.main_path_choice_data = "data_collection/main_study/v1/choice_data.csv"
        self.main_path_EC_outputs = "model_outputs/main_study/v2/outputs_EC_model_each_emotion/"
        self.main_path_choice_outputs = "model_outputs/main_study/v2/outputs_choice_model_each_emotion/"
        self.name_file_posterior_distributions = f"main_study_{emotion}_{duration}_duration_posterior_distributions.p"
        self.name_file_choice_model_outputs = f"main_study_{emotion}_choice_probs.p"
        self.rating_data = None
        self.choice_data = None
        self.load_data()

    def load_data(self):
        if self.use_mock_data:
            self.rating_data = pd.read_csv("data_collection/mock/rating_data.csv")
            self.choice_data = pd.read_csv("data_collection/mock/choice_data.csv")
            self.main_path_EC_outputs = "model_outputs/mock/outputs_EC_model/"
            self.main_path_choice_outputs = "model_outputs/mock/outputs_choice_model/"
            self.name_file_posterior_distributions = f"mock_posterior_distributions.p"
            self.name_file_choice_model_outputs = "mock_choice_probs.p"

        else:
            self.rating_data = pd.read_csv(self.main_path_rating_data)
            self.choice_data = pd.read_csv(self.main_path_choice_data)
            self.filter_data()

    def filter_data(self):
        self.rating_data = self.rating_data[self.rating_data["emotionName"] == self.emotion].copy()
        self.choice_data = self.choice_data[self.choice_data["emotionName"] == self.emotion].copy()

        if self.duration == "short":
            rating_data_phase1 = self.rating_data[self.rating_data['durationBlackScreen_phase1'] == 900][
                ['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating', 'variance_rating', 'subject_id']]
            rating_data_phase2 = self.rating_data[self.rating_data['durationBlackScreen_phase2'] == 900][
                ['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating', 'variance_rating', 'subject_id']]
        elif self.duration == "long":
            rating_data_phase1 = self.rating_data[self.rating_data['durationBlackScreen_phase1'] == 2600][
                ['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating', 'variance_rating', 'subject_id']]
            rating_data_phase2 = self.rating_data[self.rating_data['durationBlackScreen_phase2'] == 2600][
                ['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating', 'variance_rating', 'subject_id']]

        self.rating_data = pd.concat([rating_data_phase1.rename(columns={'rating_phase1': 'rating', 'durationBlackScreen_phase1': 'durationBlackScreen'}),
                                      rating_data_phase2.rename(columns={'rating_phase2': 'rating', 'durationBlackScreen_phase2': 'durationBlackScreen'})], ignore_index=True)

    def run_efficient_coding_models(self):
        all_participant_ids = np.unique(self.rating_data["subject_id"])
        # Use Manager to create a shared dictionary
        with Manager() as manager:
            posterior_distributions_all_participants = manager.dict()
            # Run models in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_efficient_coding_model, posterior_distributions_all_participants, participant_id, self.rating_data, self.use_mock_data) for participant_id in all_participant_ids]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")
            # Convert shared dictionary to a regular dictionary before saving
            posterior_distributions_all_participants = dict(posterior_distributions_all_participants)

            # Save the posterior distributions
            with open(self.main_path_EC_outputs + self.name_file_posterior_distributions, 'wb') as fp:
                pickle.dump(posterior_distributions_all_participants, fp)

            print("Processing posterior distributions completed and results saved successfully.")
            return posterior_distributions_all_participants

    def run_choice_models(self, posterior_distributions_all_participants):
        with Manager() as manager:
            choice_results = manager.dict()
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [executor.submit(run_choice_model, participant_id, data, self.rating_data, self.choice_data, choice_results) for participant_id, data in posterior_distributions_all_participants.items()]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert shared dictionary to a regular dictionary before saving
            choice_results = dict(choice_results)

            # Save the posterior distributions
            with open(self.main_path_choice_outputs + self.name_file_choice_model_outputs, 'wb') as fp:
                pickle.dump(choice_results, fp)
            print("Processing choice model completed and results saved successfully.")

    def run(self):
        posterior_distributions_all_participants = self.run_efficient_coding_models()
        if self.run_choice:
            self.run_choice_models(posterior_distributions_all_participants)
        else:
            print("Efficient coding model has been processed. Choice model run was skipped.")
