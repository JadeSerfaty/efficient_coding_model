import pickle
import concurrent.futures
from efficient_coding_model import *
from choice_model import *
from multiprocessing import Manager
import argparse


def main(run_choice=False, use_mock_data=False):
    emotion = "anxiety"
    duration = "long"
    # Define paths and other parameters
    main_path_rating_data = "data_collection/main_study/v1/rating_data.csv"
    main_path_choice_data = "data_collection/main_study/v1/choice_data.csv"
    name_file_posterior_distributions = f"main_study_{emotion}_{duration}_duration_posterior_distributions.p"
    name_file_choice_model_outputs = "choice_model_outputs.pkl"

    if use_mock_data:
        rating_data = pd.read_csv("data_collection/mock/rating_data.csv")
        choice_data = pd.read_csv("data_collection/mock/choice_data.csv")
    else:
        # Load data
        rating_data = pd.read_csv(main_path_rating_data)
        choice_data = pd.read_csv(main_path_choice_data)

        # Filter the data for the specific emotion
        rating_data = rating_data[rating_data["emotionName"] == emotion].copy()
        choice_data = choice_data[choice_data["emotionName"] == emotion].copy()

        # Filter the rating data on the duration
        if duration == "short":
            rating_data_phase1 = rating_data[rating_data['durationBlackScreen_phase1'] == 900][
                ['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating',
                 'variance_rating',
                 'subject_id']]
            rating_data_phase2 = rating_data[rating_data['durationBlackScreen_phase2'] == 900][
                ['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating',
                 'variance_rating',
                 'subject_id']]

        elif duration == "long":
            rating_data_phase1 = rating_data[rating_data['durationBlackScreen_phase1'] == 2600][
                ['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating',
                 'variance_rating',
                 'subject_id']]
            rating_data_phase2 = rating_data[rating_data['durationBlackScreen_phase2'] == 2600][
                ['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating',
                 'variance_rating',
                 'subject_id']]

        rating_data = pd.concat([rating_data_phase1.rename(
            columns={'rating_phase1': 'rating', 'durationBlackScreen_phase1': 'durationBlackScreen'}),
            rating_data_phase2.rename(columns={'rating_phase2': 'rating',
                                               'durationBlackScreen_phase2': 'durationBlackScreen'})],
            ignore_index=True)

    all_participant_ids = np.unique(rating_data["subject_id"])

    # Use Manager to create a shared dictionary
    with Manager() as manager:
        posterior_distributions_all_participants = manager.dict()

        # Run models in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_efficient_coding_model, posterior_distributions_all_participants,
                                       participant_id, rating_data)
                       for participant_id in all_participant_ids]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error: {e}")

        # Convert shared dictionary to a regular dictionary before saving
        posterior_distributions_all_participants = dict(posterior_distributions_all_participants)

        # Save the posterior distributions
        with open(name_file_posterior_distributions, 'wb') as fp:
            pickle.dump(posterior_distributions_all_participants, fp)

        print("Processing posterior distributions completed and results saved successfully.")

        if run_choice:
            choice_results = manager.dict()

            # Run models in parallel
            with concurrent.futures.ProcessPoolExecutor() as executor:
                futures = [
                    executor.submit(run_choice_model, participant_id, data, rating_data, choice_data,
                                    choice_results)
                    for participant_id, data in posterior_distributions_all_participants.items()]
                for future in concurrent.futures.as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error: {e}")

            # Convert shared dictionary to a regular dictionary before saving
            choice_results = dict(choice_results)

            # Save the posterior distributions
            with open(name_file_choice_model_outputs, 'wb') as fp:
                pickle.dump(choice_results, fp)

            print("Processing choice model completed and results saved successfully.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_choice", action="store_true",
                        help="Run the choice model after the efficient coding model")
    parser.add_argument("--use_mock_data", action="store_true", help="Use mock data instead of real data")
    args = parser.parse_args()
    main(run_choice=args.run_choice, use_mock_data=args.use_mock_data)