import concurrent.futures
from multiprocessing import Manager
import pickle
import pandas as pd

# Functions for efficient coding model
def prepare_data_for_efficient_coding(participant_id, rating_data_emo, epsilon=1e-6):
    participant_emo = rating_data_emo[rating_data_emo["SUBJECT_ID"] == participant_id].copy()

    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    participant_emo.loc[:, 'NORMALIZED_RATING'] = (participant_emo['RATING'] - 1) / (7 - 1)

    # Jitter for every participant
    participant_emo.loc[:, 'NORMALIZED_RATING'] = participant_emo.loc[:, 'NORMALIZED_RATING'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Calculate the mean NORMALIZED_RATING for each VIDEO_ID over both phases
    average_ratings = participant_emo.groupby(['VIDEO_ID', 'PHASE'])['NORMALIZED_RATING'].mean().reset_index()

    # Now, calculate the mean over both phases
    average_ratings_over_phases = average_ratings.groupby('VIDEO_ID')['NORMALIZED_RATING'].mean().reset_index()

    # Rename the column for clarity
    average_ratings_over_phases.rename(columns={'NORMALIZED_RATING': 'NORMALIZED_AVERAGE_RATING'}, inplace=True)

    participant_emo = average_ratings_over_phases

    # Extract the number of videos
    num_videos = len(participant_emo)

    # Calculate new parameters on the normalized scale
    mu_empirical = participant_emo['NORMALIZED_AVERAGE_RATING'].mean()
    s_empirical = participant_emo['NORMALIZED_AVERAGE_RATING'].std()

    print("Estimated Prior Mean:", mu_empirical)
    print("Estimated Prior Standard Deviation:", s_empirical)

    return participant_emo, mu_empirical, s_empirical, num_videos


def prepare_data_for_efficient_coding_all_emotions(participant_id, rating_data_emo, epsilon=1e-6):
    participant_emo = rating_data_emo[rating_data_emo["SUBJECT_ID"] == participant_id].copy()

    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    participant_emo.loc[:, 'NORMALIZED_RATING'] = (participant_emo['RATING'] - 1) / (7 - 1)

    # Jitter for every participant
    participant_emo.loc[:, 'NORMALIZED_RATING'] = participant_emo.loc[:, 'NORMALIZED_RATING'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Sort the data by DURATION_SHORT for simplicity in the outputs of the model
    participant_emo = participant_emo.sort_values(by='DURATION_SHORT', ascending=False).reset_index(drop=True)

    print("the length of participant_emo is:", len(participant_emo))

    # Extract the number of videos
    num_videos = len(participant_emo)

    # Calculate new parameters on the normalized scale
    mu_empirical = participant_emo['NORMALIZED_RATING'].mean()
    s_empirical = participant_emo['NORMALIZED_RATING'].std()

    print("Estimated Prior Mean:", mu_empirical)
    print("Estimated Prior Standard Deviation:", s_empirical)

    return participant_emo, mu_empirical, s_empirical, num_videos


# Functions for choice model
def prepare_data_for_choice_model(data, participant_id, rating_data_emo, choice_data_emo):
    summary_stats = data['summary_stats']
    mu_empirical = summary_stats.loc['mu', 'mean']
    s_empirical = summary_stats.loc['s', 'mean']
    v_values = summary_stats[4:]["mean"]

    participant_emo = rating_data_emo[rating_data_emo["SUBJECT_ID"] == participant_id].copy()
    video_order = participant_emo["VIDEO_ID"].to_list()

    # Ensuring that the length of v_values and video_order are the same
    assert len(v_values) == len(video_order), "Mismatch in lengths of v_values and video_order"

    # Creating dictionary to map video IDs to v values
    videoID_to_v = dict(zip(video_order, v_values))

    participant_choice = choice_data_emo[choice_data_emo["SUBJECT_ID"] == participant_id].copy()
    participant_choice_cleaned = participant_choice.dropna(subset=['CHOSEN_VIDEO_ID', 'NOT_CHOSEN_VIDEO_ID'])

    pairs = []
    observed_choices = []
    choice_probs = []  # List to store choice probabilities

    for index, row in participant_choice_cleaned.iterrows():
        v1 = videoID_to_v[row['CHOSEN_VIDEO_ID']]
        v2 = videoID_to_v[row['NOT_CHOSEN_VIDEO_ID']]
        chosen = int(
            row['CHOSEN_VIDEO_ID'] == row['CHOSEN_VIDEO_ID'])  # This will always be 1 since it's true by definition
        pairs.append((v1, v2))
        observed_choices.append(chosen)

    return pairs, observed_choices, choice_probs, mu_empirical, s_empirical


