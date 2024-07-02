import concurrent.futures
from multiprocessing import Manager
import pickle
import pandas as pd
import numpy as np

# Functions for efficient coding model
def prepare_data_for_efficient_coding(rating_data, epsilon=1e-6):
    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    rating_data.loc[:, 'NORMALIZED_RATING'] = (rating_data['RATING'] - 1) / (7 - 1)

    # Jitter for every participant
    rating_data.loc[:, 'NORMALIZED_RATING'] = rating_data.loc[:, 'NORMALIZED_RATING'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Calculate the mean NORMALIZED_RATING for each VIDEO_ID over both phases
    average_ratings = rating_data.groupby(['VIDEO_ID', 'PHASE'])['NORMALIZED_RATING'].mean().reset_index()

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


def prepare_data_for_efficient_coding_all_emotions(rating_data, duration, epsilon=1e-6):
    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    rating_data.loc[:, 'NORMALIZED_RATING'] = (rating_data['RATING'] - 1) / (7 - 1)

    # Jitter for every participant
    rating_data.loc[:, 'NORMALIZED_RATING'] = rating_data.loc[:, 'NORMALIZED_RATING'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Sort the data by DURATION in descending order to facilitate reading of outputs of model
    rating_data_emo = rating_data.sort_values(by='DURATION', ascending=False).reset_index(drop=True)

    # Calculate the mean NORMALIZED_RATING for each VIDEO_ID over both durations
    mu_empirical = rating_data_emo['NORMALIZED_RATING'].mean()
    s_empirical = rating_data_emo['NORMALIZED_RATING'].std()

    # Filter the data by duration to fit the model on each duration separately
    if duration.size == 1:
        rating_data_emo = rating_data_emo[rating_data_emo['DURATION'] == duration[0]]

    print("the length of rating_data_emo is:", len(rating_data_emo))

    # Extract the number of videos
    num_videos = len(np.unique(rating_data_emo["VIDEO_ID"]))
    print(num_videos)

    print("Estimated Prior Mean:", mu_empirical)
    print("Estimated Prior Standard Deviation:", s_empirical)

    return rating_data_emo, mu_empirical, s_empirical, num_videos


# Functions for choice model
def prepare_data_for_choice_model(posterior_distributions, rating_data, choice_data):
    summary_stats = posterior_distributions['summary_stats']
    mu_empirical = summary_stats.loc['mu', 'mean']
    s_empirical = summary_stats.loc['s', 'mean']
    v_values = summary_stats[4:]["mean"]

    video_order = rating_data["VIDEO_ID"].to_list()

    # Ensuring that the length of v_values and video_order are the same
    assert len(v_values) == len(video_order), "Mismatch in lengths of v_values and video_order"

    # Creating dictionary to map video IDs to v values
    videoID_to_v = dict(zip(video_order, v_values))

    participant_choice_cleaned = choice_data.dropna(subset=['CHOSEN_VIDEO_ID', 'NOT_CHOSEN_VIDEO_ID'])

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


