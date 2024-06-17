import concurrent.futures
from multiprocessing import Manager
import pickle

# Functions for efficient coding model
def prepare_data_for_efficient_coding(participant_id, rating_data_emo, epsilon=1e-6):
    participant_emo = rating_data_emo[rating_data_emo["subject_id"] == participant_id].copy()

    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    participant_emo.loc[:, 'normalized_rating_phase1'] = (participant_emo['rating_phase1'] - 1) / (7 - 1)
    participant_emo.loc[:, 'normalized_rating_phase2'] = (participant_emo['rating_phase2'] - 1) / (7 - 1)

    # Jitter for every participant
    participant_emo.loc[:, 'normalized_rating_phase1'] = participant_emo.loc[:, 'normalized_rating_phase1'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )
    participant_emo.loc[:, 'normalized_rating_phase2'] = participant_emo.loc[:, 'normalized_rating_phase2'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Compute the average of the normalized ratings
    participant_emo.loc[:, 'normalized_average_rating'] = participant_emo.loc[:, ['normalized_rating_phase1',
                                                                                  'normalized_rating_phase2']].mean(
        axis=1)

    # Compute the variability of the normalized ratings
    participant_emo.loc[:, 'normalized_variance_rating'] = participant_emo.loc[:, ['normalized_rating_phase1',
                                                                                   'normalized_rating_phase2']].std(
        axis=1)

    # Extract the number of videos
    num_videos = len(participant_emo)

    # Calculate new parameters on the normalized scale
    mu_empirical = participant_emo['normalized_average_rating'].mean()
    s_empirical = participant_emo['normalized_average_rating'].std()

    print("Estimated Prior Mean:", mu_empirical)
    print("Estimated Prior Standard Deviation:", s_empirical)

    return participant_emo, mu_empirical, s_empirical, num_videos


def prepare_data_for_efficient_coding_all_emotions(participant_id, rating_data_emo, epsilon=1e-6):
    participant_emo = rating_data_emo[rating_data_emo["subject_id"] == participant_id].copy()

    # Define the parameters for the prior distribution
    # Normalize the 'average_rating' to a 0-1 scale
    participant_emo.loc[:, 'normalized_rating'] = (participant_emo['rating'] - 1) / (7 - 1)

    # Jitter for every participant
    participant_emo.loc[:, 'normalized_rating'] = participant_emo.loc[:, 'normalized_rating'].apply(
        lambda x: 0 + epsilon if x == 0 else (1 - epsilon if x == 1 else x)
    )

    # Extract the number of videos
    num_videos = len(participant_emo)

    # Calculate new parameters on the normalized scale
    mu_empirical = participant_emo['normalized_rating'].mean()
    s_empirical = participant_emo['normalized_rating'].std()

    print("Estimated Prior Mean:", mu_empirical)
    print("Estimated Prior Standard Deviation:", s_empirical)

    return participant_emo, mu_empirical, s_empirical, num_videos


# Functions for choice model
def prepare_data_for_choice_model(data, participant_id, rating_data_emo, choice_data_emo):
    summary_stats = data['summary_stats']
    mu_empirical = summary_stats.loc['mu', 'mean']
    s_empirical = summary_stats.loc['s', 'mean']
    v_values = summary_stats[4:]["mean"]

    participant_emo = rating_data_emo[rating_data_emo["subject_id"] == participant_id].copy()
    video_order = participant_emo["videoID"].to_list()

    # Ensuring that the length of v_values and video_order are the same
    assert len(v_values) == len(video_order), "Mismatch in lengths of v_values and video_order"

    # Creating dictionary to map video IDs to v values
    videoID_to_v = dict(zip(video_order, v_values))

    participant_choice = choice_data_emo[choice_data_emo["subject_id"] == participant_id].copy()
    participant_choice_cleaned = participant_choice.dropna(subset=['chosenVideoID', 'notChosenVideoID'])

    pairs = []
    observed_choices = []
    choice_probs = []  # List to store choice probabilities

    for index, row in participant_choice_cleaned.iterrows():
        v1 = videoID_to_v[row['chosenVideoID']]
        v2 = videoID_to_v[row['notChosenVideoID']]
        chosen = int(
            row['chosenVideoID'] == row['chosenVideoID'])  # This will always be 1 since it's true by definition
        pairs.append((v1, v2))
        observed_choices.append(chosen)

    return pairs, observed_choices, choice_probs, mu_empirical, s_empirical


