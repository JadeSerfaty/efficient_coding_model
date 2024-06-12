import numpy as np
import pandas as pd
import pymc3 as pm
import concurrent.futures
import theano.tensor as tt
import pymc3.parallel_sampling as ps
import pickle
from multiprocessing import Manager


# Functions for efficient coding model
def f(v, mu, s):
    return 0.5 * (1 + np.tanh((v - mu) / (2 * s)))


def f_theano(v, mu, s):
    return 0.5 * (1 + tt.tanh((v - mu) / (2 * s)))


def f_inv(v_tilde, mu, s):
    v_tilde = np.clip(v_tilde, 1e-8, 1 - 1e-8)
    result = mu + s * np.log(v_tilde / (1 - v_tilde))
    return result


def phi_prime(v_tilde, s):
    # The first derivative of the inverse CDF
    return s / (v_tilde * (1 - v_tilde))


def phi_double_prime(v_tilde, s):
    # The second derivative of the inverse CDF
    return s * (2 * v_tilde - 1) / (v_tilde ** 2 * (1 - v_tilde) ** 2)


def f_prime_theano(v, mu, s):
    p = f_theano(v, mu, s)
    return p * (1 - p)


def g(v_hat):
    return 1 / (1 + np.exp(-v_hat))


def g_inv_theano(v):
    return tt.log(v / (1 - v))


def g_inv_prime_theano(v):
    return 1 / (v * (1 - v))


def sech(x):
    '''define the sech function using theano'''
    return 2 / (tt.exp(x) + tt.exp(-x))


def logp_logistic(v, mu, s):
    # logistic PDF transformed to log-probability
    #  Define the log probability function for the logistic distribution
    return tt.log(sech((v - mu) / (2 * s)) ** 2 / (4 * s))


def sech_bis(x):
    # Adaptation of functions using theanos, with numpy for visualization of prior distribution
    return 2 / (np.exp(x) + np.exp(-x))


def logp_logistic_bis(v, mu, s):
    return np.log(sech_bis((v - mu) / (2 * s)) ** 2 / (4 * s))


def normal_cdf(x):
    """ Compute the CDF of the standard normal distribution at x. """
    return 0.5 * (1 + tt.erf(x / tt.sqrt(2)))


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


def run_efficient_coding_model(posterior_distributions_all_participants_higher_sampling, participant_id,
                               rating_data_emo):
    try:
        participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(participant_id,
                                                                                                   rating_data_emo,
                                                                                                   epsilon=1e-6)

        # Bayesian model Setup
        with pm.Model() as model:
            # Prior distributions for parameters
            mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
            s = pm.HalfNormal('s', sigma=s_empirical)
            sigma = pm.HalfNormal('sigma', sigma=1)
            sigma_ext = pm.HalfNormal('sigma_ext', sigma=1)

            # Observations
            # observed_noisy_ratings = pm.Data("observed_noisy_ratings", participant_emo['normalized_average_rating'])
            observed_noisy_ratings = pm.Data("observed_noisy_ratings", participant_emo['normalized_rating'])

            # Define the latent variable v with a logistic prior
            v = pm.Uniform('v', 0, 1, shape=num_videos)  # Starting with a Uniform prior for simplicity
            # Add the custom logistic prior using a Potential
            pm.Potential('v_prior_potential', logp_logistic(v, mu, s))

            # Likelihood function components
            phi_prime_val = phi_prime(v, s)
            phi_double_prime_val = phi_double_prime(v, s)
            F_prime_v0_val = f_prime_theano(v, mu, s)
            g_inv_prime_val = g_inv_prime_theano(observed_noisy_ratings)

            # The mean and variance for the likelihood normal distribution
            mean_likelihood = v + phi_double_prime_val * sigma ** 2
            sd_likelihood = tt.sqrt((phi_prime_val ** 2) * sigma ** 2 + sigma_ext)

            # Define the observed likelihood
            observed = pm.Normal('observed',
                                 mu=mean_likelihood,
                                 sigma=sd_likelihood,
                                 observed=observed_noisy_ratings)

            # Include the multiplicative factors in the likelihood using a Potential
            likelihood_factors = pm.Potential('likelihood_factors',
                                              tt.log(F_prime_v0_val) +
                                              tt.log(g_inv_prime_val))

            # Sampling from the posterior
            trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.95, return_inferencedata=False)

            # Check the convergence
            summary_stats = pm.summary(trace).round(2)

            # Save posterior distributions for participant
            posterior_distributions_all_participants_higher_sampling[participant_id] = {
                "summary_stats": summary_stats,
                "trace": trace
            }

            # Posterior Predictive Checks
            ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])

            # Extract the predicted data
            predicted_data = ppc['observed']
            posterior_distributions_all_participants_higher_sampling[participant_id]["predicted_data"] = predicted_data

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError for participant {participant_id}: {e}")


if __name__ == "__main__":
    # Paths and configuration
    # emotion = "disgust"
    duration = "long"
    main_path_rating_data = "../main_study_data/main_study_rating_data/"
    print(main_path_rating_data)
    # name_file_posterior_distributions = f"main_study_{emotion}_posterior_distributions_complete.p"
    name_file_posterior_distributions = f"main_study_all_emotions_{duration}_duration_posterior_distributions_complete.p"

    # # Load data
    # rating_data_first_part = pd.read_csv(main_path_rating_data + "rating_data_first_part_" + emotion + ".csv")
    # rating_data_second_part = pd.read_csv(main_path_rating_data + "rating_data_second_part_" + emotion + ".csv")

    # Combine the data
    # rating_data_combined = pd.concat([rating_data_first_part, rating_data_second_part])
    rating_data_emo = pd.read_csv(main_path_rating_data + "rating_data_complete_" + duration + "_duration_all_emotions.csv")

    # Filter for the specific emotion
    # rating_data_emo = rating_data_combined[rating_data_combined["emotionName"] == emotion].copy()
    #
    # # Check rating_data_emo and rating_data_combined have the same length
    # if len(rating_data_emo) != len(rating_data_combined):
    #     print("Warning: rating_data_emo and rating_data_combined have different lengths")

    # Filter for the specific duration:
    # short_rating_data_phase1 = all_rating_data[all_rating_data['durationBlackScreen_phase1'] == 900][['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating', 'variance_rating', 'subject_id']]
    # short_rating_data_phase2 = all_rating_data[all_rating_data['durationBlackScreen_phase2'] == 900][['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating', 'variance_rating', 'subject_id']]
    # short_rating_data = pd.concat([short_rating_data_phase1.rename(columns={'rating_phase1': 'rating', 'durationBlackScreen_phase1': 'durationBlackScreen'}), short_rating_data_phase2.rename(columns={'rating_phase2': 'rating', 'durationBlackScreen_phase2': 'durationBlackScreen'})], ignore_index=True)

    # long_rating_data_phase1 = all_rating_data[all_rating_data['durationBlackScreen_phase1'] == 2600][['videoID', 'emotionName', 'durationBlackScreen_phase1', 'rating_phase1', 'average_rating', 'variance_rating', 'subject_id']]
    # long_rating_data_phase2 = all_rating_data[all_rating_data['durationBlackScreen_phase2'] == 2600][['videoID', 'emotionName', 'durationBlackScreen_phase2', 'rating_phase2', 'average_rating', 'variance_rating', 'subject_id']]
    # long_rating_data = pd.concat([long_rating_data_phase1.rename(columns={'rating_phase1': 'rating', 'durationBlackScreen_phase1': 'durationBlackScreen'}), long_rating_data_phase2.rename(columns={'rating_phase2': 'rating', 'durationBlackScreen_phase2': 'durationBlackScreen'})], ignore_index=True)

    all_participant_ids = np.unique(rating_data_emo["subject_id"])

    # Use Manager to create a shared dictionary
    with Manager() as manager:
        posterior_distributions_all_participants = manager.dict()

        # Run models in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [executor.submit(run_efficient_coding_model, posterior_distributions_all_participants, participant_id,
                                       rating_data_emo)
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

        print("Processing completed and results saved successfully.")
