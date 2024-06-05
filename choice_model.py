import numpy as np
import pandas as pd
import pymc3 as pm
import concurrent.futures
import theano.tensor as tt
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


def run_choice_model(participant_id, data, rating_data_emo, choice_data_emo, choice_results):
    pairs, observed_choices, choice_probs, mu_empirical, s_empirical = prepare_data_for_choice_model(data,
                                                                                                     participant_id,
                                                                                                     rating_data_emo,
                                                                                                     choice_data_emo)

    try:
        with pm.Model() as choice_model:
            s = s_empirical
            mu = mu_empirical
            sigma = pm.HalfCauchy('sigma', beta=1)
            sigma_ext = pm.HalfCauchy('sigma_ext', beta=1)
            counter = 0  # Ensure unique naming

            for i, ((v1, v2), observed) in enumerate(zip(pairs, observed_choices)):
                # Likelihood function components
                E_v1 = v1 + phi_double_prime(v1, s) * (sigma ** 2)
                Var_v1 = (phi_prime(v1, s) ** 2) * (sigma ** 2) + sigma_ext
                E_v2 = v2 + phi_double_prime(v2, s) * (sigma ** 2)
                Var_v2 = (phi_prime(v2, s) ** 2) * (sigma ** 2) + sigma_ext

                # Compute the probability using Normal CDF
                denom = tt.sqrt(Var_v1 + Var_v2 + 2 * sigma_ext ** 2)
                delta_E = E_v1 - E_v2

                prob_choice = pm.Deterministic(f'prob_choice_{i}', normal_cdf(delta_E / denom))
                pm.Bernoulli(f'choice_{i}', p=prob_choice, observed=observed)

                counter += 1

            # Sampling from the model
            trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.98, start={'sigma': 1, 'sigma_ext': 1},
                              return_inferencedata=False, max_treedepth=15)

            # Extract and store the probability values from the trace
            for var_name in trace.varnames:
                if var_name.startswith('prob_choice_'):
                    prob_values = trace.get_values(var_name)
                    mean_prob = np.mean(prob_values)
                    choice_probs.append(mean_prob)
                    print(mean_prob)

            choice_results[participant_id] = {
                "choice_probabilities": choice_probs,
                "observed_choices": observed_choices
            }
            print(f"Results saved for participant {participant_id}")

    except Exception as e:
        print(f"An error occurred with participant {participant_id}: {e}")


if __name__ == "__main__":
    # Paths and configuration
    emotion = "sadness"
    main_path_rating_data = ("/Users/jadeserfaty/Library/Mobile "
                             "Documents/com~apple~CloudDocs/code/lido/introspection_task/main_study_data"
                             "/main_study_rating_data/")
    main_path_choice_data = ("/Users/jadeserfaty/Library/Mobile "
                             "Documents/com~apple~CloudDocs/code/lido/introspection_task/main_study_data"
                             "/main_study_choice_data/")
    name_file_posterior_distributions = f"main_study_{emotion}_posterior_distributions_complete.p"
    name_file_choice_model_outputs = f"main_study_{emotion}_choice_model_complete.p"

    # Load the posterior distributions
    with open(name_file_posterior_distributions, 'rb') as fp:
        posterior_distributions = pickle.load(fp)

    # Load data
    rating_data_first_part = pd.read_csv(main_path_rating_data + "rating_data_first_part_" + emotion + ".csv")
    rating_data_second_part = pd.read_csv(main_path_rating_data + "rating_data_second_part_" + emotion + ".csv")
    choice_data_first_part = pd.read_csv(main_path_choice_data + "choice_data_first_part_" + emotion + ".csv")
    choice_data_second_part = pd.read_csv(main_path_choice_data + "choice_data_second_part_" + emotion + ".csv")

    # Combine the data
    rating_data_combined = pd.concat([rating_data_first_part, rating_data_second_part])
    choice_data_combined = pd.concat([choice_data_first_part, choice_data_second_part])

    rating_data_emo = rating_data_combined[rating_data_combined["emotionName"] == emotion].copy()
    choice_data_emo = choice_data_combined[choice_data_combined["emotionName"] == emotion].copy()

    # Use Manager to create a shared dictionary
    with Manager() as manager:
        choice_results = manager.dict()

        # Run models in parallel
        with concurrent.futures.ProcessPoolExecutor() as executor:
            futures = [
                executor.submit(run_choice_model, participant_id, data, rating_data_emo, choice_data_emo, choice_results)
                for participant_id, data in posterior_distributions.items()]
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

        print("Processing completed and results saved successfully.")
