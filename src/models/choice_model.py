import pymc3 as pm
from src.utils.utils import *
from src.utils.math import *

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
