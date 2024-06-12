import pymc3 as pm
import pymc3.parallel_sampling as ps
from utils import *

def run_efficient_coding_model(posterior_distributions_all_participants, participant_id,
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
            posterior_distributions_all_participants[participant_id] = {
                "summary_stats": summary_stats,
                "trace": trace
            }

            # Posterior Predictive Checks
            ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])

            # Extract the predicted data
            predicted_data = ppc['observed']
            posterior_distributions_all_participants[participant_id]["predicted_data"] = predicted_data

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError for participant {participant_id}: {e}")

