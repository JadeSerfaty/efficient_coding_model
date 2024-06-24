import pymc3 as pm
import pymc3.parallel_sampling as ps
from scipy import stats
from src.utils.utils import prepare_data_for_efficient_coding, prepare_data_for_efficient_coding_all_emotions
from src.utils.math import *

def run_efficient_coding_model(rating_data, use_mock_data=False):
    results = {}
    try:
        if use_mock_data:
            participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding(rating_data,
                                                                                                       epsilon=1e-6)
        else:
            participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(rating_data, epsilon=1e-6)


        # Bayesian model Setup
        with pm.Model() as model:
            # Prior distributions for parameters
            mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
            s = pm.HalfNormal('s', sigma=s_empirical)
            sigma = pm.HalfNormal('sigma', sigma=1)
            sigma_ext = pm.HalfNormal('sigma_ext', sigma=1)

            # Observations
            if use_mock_data:
                observed_noisy_ratings = pm.Data("observed_noisy_ratings", participant_emo['NORMALIZED_AVERAGE_RATING'])
            else:
                observed_noisy_ratings = pm.Data("observed_noisy_ratings", participant_emo['NORMALIZED_RATING'])

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

            # Posterior Predictive Checks
            ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])

            # Extract the predicted data
            predicted_data = ppc['observed']

            # Calculate the log-likelihood for each data point and sum them
            log_likelihoods = []
            for pred in predicted_data:
                log_likelihoods.append(stats.norm.logpdf(observed_noisy_ratings.get_value(), loc=pred).sum())

            # Convert log-likelihoods to likelihoods
            likelihoods = np.exp(log_likelihoods)

            # Save posterior distributions for participant
            results = {
                "summary_stats": summary_stats,
                "predicted_data": predicted_data,
                "likelihoods": likelihoods
                # "trace": trace
            }

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError")

    return results

def run_separate_sigma_model(rating_data):
    results = {}
    try:
        rating_data, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(rating_data, epsilon=1e-6)

        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
            s = pm.HalfNormal('s', sigma=s_empirical, initval=0.1)
            sigma_short = pm.HalfNormal('sigma_short', sigma=0.5)
            sigma_long = pm.HalfNormal('sigma_long', sigma=0.5)  # Separate sigma for each duration
            sigma_ext = pm.HalfNormal('sigma_ext', sigma=0.5, initval=0.1)

            # Observations
            observed_noisy_ratings = rating_data['NORMALIZED_RATING'].values
            duration_short = rating_data['DURATION_SHORT'].values

            # Define the latent variable v with a logistic prior
            v = pm.Uniform('v', 0, 1, shape=num_videos)  # Starting with a Uniform prior for simplicity
            # Add the custom logistic prior using a Potential
            pm.Potential('v_prior_potential', logp_logistic(v, mu, s))

            # Likelihood function components
            phi_prime_val = phi_prime(v, s)
            phi_double_prime_val = phi_double_prime(v, s)
            F_prime_v0_val = f_prime_theano(v, mu, s)
            g_inv_prime_val = g_inv_prime_theano(observed_noisy_ratings)

            # Conditional mean and variance
            mean_likelihood = v + phi_double_prime_val * (
                    sigma_short ** 2 * duration_short + sigma_long ** 2 * (1 - duration_short))
            sd_likelihood = tt.sqrt(
                (phi_prime_val ** 2) * (sigma_short ** 2 * duration_short + sigma_long ** 2 * (
                            1 - duration_short)) + sigma_ext)

            observed = pm.Normal('observed', mu=mean_likelihood, sigma=sd_likelihood, observed=observed_noisy_ratings)
            likelihood_factors = pm.Potential('likelihood_factors', tt.log(F_prime_v0_val) + tt.log(g_inv_prime_val))

            trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.95, return_inferencedata=False)
            summary_stats = pm.summary(trace).round(2)

            ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])
            predicted_data = ppc['observed']

            log_likelihoods = []
            for pred in predicted_data:
                log_likelihoods.append(stats.norm.logpdf(observed_noisy_ratings, loc=pred).sum())
            likelihoods = np.exp(log_likelihoods)

            results = {
                "summary_stats": summary_stats,
                "predicted_data": predicted_data,
                "likelihoods": likelihoods
            }

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError")

    return results
