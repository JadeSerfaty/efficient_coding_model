import pymc3 as pm
import pymc3.parallel_sampling as ps
from scipy import stats
from src.utils.utils import prepare_data_for_efficient_coding, prepare_data_for_efficient_coding_all_emotions
from src.utils.math import *

DURATIONS_MAPPING_DICT = {900: 0,
                          2600: 1}


def run_efficient_coding_model(rating_data, duration):  #use_mock_data=False
    print("Running efficient coding model")
    results = {}
    try:
        # if use_mock_data:
        #     participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding(rating_data,
        #                                                                                                epsilon=1e-6)
        # else:
        participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(
            rating_data, duration, epsilon=1e-6)

        # Bayesian model Setup
        with pm.Model() as model:
            # Prior distributions for parameters
            mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
            s = pm.HalfNormal('s', sigma=s_empirical, initval=0.1)
            sigma = pm.HalfNormal('sigma', sigma=1)
            sigma_ext = pm.HalfNormal('sigma_ext', sigma=1, initval=0.1)

            # # Observations
            # if use_mock_data:
            #     observed_noisy_ratings = pm.Data("observed_noisy_ratings", participant_emo['NORMALIZED_AVERAGE_RATING'])
            # else:
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


def run_separate_sigma_model(rating_data, duration):
    print("Running separate sigma model")
    results = {}
    try:
        rating_data, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(rating_data, duration, epsilon=1e-6)

        with pm.Model() as model:
            # Priors
            mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
            s = pm.HalfNormal('s', sigma=s_empirical, initval=0.1)
            sigma_short = pm.HalfNormal('sigma_short', sigma=0.5)
            sigma_long = pm.HalfNormal('sigma_long', sigma=0.5)
            sigma_ext = pm.HalfNormal('sigma_ext', sigma=0.5, initval=0.1)

            # Observations
            observed_noisy_ratings = rating_data['NORMALIZED_RATING'].values
            duration_short = np.array([DURATIONS_MAPPING_DICT[val] for val in rating_data['DURATION'].values])
            observed_noisy_ratings_short = observed_noisy_ratings[duration_short == 0]
            observed_noisy_ratings_long = observed_noisy_ratings[duration_short == 1]
            print(observed_noisy_ratings_short.shape)
            print(observed_noisy_ratings_long.shape)

            # Define the latent variable v with a logistic prior
            v = pm.Uniform('v', 0, 1, shape=num_videos)
            pm.Potential('v_prior_potential', logp_logistic(v, mu, s))

            # Likelihood function components
            phi_prime_val = phi_prime(v, s)
            phi_double_prime_val = phi_double_prime(v, s)
            F_prime_v0_val = f_prime_theano(v, mu, s)
            g_inv_prime_val_long = g_inv_prime_theano(observed_noisy_ratings_long)
            g_inv_prime_val_short = g_inv_prime_theano(observed_noisy_ratings_short)

            # Conditional mean and variance
            mean_likelihood_short = v + phi_double_prime_val * sigma_short ** 2
            mean_likelihood_long = v + phi_double_prime_val * sigma_long ** 2
            sd_likelihood_short = tt.sqrt((phi_prime_val ** 2) * sigma_short ** 2 + sigma_ext)
            sd_likelihood_long = tt.sqrt((phi_prime_val ** 2) * sigma_long ** 2 + sigma_ext)

            observed_short = pm.Normal('observed_short', mu=mean_likelihood_short, sigma=sd_likelihood_short, observed=observed_noisy_ratings_short)
            observed_long = pm.Normal('observed_long', mu=mean_likelihood_long, sigma=sd_likelihood_long, observed=observed_noisy_ratings_long)

            likelihood_factors = pm.Potential('likelihood_factors', tt.log(F_prime_v0_val) + tt.log(g_inv_prime_val_long) + tt.log(g_inv_prime_val_short))

            trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.95, return_inferencedata=False)
            summary_stats = pm.summary(trace).round(2)

            ppc = pm.sample_posterior_predictive(trace, var_names=["observed_long", "observed_short"])
            predicted_data_long = ppc['observed_long']
            predicted_data_short = ppc['observed_short']

            log_likelihoods_long = []
            log_likelihoods_short = []

            for pred_long, pred_short in zip(predicted_data_long, predicted_data_short):
                log_likelihood_long = stats.norm.logpdf(observed_noisy_ratings_long, loc=pred_long).sum()
                log_likelihood_short = stats.norm.logpdf(observed_noisy_ratings_short, loc=pred_short).sum()
                log_likelihoods_long.append(log_likelihood_long)
                log_likelihoods_short.append(log_likelihood_short)

            likelihoods_long = np.exp(log_likelihoods_long)
            likelihoods_short = np.exp(log_likelihoods_short)

            results = {
                "summary_stats": summary_stats,
                "predicted_data_long": predicted_data_long,
                "predicted_data_short": predicted_data_short,
                "likelihoods_long": likelihoods_long,
                "likelihoods_short": likelihoods_short,
                "v_estimates": trace['v'],
                "sigma_short": trace['sigma_short'],
                "sigma_long": trace['sigma_long'],
                "sigma_ext": trace['sigma_ext']
            }

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError")

    return results