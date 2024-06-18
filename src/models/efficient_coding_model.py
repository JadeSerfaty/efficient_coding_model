import pymc3 as pm
import pymc3.parallel_sampling as ps
from scipy import stats
from src.utils.utils import prepare_data_for_efficient_coding, prepare_data_for_efficient_coding_all_emotions
from src.utils.math import *

def run_efficient_coding_model(posterior_distributions_all_participants, participant_id,
                               rating_data_emo, use_mock_data=False):
    try:
        if use_mock_data:
            participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding(participant_id,
                                                                                                       rating_data_emo,
                                                                                                       epsilon=1e-6)
        else:
            participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(
                participant_id, rating_data_emo, epsilon=1e-6)


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
            posterior_distributions_all_participants[participant_id] = {
                "summary_stats": summary_stats,
                "predicted_data": predicted_data,
                "likelihoods": likelihoods
                # "trace": trace
            }

    except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
        print(f"SamplingError for participant {participant_id}: {e}")


def run_separate_sigma_model(posterior_distributions_all_participants, participant_id,
                               rating_data_emo):
    # try:
    participant_emo, mu_empirical, s_empirical, num_videos = prepare_data_for_efficient_coding_all_emotions(
        participant_id, rating_data_emo, epsilon=1e-6)

    # Ensure num_videos is correctly reflecting the unique videos
    num_videos = len(participant_emo['VIDEO_ID'].unique())

    # Adjust videoID to be zero-indexed if necessary
    participant_emo['VIDEO_ID'] = participant_emo['VIDEO_ID'].astype('category').cat.codes

    with pm.Model() as model:

        # Priors
        mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
        s = pm.HalfNormal('s', sigma=s_empirical)
        sigma_short = pm.HalfNormal('sigma_short', sigma=0.5)
        sigma_long = pm.HalfNormal('sigma_long', sigma=0.5)  # Separate sigma for each duration
        sigma_ext = pm.HalfNormal('sigma_ext', sigma=1)

        # Observations
        observed_noisy_ratings_short = participant_emo['NORMALIZED_RATING'][
            participant_emo['DURATION_SHORT'] == 1].values
        observed_noisy_ratings_long = participant_emo['NORMALIZED_RATING'][
            participant_emo['DURATION_SHORT'] == 0].values
        duration_short = participant_emo['DURATION_SHORT'].values

        # Combine short and long ratings into a single array

        # observed_noisy_ratings_combined = observed_noisy_ratings_short * duration_short + observed_noisy_ratings_long * (
        #             1 - duration_short)

        # Combine short and long ratings into a single array
        observed_noisy_ratings_combined = np.concatenate([observed_noisy_ratings_short, observed_noisy_ratings_long])

        # Ensure video IDs are integers and map the latent variable v to each rating
        video_ids_short = participant_emo['VIDEO_ID'][participant_emo['DURATION_SHORT'] == 1].values
        video_ids_long = participant_emo['VIDEO_ID'][participant_emo['DURATION_SHORT'] == 0].values
        video_ids_combined = np.concatenate([video_ids_short, video_ids_long])

        v = pm.Uniform('v', 0, 1, shape=num_videos)
        v_mapped = v[video_ids_combined]

        print("dimensions of observed_noisy_ratings_combined: "+ str(observed_noisy_ratings_combined.shape))

        assert not np.any(np.isnan(observed_noisy_ratings_combined)), "NaNs in observed ratings"
        assert not np.any(np.isinf(observed_noisy_ratings_combined)), "Infinities in observed ratings"

        # Convert combined arrays to pm.Data
        observed_noisy_ratings = pm.Data("observed_noisy_ratings", observed_noisy_ratings_combined)
        duration_short = pm.Data("duration_short", duration_short)

        # Logistic prior
        pm.Potential('v_prior_potential', logp_logistic(v_mapped, mu, s))

        # Likelihood function components
        phi_prime_val = phi_prime(v_mapped, s)
        phi_double_prime_val = phi_double_prime(v_mapped, s)
        F_prime_v0_val = f_prime_theano(v_mapped, mu, s)
        g_inv_prime_val = g_inv_prime_theano(observed_noisy_ratings)

        # Conditional mean and variance
        mean_likelihood = v_mapped + phi_double_prime_val * (
                    sigma_short ** 2 * duration_short + sigma_long ** 2 * (1 - duration_short))
        sd_likelihood = tt.sqrt(
            (phi_prime_val ** 2) * (sigma_short ** 2 * duration_short + sigma_long ** 2 * (1 - duration_short)) + sigma_ext)

        observed = pm.Normal('observed', mu=mean_likelihood, sigma=sd_likelihood, observed=observed_noisy_ratings)
        likelihood_factors = pm.Potential('likelihood_factors', tt.log(F_prime_v0_val) + tt.log(g_inv_prime_val))

        trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.95, return_inferencedata=False)
        summary_stats = pm.summary(trace).round(2)

        ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])
        predicted_data = ppc['observed']

        log_likelihoods = []
        for pred in predicted_data:
            log_likelihoods.append(stats.norm.logpdf(observed_noisy_ratings.get_value(), loc=pred).sum())
        likelihoods = np.exp(log_likelihoods)

        posterior_distributions_all_participants[participant_id] = {
            "summary_stats": summary_stats,
            "predicted_data": predicted_data,
            "likelihoods": likelihoods
        }

        # # Priors
        # mu = pm.Normal('mu', mu=mu_empirical, sigma=s_empirical)
        # s = pm.HalfNormal('s', sigma=s_empirical)
        # sigma_short = pm.HalfNormal('sigma_short', sigma=1)
        # sigma_long = pm.HalfNormal('sigma_long', sigma=1)  # Separate sigma for each duration
        # sigma_ext = pm.HalfNormal('sigma_ext', sigma=1)
        #
        # # Define the binary indicator and observed noisy ratings
        # observed_noisy_ratings_short = pm.Data("observed_noisy_ratings_short",
        #                                        participant_emo['NORMALIZED_RATING'][participant_emo['DURATION_SHORT'] == 1])
        # print("the length of observed_noisy_ratings_short is:", len(observed_noisy_ratings_short))
        # observed_noisy_ratings_long = pm.Data("observed_noisy_ratings_long",
        #                                       participant_emo['NORMALIZED_RATING'][participant_emo['DURATION_SHORT'] == 0])
        # print("the length of observed_noisy_ratings_short is:", len(observed_noisy_ratings_short))
        # duration_short = pm.Data("duration_short", participant_emo['DURATION_SHORT'])
        #
        # # Latent variable v for each video (30 videos)
        # v = pm.Uniform('v', 0, 1, shape=num_videos)
        #
        # # Map the observed ratings to the latent variables
        # v_mapped = v[participant_emo['VIDEO_ID'].values]
        # print("the length of v_mapped is:", len(v_mapped))
        #
        # # Add the custom logistic prior using a Potential
        # pm.Potential('v_prior_potential', logp_logistic(v, mu, s))
        #
        # phi_prime_val = phi_prime(v_mapped, s)
        # phi_double_prime_val = phi_double_prime(v_mapped, s)
        # F_prime_v0_val = f_prime_theano(v_mapped, mu, s)
        #
        # # Conditional mean and variance
        # mean_likelihood_short = v_mapped + phi_double_prime_val * sigma_short ** 2
        # mean_likelihood_long = v_mapped + phi_double_prime_val * sigma_long ** 2
        # sd_likelihood_short = tt.sqrt((phi_prime_val ** 2) * sigma_short ** 2 + sigma_ext)
        # sd_likelihood_long = tt.sqrt((phi_prime_val ** 2) * sigma_long ** 2 + sigma_ext)
        #
        # # Combine short and long likelihoods using the binary indicator
        # mean_likelihood = mean_likelihood_short * duration_short + mean_likelihood_long * (1 - duration_short)
        # sd_likelihood = sd_likelihood_short * duration_short + sd_likelihood_long * (1 - duration_short)
        # observed_noisy_ratings_combined = observed_noisy_ratings_short * duration_short + observed_noisy_ratings_long * (
        #             1 - duration_short)
        #
        # g_inv_prime_val = g_inv_prime_theano(observed_noisy_ratings_combined)
        #
        # observed = pm.Normal('observed', mu=mean_likelihood, sigma=sd_likelihood,
        #                      observed=observed_noisy_ratings_combined)
        # likelihood_factors = pm.Potential('likelihood_factors', tt.log(F_prime_v0_val) + tt.log(g_inv_prime_val))
        #
        # trace = pm.sample(10000, tune=1000, cores=3, target_accept=0.95, return_inferencedata=False)
        # summary_stats = pm.summary(trace).round(2)
        #
        # ppc = pm.sample_posterior_predictive(trace, var_names=["observed"])
        # predicted_data = ppc['observed']
        #
        # log_likelihoods = []
        # for pred in predicted_data:
        #     log_likelihoods.append(stats.norm.logpdf(observed_noisy_ratings_combined.get_value(), loc=pred).sum())
        # likelihoods = np.exp(log_likelihoods)
        #
        # posterior_distributions_all_participants[participant_id] = {
        #     "summary_stats": summary_stats,
        #     "predicted_data": predicted_data,
        #     "likelihoods": likelihoods
        # }

    # except (pm.exceptions.SamplingError, ps.ParallelSamplingError) as e:
    #     print(f"SamplingError for participant {participant_id}: {e}")


