# src/bayes.py
import numpy as np
from scipy.stats import norm

def posterior_diff_normal(control: np.ndarray, treatment: np.ndarray, prior_mu=0.0, prior_sigma=1000.0):
    """
    Computes posterior for difference in means assuming Normal likelihood and conjugate Normal prior.
    
    Args:
        control: Array of control outcomes
        treatment: Array of treatment outcomes
        prior_mu: Prior mean for treatment effect
        prior_sigma: Prior std deviation
    
    Returns:
        posterior_mu: Posterior mean of lift (treatment - control)
        posterior_sigma: Posterior std of lift
    """
    n_c = len(control)
    n_t = len(treatment)
    mean_c = np.mean(control)
    mean_t = np.mean(treatment)
    var_c = np.var(control, ddof=1)
    var_t = np.var(treatment, ddof=1)
    
    # Likelihood variance
    lik_var = var_c / n_c + var_t / n_t
    # Posterior variance
    post_var = 1 / (1/prior_sigma**2 + 1/lik_var)
    # Posterior mean
    post_mu = post_var * (prior_mu/prior_sigma**2 + (mean_t - mean_c)/lik_var)
    post_sigma = np.sqrt(post_var)
    
    return post_mu, post_sigma

def prob_greater_than_zero(mu: float, sigma: float):
    """
    Probability that posterior lift > 0
    """
    return 1 - norm.cdf(0, loc=mu, scale=sigma)

def prob_in_rope(mu: float, sigma: float, rope=(-0.005, 0.005)):
    """
    Probability that posterior lift is within ROPE (Region of Practical Equivalence)
    """
    lower, upper = rope
    return norm.cdf(upper, loc=mu, scale=sigma) - norm.cdf(lower, loc=mu, scale=sigma)

def bayesian_lift_summary(control: np.ndarray, treatment: np.ndarray, rope=(-0.005,0.005), prior_mu=0.0, prior_sigma=1000.0):
    """
    Returns a dictionary with posterior mean, sigma, P(lift>0), P(lift in ROPE)
    """
    mu, sigma = posterior_diff_normal(control, treatment, prior_mu, prior_sigma)
    return {
        'posterior_mu': float(mu),
        'posterior_sigma': float(sigma),
        'p_lift_gt_0': float(prob_greater_than_zero(mu, sigma)),
        'p_lift_in_rope': float(prob_in_rope(mu, sigma, rope))
    }

if __name__ == "__main__":
    # Example usage
    np.random.seed(42)
    control = np.random.normal(100, 10, 50)
    treatment = np.random.normal(103, 12, 50)
    
    summary = bayesian_lift_summary(control, treatment, rope=(-2,2))
    print("Bayesian Lift Summary:")
    print(summary)
