import numpy as np

def log_normal(X, mu, sigma):
    """Return log-likelihood of data given parameters"

    Computes the log-likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar log-likelihood
    """
    len_X = X.size   #length of X vector
    first_term = -0.5*len_X*np.log(2*np.pi)
    second_term = -0.5*len_X*np.log(sigma**2)
    third_term = -0.5*np.sum((X-mu)**2/sigma**2)
    
    
    loglik = first_term + second_term + third_term
    return loglik
 