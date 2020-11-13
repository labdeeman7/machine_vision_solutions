import numpy as np

def normal(X, mu, sigma):
    """Return likelihood of data given parameters"

    Computes the likelihood that the data X have been generated
    from the given parameters (mu, sigma) of the one-dimensional
    normal distribution.

    Args:
        X: vector of point samples
        mu: mean
        sigma: standard deviation
    Returns:
        a scalar likelihood
    """
    
    exponent_part = -0.5 * np.sum((X-mu)**2/sigma**2)
    mantissa_part = 1/(np.power((2*np.pi*sigma**2), X.size/2))
    
    lik = mantissa_part * np.exp(exponent_part)
    
#     lik = 1.
    return lik