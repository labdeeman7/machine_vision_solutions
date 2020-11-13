import numpy as np
from scipy.special import gamma as gamma_function

def normal_inv_gamma(alpha, beta, delta, gamma, mu, sigma):
    """Return the probability density function for the normal
    inverse gamma density at (mu, sigma)
    
    Args:
        alpha: shape of variance
        beta: scale of variance
        delta: mean of mu
        gamma: precision of mu
        mu: normal mean
        sigma: normal standard deviation
    Returns:
        a probability density function
    """
    # You will find scipy.special.gamma useful
    mantissa_part = (np.sqrt(gamma)/ sigma*np.sqrt(2*np.pi)) * (beta**alpha/gamma_function(alpha)) * (1/sigma**2)**(alpha+1)
    exponent_part = -1 * (2*beta+ gamma*((delta-mu)**2))/2*sigma**2
    pdf = mantissa_part * np.exp(exponent_part)

    return pdf
