"""Modules to generate RUM and RRM data"""

# Load modules
import numpy as np

# Function to generate RUM data
def gen_rum(V: np.ndarray, rng: np.random.Generator = None):
    """Generate RUM data"""
    
    # Create scalars
    N, J = V.shape

    # Create Gumbel-distributed error terms
    if rng is not None:
        generator = rng
    else:
        generator = np.random.Generator()

    e = generator.gumbel(size=(N,J))

    # Create random utilities
    U = V + e

    # Choice is the alternative that maximises the random utility
    Y = U.argmax(axis=1) + 1

    # Compute the choice probabilities, log-likelihood and Rho-squared
    p = np.exp(V)/np.exp(V).sum(axis=1,keepdims=True)

    # Return choices, probabilities and log-likelihood
    return Y, V, p

# Function to generate RRM data
def gen_rrm(R,N):
    """Generate RRM data"""
    pass