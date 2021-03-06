import numpy as np


class Gaussian:
    """
    A Gaussian kernel.
    """

    def __init__(self, sigma=0.5):
        self.sigma = sigma

    def __call__(self, X, Z):
        return Gaussian.kernel(X, Z, self.sigma)

    @classmethod
    def kernel(cls, X, Z, sigma):
        """
        Computes the Gaussian kernel for the matrices X and Z.
        """
        # make sure X and Z are Numpy matrices, and float values
        X = np.matrix(X,)
        Z = np.matrix(Z)
#        # Normalise between different bandwidths
#        if hasattr(sigma, '__iter__'):
#           X /= 1.4142 * (np.array(sigma))
#           Z /= 1.4142 * (np.array(sigma))
#           sigma = 1.0
#        else:
#           sigma = float(sigma)
        n, m = X.shape[0], Z.shape[0]
        XX = np.multiply(X, X)
        XX = XX.sum(axis = 1)
        ZZ = np.multiply(Z, Z)
        ZZ = ZZ.sum(axis = 1)
        d = np.tile(XX, (1, m)) + np.tile(ZZ.T, (n, 1)) - 2 * X * Z.T
        Kexpd = np.exp(-d.T / (2 * sigma * sigma))
        return np.matrix(Kexpd)
