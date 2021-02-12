import random
import numpy as np

class Gaussian:

    def __init__(self, mean, cov):
        self.mean = mean
        self.cov = cov

    '''
    def gauss_2d(self):
        x = random.gauss(self.mu_x, self.sigma_x)
        y = random.gauss(self.mu_y, self.sigma_y)
        return (x, y)
    '''

    # where mean.shape==(2,) and cov.shape==(2,2).
    def gauss_2d(self):
        point = np.random.multivariate_normal(self.mean, self.cov, 1)
        return point[0]
