# MIT License

# Copyright (c) 2016 Denis Vida

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

################
# Sources used:
# - https://sciencehouse.wordpress.com/2010/06/23/mcmc-and-fitting-models-to-data/ (accessed Dec 19, 2016)

import scipy.optimize
import numpy as np
import matplotlib.pyplot as plt


def func(x, params):
    """ A function that describes the underlying model.
        NOTE: MODIFY THIS FUNCTION ACCORDING TO YOUR MODEL

        Gets the evaluation of the function at the given point with given parameters. """

    a, b, c, d, e = params

    return a*np.sin(b*x) + c*x**2 + d*x + e




class MCMC(object):

    def __init__(self, func, n_params, x, y, stddev_guess, sigma_error_estimate, chi_tolerance=0.05, 
        max_iter=200000):

        """ Markov Chain Monte Carlo regression, initialization step. 
        
            Arguments:
                func: [function object] a model function on which the evaluations are performed
                n_params: [int] the number of parameters the function is taking (i.e. the dimension of the model)
                x: [ndarray] numpy array containing samples of the independant variable
                y: [ndarray] numpy array containing samples of the dependant variable
                stddev_guess: [float] 'size' of each step in the parameter step, if this is large, the
                    algorithm will stake larger strides in the parameter space, but values too large can
                    lead to non-convergence (keep it at about 0.05)
                sigma_error_estimate: [float] estimate of the underlying error in the data, often a 
                    measurement error sigma is used, but you can just make a guess

            Keyword arguments:
                chi_tolerance: [float] when the average square residuals is less than this number, the 
                    algorithm will stop - this is used to determine when the fit is 'good enough' and stop
                    further iterations. This value will largely depend on the noise in your data.
                max_iter: [int] the maximum number of iterations, i.e. steps which will be taken in the 
                    parameter space (increase it if you cannot get a good convergence)

            Return:
                theta1: [ndarray] numpy array containing the final fitted parameters

        """
        
        # Model parameters
        self.n_params = n_params

        self.func = func
        self.x = x
        self.y = y
        self.stddev_guess = stddev_guess
        self.sigma_error_estimate = sigma_error_estimate
        self.chi_tolerance = chi_tolerance
        self.max_iter = max_iter


    def fit(self):
        """ Markov Chain Monte Carlo univariate regression.

            This function fits the given model to the given data using the Markov Chain Monte Carlo Method. 
            The guesses are chosen using the Gaussian distribution.
            """

        # Initial guess
        theta1 = np.zeros(self.n_params)

        # Solve for the initial parameters
        y_model = self.func(self.x, theta1)

        # Calculate the sum of squared diffs
        chi1 = np.sum((self.y - y_model)**2)

        x_size = len(self.x)

        # Run for the max. number of iterations
        for n in range(self.max_iter):

            # Do a random walk in the parameter space
            theta2 = theta1 + np.random.normal(0.0, self.stddev_guess, self.n_params)

            # Solve for the given parameters
            y_model = self.func(self.x, theta2)

            # Calculate the sum of squared diffs
            chi2 = np.sum((self.y - y_model)**2)

            # Likelihood ratio
            ratio = np.exp((-chi2+chi1)/(2*self.sigma_error_estimate**2))

            # If the new guess has a better likelihood ratio, keep it
            if np.random.uniform(0.0, 1.0) < ratio:
                theta1 = theta2
                chi1 = chi2

                # Break if the desired tolerance is reached
                if chi1/x_size < self.chi_tolerance:
                    break


        print ('Iterations:', "{:,d} out of max. {:,d}".format(n+1, self.max_iter))
        print ('Final mean square residuals: ', chi1/x_size)

        return theta1



if __name__ == '__main__':

    # Independant varibale
    x_data = np.linspace(0, 10, 200)

    # Underlying model parameters
    params = [1.5, 2, 0.1, 0.1, 3]

    # Dependant variable data
    y_data = func(x_data, params)

    # Add some noise to the data points
    y_measurements = y_data + np.random.normal(0, 0.5, len(y_data))

    # Add a few large outliers
    y_measurements[np.random.randint(0, len(x_data), 10)] += np.random.normal(0, 3.0, 10)

    # Take every other point in the measurements (simulate sparser sampling)
    x_sample = x_data[::2]
    y_sample = y_measurements[::2]

    # Plot the input data
    plt.plot(x_data, y_data, label='Underlying model')
    plt.plot(x_data, y_measurements, label='Model with the added noise')
    plt.scatter(x_sample, y_sample, label='Samples taken from the noisy model')

    plt.legend(loc='lower right')
    
    plt.show()
    plt.clf()


    # Init the MCMC fitter
    mcmc = MCMC(func, 5, x_sample, y_sample, 0.2, 2.0)

    print ('Running MCMC...')

    # Run the MCMC fit
    fit = mcmc.fit()

    # Fit the model using least squares (for comparison only)
    popt, pconv = scipy.optimize.curve_fit(lambda x, a, b, c, d, e: func(x, [a, b, c, d, e]), x_sample, y_sample)

    print ('Real parameters:        ', params)
    print ('LS fitted parameters:   ', popt)
    print ('MCMC fitted parameters: ', fit)


    plt.plot(x_data, y_data, label='Underlying model')
    plt.plot(x_data, y_measurements, label='Model with the added noise')
    plt.scatter(x_sample, y_sample, label='Samples taken from the noisy model')
    plt.plot(x_sample, func(x_sample, fit), label='Fitted model - MCMC')
    plt.plot(x_sample, func(x_sample, popt), label='Fitted model - LS')

    plt.legend(loc='lower right')

    plt.show()