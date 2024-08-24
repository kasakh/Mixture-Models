"""Code for the base mixture model class.

"""

import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
from autograd import grad, hessian_vector_product
from scipy.optimize import minimize
from autograd.scipy.special import logsumexp
import autograd.scipy.stats.multivariate_normal as mvn
from autograd.misc.flatten import flatten_func
from scipy import linalg
from sklearn.metrics.cluster import adjusted_rand_score
from scipy.optimize import OptimizeResult


class MM(object):
    """Base mixture model class.

    This base class (i.e. MM) should not be instantiated explicitly;
    use the specific subclass according to the desired model (e.g. GMM) instead.

    Parameters
    ----------
    data : matrix
        Input data to be fitted. Must be a real matrix with finite values,
        e.g. missing values (NA, NaN), +Inf, -Inf are all forbidden.

    Attributes
    ----------
    data
    num_dim : int
        Number of variables in the data.
    num_datapoints : int
        Number of observations in the data.

    """

    def __init__(self, data):
        self.data_checker(data)
        self.data = np.copy(data)
        self.num_dim = np.shape(self.data)[1]
        self.num_datapoints = np.shape(self.data)[0]

    def mvn_logpdf(self, X, mu, cov_sqrt):
        """Computes the multivariate normal log probability density function.

        Parameters
        ----------
        X : (..., N, D) ndarray
            Arguments at which the normal log-pdf is to be evaluated.
            Here D is the number of dimensions, and N the number of arguments.
        mu : (..., D) vector
            Mean vector. Must be broadcastable to the same shape as X.
        cov_sqrt : (..., D, D) matrix
            Nonsingular matrix. Must be conformable with `X-mu`.

        Returns
        -------
        mvn_logpdf : (..., N)
            Values of the D-dimensional normal log-pdf evaluated at X.

        Notes
        -----
        The multivariate normal probability density function is traditionally specified by

        .. math:: f(x|\mu,\Sigma) = |2\pi\Sigma|^{-1/2} \cdot \exp\left\{-(x-\mu)^T\Sigma^{-1}(x-\mu)/2\right}

        for argument :math:`x`, mean vector :math:`\mu` and covariance matrix :math:`\Sigma`.

        This method differs from the traditional specification in two respects:
        Firstly, instead of being parametrized by covariance matrix :math:`\Sigma`,
        it takes as argument the (Cholesky) square root :math:`\Sigma^{1/2}` such that :math:`\Sigma^{1/2}\Sigma^{1/2}=\Sigma`.
        Secondly, instead of returning :math:`f`, it returns the logarithm :math:`\log f` instead.

        Examples
        --------
        >>> X = np.arange(6).reshape((2,3))
        >>> mu = np.array([2,3,4])
        >>> cov_sqrt = np.eye(3)
        >>> MM(X).mvn_logpdf(X, mu, cov_sqrt)
        array([-8.7568156, -4.2568156])
        """
        return -0.5 * np.log(
            np.linalg.det(2 * np.pi * cov_sqrt.T @ cov_sqrt)
        ) - 0.5 * np.sum(((X - mu) @ np.linalg.inv(cov_sqrt)) ** 2, axis=1)

    def log_normalize(self, x):
        """Given a log vector of unnormalized probabilities, returns a log vector of normalized probabilities."""
        return x - logsumexp(x)

    def gmm_log_likelihood(self, params, data):
        """Calculates the log-likelihood for the GMM model."""
        cluster_lls = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            cluster_lls.append(log_proportion + self.mvn_logpdf(data, mean, cov_sqrt))

        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def alt_gmm_log_likelihood(self, params, data):
        """Calculates the log-likelihood for the GMM model."""
        cluster_lls = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            cov = cov_sqrt.T @ cov_sqrt
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def plot_ellipse(self, ax, mean, cov_sqrt, alpha, num_points=100):
        angles = np.linspace(0, 2 * np.pi, num_points)
        circle_pts = np.vstack([np.cos(angles), np.sin(angles)]).T * 2.0
        cur_pts = mean + np.dot(circle_pts, cov_sqrt)
        ax.plot(cur_pts[:, 0], cur_pts[:, 1], "-", alpha=alpha)
        plt.plot(cur_pts[:, 0], cur_pts[:, 1], "-", alpha=alpha)

    # plt.show()
    # ax.plot(range(120),range(120))

    def plot_gaussian_mixture(self, params, ax):
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            alpha = np.minimum(1.0, np.exp(log_proportion) * 10)
            self.plot_ellipse(ax, mean, cov_sqrt, alpha)
        # plot_ellipse(ax, mean, cov_sqrt, alpha)

    def draw_clusters(self, params, input_data=None):
        if input_data is None:
            input_data = self.data
        if np.shape(input_data)[1] != 2:
            raise ValueError("Input data to be plotted must be 2-dimensional")
        fig = plt.figure()
        ax = fig.add_subplot(111)
        self.plot_gaussian_mixture(params, ax)
        ax.plot(input_data[:, 0], input_data[:, 1], "k.")
        plt.show()

    def kl_mvn(self, m0, S0, m1, S1):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.

        - accepts stacks of means, but only one S0 and S1

        From wikipedia
        KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                      (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term = np.trace(iS1 @ S0)
        det_term = np.log(np.linalg.det(S1)) - np.log(
            np.linalg.det(S0)
        )  # np.sum(np.log(S1)) - np.sum(np.log(S0))
        quad_term = (
            diff.T @ np.linalg.inv(S1) @ diff
        )  # np.sum( (diff*diff) * iS1, axis=1)
        # print("det term {}",format(det_term))
        return 0.5 * (tr_term + det_term + quad_term - N)

    def kl_inverse_mvn(self, m1, S1, m0, S0):
        """
        Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
        Also computes KL divergence from a single Gaussian pm,pv to a set
        of Gaussians qm,qv.
        Diagonal covariances are assumed.  Divergence is expressed in nats.

        - accepts stacks of means, but only one S0 and S1

        From wikipedia
        KL( (m0, S0) || (m1, S1))
             = .5 * ( tr(S1^{-1} S0) + log |S1|/|S0| +
                      (m1 - m0)^T S1^{-1} (m1 - m0) - N )
        """
        # store inv diag covariance of S1 and diff between means
        N = m0.shape[0]
        iS1 = np.linalg.inv(S1)
        diff = m1 - m0

        # kl is made of three terms
        tr_term = np.trace(iS1 @ S0)
        det_term = np.log(np.linalg.det(S1)) - np.log(
            np.linalg.det(S0)
        )  # np.sum(np.log(S1)) - np.sum(np.log(S0))
        quad_term = (
            diff.T @ np.linalg.inv(S1) @ diff
        )  # np.sum( (diff*diff) * iS1, axis=1)
        # print(tr_term,det_term,quad_term)
        return 0.5 * (tr_term + det_term + quad_term - N)

    # def kl_div_tot(self, means, covariances):
    #     kl_divs = []
    #     for i in range(0, means.shape[0]):
    #         # print(i)
    #         for j in range(i, means.shape[0]):
    #             # print(i,j)
    #             kl_divs.append(
    #                 self.kl_mvn(means[i], covariances[i], means[j], covariances[j])
    #             )
    #     # print(kl_divs)
    #     return np.sum(kl_divs)

    # def kl_div_inverse_tot(self, means, covariances):
    #     kl_divs = []
    #     for i in range(0, means.shape[0]):
    #         # print(i)
    #         for j in range(i, means.shape[0]):
    #             # print(i,j)
    #             kl_divs.append(
    #                 self.kl_inverse_mvn(
    #                     means[i], covariances[i], means[j], covariances[j]
    #                 )
    #             )
    #     # print(kl_divs)
    #     return np.sum(kl_divs)

    # def kl_div_tot_print(self, means, covariances):
    #     kl_divs = []
    #     for i in range(0, means.shape[0]):
    #         # print(i)
    #         for j in range(i, means.shape[0]):
    #             # print(i,j)
    #             kl_divs.append(
    #                 self.kl_mvn(means[i], covariances[i], means[j], covariances[j])
    #             )
    #     print("####start of KLB####")
    #     print("pairwise KLF", kl_divs)
    #     print(" total KLF ", np.sum(kl_divs))
    #     return "####end of KLF####"

    # def kl_div_inverse_tot_print(self, means, covariances):
    #     kl_divs = []
    #     for i in range(0, means.shape[0]):
    #         # print(i)
    #         for j in range(i, means.shape[0]):
    #             # print(i,j)
    #             kl_divs.append(
    #                 self.kl_inverse_mvn(
    #                     means[i], covariances[i], means[j], covariances[j]
    #                 )
    #             )
    #     print("####start of KLB####")
    #     print("pairwise KLB ", kl_divs)
    #     print("total KLB", np.sum(kl_divs))
    #     return "####end of KLB####"

    def sgd1(
        self,
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.001,
        mass=0.9,
        startiter=0,
        maxiter=1000,
        tol=1e-5,
        callback=None,
        **kwargs
    ):
        """``scipy.optimize.minimize`` compatible implementation of stochastic
        gradient descent with momentum.
        Adapted from ``autograd/misc/optimizers.py``.
        """
        x = x0
        velocity = np.zeros_like(x)
        prev_val = fun(x)
        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            velocity = mass * velocity - (1.0 - mass) * g
            x = x + learning_rate * velocity

            if abs(fun(x) - prev_val) < tol:
                print("fun(x) - prev_val", fun(x) - prev_val)
                break

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)

    def rmsprop1(
        self,
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.1,
        gamma=0.9,
        eps=1e-8,
        startiter=0,
        maxiter=1000,
        tol=1e-5,
        callback=None,
        **kwargs
    ):
        """``scipy.optimize.minimize`` compatible implementation of root mean
        squared prop: See Adagrad paper for details.
        Adapted from ``autograd/misc/optimizers.py``.
        """
        x = x0
        avg_sq_grad = np.ones_like(x)
        prev_val = fun(x)
        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            avg_sq_grad = avg_sq_grad * gamma + g**2 * (1 - gamma)
            x = x - learning_rate * g / (np.sqrt(avg_sq_grad) + eps)
            if abs(fun(x) - prev_val) < tol:
                break

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)

    def adam1(
        self,
        fun,
        x0,
        jac,
        args=(),
        learning_rate=0.05,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        startiter=0,
        maxiter=50,
        tol=1e-5,
        callback=None,
        **kwargs
    ):
        """``scipy.optimize.minimize`` compatible implementation of ADAM -
        [http://arxiv.org/pdf/1412.6980.pdf].
        Adapted from ``autograd/misc/optimizers.py``.
        """
        x = x0
        m = np.zeros_like(x)
        v = np.zeros_like(x)
        prev_val = fun(x)
        for i in range(startiter, startiter + maxiter):
            g = jac(x)

            if callback and callback(x):
                break

            m = (1 - beta1) * g + beta1 * m  # first  moment estimate.
            v = (1 - beta2) * (g**2) + beta2 * v  # second moment estimate.
            mhat = m / (1 - beta1 ** (i + 1))  # bias correction.
            vhat = v / (1 - beta2 ** (i + 1))
            x = x - learning_rate * mhat / (np.sqrt(vhat) + eps)
            if abs(fun(x) - prev_val) < tol:
                break

        i += 1
        return OptimizeResult(x=x, fun=fun(x), jac=g, nit=i, nfev=i, success=True)

    def rate_checker(self, param_name, float_input):
        """
        input: float_input, param_name

        output: BOOLEAN

        description: Checks whether the float_input is a float between 0 and 1. Useful for validating
        inputs to learning rates, momentums, etc. param_name is the parameter name which we are trying to validate.
        """
        if (
            np.isfinite(float_input)
            and isinstance(float_input, float)
            and float_input > 0
            and float_input < 1
        ):
            return True
        else:
            raise ValueError(
                "%s is input as %f. It should be a float between 0 and 1."
                % (param_name, float_input)
            )

    def optimize(
        self,
        flattened_obj,
        flattened_init_params,
        callback,
        opt_routine,
        **optim_params
    ):
        if "learning_rate" in optim_params:
            if self.rate_checker("learning_rate", optim_params["learning_rate"]):
                learning_rate = optim_params["learning_rate"]
        else:
            learning_rate = 0.05

        if "tol" in optim_params:
            if self.rate_checker("tol", optim_params["tol"]):
                tol = optim_params["tol"]
        else:
            tol = 1e-5

        if "beta1" in optim_params:
            if self.rate_checker("beta1", optim_params["beta1"]):
                beta1 = optim_params["beta1"]
        else:
            beta1 = 0.9

        if "beta2" in optim_params:
            if self.rate_checker("beta2", optim_params["beta2"]):
                beta2 = optim_params["beta2"]
        else:
            beta2 = 0.999

        if "gamma" in optim_params:
            if self.rate_checker("gamma", optim_params["gamma"]):
                gamma = optim_params["gamma"]
        else:
            gamma = 0.9

        if "maxiter" in optim_params:
            if isinstance(optim_params["maxiter"], int) and optim_params["maxiter"] > 0:
                maxiter = optim_params["maxiter"]
            else:
                raise ValueError("maxiter should be a positive integer")
        else:
            maxiter = 100

        if "mass" in optim_params:
            if self.rate_checker("mass", optim_params["mass"]):
                mass = optim_params["mass"]
        else:
            mass = 0.9

        if opt_routine == "Newton-CG":
            minimize(
                flattened_obj,
                flattened_init_params,
                jac=grad(flattened_obj),
                hessp=hessian_vector_product(flattened_obj),
                method="Newton-CG",
                tol=1e-5,
                options={"maxiter": maxiter},
                callback=callback,
            )
        elif opt_routine == "adam":
            flattened_params = flattened_init_params
            flattened_params = self.adam1(
                flattened_obj,
                flattened_params,
                grad(flattened_obj),
                learning_rate=learning_rate,
                beta1=beta1,
                beta2=beta2,
                eps=1e-5,
                startiter=0,
                maxiter=maxiter,
                tol=tol,
                callback=callback,
            )

        elif opt_routine == "rms_prop":
            flattened_params = flattened_init_params
            flattened_params = self.rmsprop1(
                flattened_obj,
                flattened_params,
                grad(flattened_obj),
                learning_rate=learning_rate,
                gamma=gamma,
                eps=1e-5,
                startiter=0,
                maxiter=maxiter,
                tol=tol,
                callback=callback,
            )

        elif opt_routine == "grad_descent":
            flattened_params = flattened_init_params
            flattened_params = self.sgd1(
                flattened_obj,
                flattened_params,
                grad(flattened_obj),
                learning_rate=learning_rate,
                mass=mass,
                startiter=0,
                maxiter=maxiter,
                tol=tol,
                callback=callback,
            )
        else:
            raise ValueError(
                'opt_routine should be one of {"grad_descent","rms_prop","adam","Newton-CG"} '
            )

    def report_likelihood(self, likelihood):
        """Auxiliary function for use in callbacks."""
        if not np.isfinite(likelihood):
            raise ValueError("Log likelihood is {}".format(likelihood))
        print("Log likelihood {}".format(likelihood))

    def data_checker(self, data):
        """Verifies that data is of the correct form for a mixture model to be fitted."""
        if np.isnan(data).any():
            raise ValueError("Input data contains NANs")
        elif not (np.isfinite(data).all()):
            raise ValueError("Input data contains non-finite numbers")
        elif not (np.isreal(data).all()):
            raise ValueError("Input data contains non-real numbers")

    def num_clust_checker(self, K):
        """Verifies that the given number of clusters is consistent with the mixture model data."""
        if np.isnan(K).any():
            raise ValueError("Number of clusters should not be NaN")
        elif not (np.isfinite(K).all()):
            raise ValueError("Number of clusters should not be infinite")
        elif not (isinstance(K, int) and (K > 0)):
            raise ValueError(
                "Number of clusters should not be non positive integers or floats or other datatypes; only positive integers are allowed"
            )
        elif self.num_datapoints < K:
            raise ValueError(
                "Number of clusters larger than number of input datapoints"
            )

    def print_mpkl(self, means, covariances):
        kl_divs = []
        for i in range(0, means.shape[0]):
            # print(i)
            for j in range(i, means.shape[0]):
                # print(i,j)
                kl_divs.append(
                    self.kl_mvn(means[i], covariances[i], means[j], covariances[j])
                    - self.kl_inverse_mvn(
                        means[i], covariances[i], means[j], covariances[j]
                    )
                )
        #         print(kl_divs)
        return np.max(np.abs(kl_divs))
