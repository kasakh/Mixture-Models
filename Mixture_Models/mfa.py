from .mixture_models import *
from .checkers import *
import autograd.numpy as np

class MFA(MM):
    def objective(self, params):
        return -self.fac_log_likelihood(params, self.data)

    def mvn_cov_logpdf(self, X, mu, cov):
        return -0.5 * np.log(np.linalg.det(2 * np.pi * cov)) - 0.5 * np.sum(
            np.dot((X - mu), np.linalg.inv(cov)) * (X - mu), axis=1
        )

    def alt_objective(self, params):
        return -self.fac_log_likelihood_alt(params, self.data)

    def likelihood(self, params):
        return -self.alt_objective(params)

    def aic(self, params):
        return 2 * self.num_freeparam + 2 * self.alt_objective(params)

    def bic(self, params):
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.alt_objective(params)

    def init_params(self, num_components, q, scale=1.0):
        self.num_clust_checker(num_components)
        p = self.num_dim
        self.num_freeparam = num_components * (1 + p + p * q + p) - 1
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": np.random.randn(num_components, p) * scale,
            "fac_loadings": np.random.rand(num_components, p, q) * scale,
            "error": np.random.randn(num_components, p) * scale,
        }

    def unpack_params(self, params):
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        return (
            normalized_log_proportions,
            params["means"],
            params["fac_loadings"],
            params["error"],
        )
    
    def params_checker(self, params, nonneg=True):
        p = self.num_dim
        q = np.shape(params["fac_loadings"])[-1]
        proportions = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            check_dim(log_proportion,()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean,(p,)) and check_finite(mean)
            check_dim(cov_sqrt,(p,q)) and check_finite(cov_sqrt)
            check_dim(error,(p,)) and check_finite(mean)
        check_probdist(np.array(proportions))

    def fac_log_likelihood(self, params, data):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(log_proportion + self.mvn_cov_logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def fac_log_likelihood_alt(self, params, data):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(
                log_proportion + self.mvn_logpdf(data, mean, np.linalg.cholesky(cov).T)
            )
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def labels(self, data, params):
        cluster_lls = []

        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(
                log_proportion + self.mvn_logpdf(data, mean, np.linalg.cholesky(cov).T)
            )

        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.objective, init_params
        )
        
        def callback(flattened_params):
            params = unflatten(flattened_params)
            self.params_checker(params,nonneg=False)
            likelihood = self.likelihood(params)
            if not np.isfinite(likelihood):
                raise ValueError("Log likelihood is {}".format(likelihood))
            print("Log likelihood {}".format(likelihood))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store
