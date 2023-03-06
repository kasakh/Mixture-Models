from .gmm import *
from autograd.scipy.special import gammaln
import autograd.numpy as np

class TMM(GMM):
    def init_params(self, num_components, scale=1.0):
        self.num_clust_checker(num_components)
        D = self.num_dim
        
        self.num_freeparam = num_components * (1+D+0.5*D*(D+1)+1)-1

        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((num_components, D, D)) + np.eye(D),
            "log_dofs": np.random.randn(num_components),
            
        }

    def unpack_params(self, params):
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        dofs = np.exp(params["log_dofs"])
        return normalized_log_proportions, params["means"], params["sqrt_covs"], dofs

    def params_checker(self, params, nonneg=True):
        D = self.num_dim
        proportions = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params)):
            check_dim(log_proportion,()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean,(D,)) and check_finite(mean)
            check_dim(cov_sqrt,(D,D)) and check_finite(cov_sqrt)
            if not nonneg:
                check_posdef(cov_sqrt.T @ cov_sqrt)
            check_dim(dof,()) and check_pos(dof,nonneg)
        check_probdist(np.array(proportions))

    def likelihood(self, params):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params)):
            D = len(mean)
            logpdf = gammaln((dof+D)/2)-gammaln(dof/2)-0.5*np.log(np.linalg.det(np.pi*dof*cov_sqrt.T @ cov_sqrt)) - 0.5*(dof+D)*np.log(1+np.sum(((self.data-mean)@np.linalg.inv(cov_sqrt))**2,axis=1)/dof)
            cluster_lls.append(log_proportion + logpdf)
        return np.sum(logsumexp(np.vstack(cluster_lls),axis=0))

        
    def objective(self, params):
        return -self.likelihood(params)


    def aic(self, params):
        return 2 * self.num_freeparam + 2 * self.objective(params)

    def bic(self, params):
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.objective(params)
    
    def labels(self, data, params_store):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params_store)):
            D = len(mean)
            logpdf = gammaln((dof+D)/2)-gammaln(dof/2)-0.5*np.log(np.linalg.det(np.pi*dof*cov_sqrt.T @ cov_sqrt)) - 0.5*(dof+D)*np.log(1+np.sum(((data-mean)@np.linalg.inv(cov_sqrt))**2,axis=1)/dof)
            cluster_lls.append(log_proportion + logpdf)
        return np.argmax(np.array(cluster_lls).T,axis=1)
    
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