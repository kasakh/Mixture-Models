from .gmm import *


class GMM_Constrainted(GMM):
    def __init__(self, data):
        self.data_checker(data)

    def init_params(self, num_components, scale=1.0):
        self.num_clust_checker(num_components)
        D = self.num_dim
        self.num_freeparam = num_components * (1 + D) - 1 + 0.5 * (D * (1 + D))
        # rs = npr.seed(1)
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((D, D)) + np.eye(D),
        }

    def unpack_params(self, params):
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        lower_triangles = np.array(
            [params["sqrt_covs"]] * np.size(normalized_log_proportions)
        )
        return normalized_log_proportions, params["means"], lower_triangles
    
    def alt_gmm_log_likelihood(self, params, data):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            cov = cov_sqrt.T @ cov_sqrt
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))
    
    def alt_objective(self, params):
        return -self.alt_gmm_log_likelihood(params,self.data)

    def likelihood(self, params):
        return -self.alt_objective(params)
    
    def fit(self, init_params, opt_routine, **kargs):
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.alt_objective, init_params
        )

        def callback(flattened_params):
            params = unflatten(flattened_params)
            print("Log likelihood {}".format(self.likelihood(params)))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store
