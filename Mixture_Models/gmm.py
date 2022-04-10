from .mixture_models import *


class GMM(MM):
    def __init__(self, data):
        self.data_checker(data)

    def objective(self, params):
        kl_cov = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            kl_cov.append(cov_sqrt.T @ cov_sqrt)
        return (
            -1 * self.gmm_log_likelihood(params, self.data)
            - (0 * self.kl_div_tot(params["means"], kl_cov))
            - (-0.0 * self.kl_div_inverse_tot(params["means"], kl_cov))
        )

    def alt_objective(self, params):
        return -self.alt_gmm_log_likelihood(params, self.data)

    def likelihood(self, params):
        return -self.alt_objective(params)

    def init_params(self, num_components, scale=1.0):
        self.num_clust_checker(num_components)
        D = self.num_dim

        self.num_freeparam = num_components * (1 + D + 0.5 * (D * (1 + D))) - 1
        # rs = npr.seed(1)
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((num_components, D, D)) + np.eye(D),
        }

    def unpack_params(self, params):
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        return normalized_log_proportions, params["means"], params["sqrt_covs"]

    def aic(self, params):
        return 2 * self.num_freeparam + 2 * self.alt_objective(params)

    def bic(self, params):
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.alt_objective(params)

    def labels(self, data, params_store):
        cluster_lls = []

        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params_store)):

            cluster_lls.append(log_proportion + self.mvn_logpdf(data, mean, cov_sqrt))

        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.objective, init_params
        )

        def callback(flattened_params):
            params = unflatten(flattened_params)
            print("Log likelihood {}".format(self.likelihood(params)))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store
