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
