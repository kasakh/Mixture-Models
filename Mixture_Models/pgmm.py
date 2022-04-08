from .mfa import *


class PGMM(MFA):
    def __init__(self, data, constraint="CCC"):
        self.constraint_set = {"CCC", "CCU", "CUC", "CUU", "UCC", "UCU", "UUC", "UUU"}
        if constraint in self.constraint_set:
            self.constraint = constraint
        else:
            raise ValueError(
                'constraint should be one of {"CCC","CCU","CUC","CUU","UCC","UCU","UUC","UUU"} '
            )

        self.data_checker(data)

    def init_params(self, num_components, q, scale=1.0):
        self.num_clust_checker(num_components)
        p = self.num_dim
        return_dict = {}
        return_dict["log proportions"] = np.random.randn(num_components) * scale
        return_dict["means"] = np.random.randn(num_components, p) * scale

        if self.constraint == "CCC":
            return_dict["fac_loadings"] = np.random.randn(p, q) * scale
            return_dict["error"] = np.random.randn(1) * scale
            self.num_freeparam = p * q - 0.5 * q * (q - 1) + 1

        if self.constraint == "UCC":
            return_dict["fac_loadings"] = np.random.randn(num_components, p, q) * scale
            return_dict["error"] = np.random.randn(1) * scale
            self.num_freeparam = num_components * (p * q - 0.5 * q * (q - 1)) + 1

        if self.constraint == "UCU":
            return_dict["fac_loadings"] = np.random.randn(num_components, p, q) * scale
            return_dict["error"] = np.random.randn(p) * scale
            self.num_freeparam = num_components * (p * q - 0.5 * q * (q - 1)) + p

        if self.constraint == "UUU":
            return_dict["fac_loadings"] = np.random.randn(num_components, p, q) * scale
            return_dict["error"] = np.random.randn(num_components, p) * scale
            self.num_freeparam = num_components * (p * q - 0.5 * q * (q - 1) + p)

        if self.constraint == "UUC":
            return_dict["fac_loadings"] = np.random.randn(num_components, p, q) * scale
            return_dict["error"] = np.random.randn(num_components) * scale
            self.num_freeparam = num_components * (p * q - 0.5 * q * (q - 1) + 1)

        if self.constraint == "CUC":
            return_dict["fac_loadings"] = np.random.randn(p, q) * scale
            return_dict["error"] = np.random.randn(num_components) * scale
            self.num_freeparam = num_components + p * q - 0.5 * q * (q - 1)

        if self.constraint == "CUU":
            return_dict["fac_loadings"] = np.random.randn(p, q) * scale
            return_dict["error"] = np.random.randn(num_components, p) * scale
            self.num_freeparam = p * q - 0.5 * q * (q - 1) + num_components * (p)

        if self.constraint == "CCU":
            return_dict["fac_loadings"] = np.random.randn(p, q) * scale
            return_dict["error"] = np.random.randn(p) * scale
            self.num_freeparam = p * q - 0.5 * q * (q - 1) + p

        return return_dict

    def unpack_params(self, params):
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        if self.constraint == "CCC":
            fac_loadings = np.array(
                [params["fac_loadings"]] * np.size(normalized_log_proportions)
            )
            error = np.array(
                [[params["error"]] * self.num_dim] * np.size(normalized_log_proportions)
            )
        if self.constraint == "UCC":
            fac_loadings = params["fac_loadings"]
            error = np.array(
                [[params["error"]] * self.num_dim] * np.size(normalized_log_proportions)
            )
        if self.constraint == "UCU":
            fac_loadings = params["fac_loadings"]
            error = np.array([params["error"]] * np.size(normalized_log_proportions))

        if self.constraint == "UUU":
            fac_loadings = params["fac_loadings"]
            error = params["error"]

        if self.constraint == "UUC":
            fac_loadings = params["fac_loadings"]
            error = np.array([[i] * self.num_dim for i in params["error"]])
        #             np.array([[params['error']]*self.num_dim ]*np.size(normalized_log_proportions))

        if self.constraint == "CUC":
            fac_loadings = np.array(
                [params["fac_loadings"]] * np.size(normalized_log_proportions)
            )
            error = np.array([[i] * self.num_dim for i in params["error"]])

        if self.constraint == "CUU":
            fac_loadings = np.array(
                [params["fac_loadings"]] * np.size(normalized_log_proportions)
            )
            error = params["error"]

        if self.constraint == "CCU":
            fac_loadings = np.array(
                [params["fac_loadings"]] * np.size(normalized_log_proportions)
            )
            error = np.array([params["error"]] * np.size(normalized_log_proportions))

        return normalized_log_proportions, params["means"], fac_loadings, error
