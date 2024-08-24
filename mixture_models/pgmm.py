from .mfa import *
import autograd.numpy as np


class PGMM(MFA):
    """
    Class for parsimonious Gaussian mixture models (PGMMs) as specified in McNicholas and Murphy (2008).

    Instances of this class are MFA models with constraints on their component covariance matrices,
    and inherit from that class.

    Parameters
    ----------
    data : matrix
        Input data to be fitted. Must be a real matrix with finite values,
        e.g. missing values (NA, NaN), +Inf, -Inf are all forbidden.
    constraint : {'CCC','CCU','CUC','CUU','UCC','UCU','UUC','UUU'}
        Name of the model to be initialized.
        The interpretation of these identifiers follows the specification in [1]_.

    Attributes
    ----------
    constraint : {'CCC','CCU','CUC','CUU','UCC','UCU','UUC','UUU'}
        The constraint specified on initialization.
    num_freeparam : int
        Number of degrees of freedom (used for calculating AIC, BIC etc.)
    params_store : list
        Initialized after calling method `fit`.
        Contains the history of fitted parameters across iterations.

    Methods
    -------
    init_params(num_components, scale=1.0)
        Initializes random PGMM parameters, for a given number of components.
    fit(init_params, opt_routine, **kargs)
        Runs optimization routine with the given initialization.
    labels(data, params)
        Returns 'hard' cluster assignments for given data, based on fitted parameters.

    See Also
    --------
    MFA

    References
    ----------

    .. [1] McNicholas, Paul David, and Thomas Brendan Murphy. "Parsimonious Gaussian mixture models."
       Statistics and Computing 18 (2008): 285-296.
    """

    def __init__(self, data, constraint="CCC"):
        self.constraint_set = {"CCC", "CCU", "CUC", "CUU", "UCC", "UCU", "UUC", "UUU"}
        if constraint in self.constraint_set:
            self.constraint = constraint
        else:
            raise ValueError(
                'constraint should be one of {"CCC","CCU","CUC","CUU","UCC","UCU","UUC","UUU"} '
            )
        super().__init__(data)

    def init_params(self, num_components, q, scale=1.0, use_kmeans=False, **kwargs):
        """Initialize the PGMM with random parameters.

        Parameters
        ----------
        num_components : int
            Number of mixture components (i.e. clusters) to be fitted.
            Must be a positive integer :math:`\geq` number of input datapoints.

        q : int
            Number of latent (unobserved) factors in the covariance matrices.
            Must be a positive integer < number of data dimensions.

        scale : float, optional
            Scale parameter, defaults to 1.
            Corresponds roughly to the amount of heterogeneity between clusters.

        Returns
        -------
        init_params : dict
            Dictionary of named parameters, whose entries depend on
            the `constraint` attribute set upon initialization.

            Consists of the following entries:

            log proportions
              Real vector of shape (num_components)
            means
              Real matrix of shape (num_components, p)
            fac_loadings
              Vector of real matrices of shape (p, q) or (num_components, p, q), depending on constraint
            error
              Real matrix of shape (1) or (p) or (num_components, p), depending on constraint

            where p = number of data dimensions.

        Other Parameters
        ----------------
        use_kmeans : bool, optional
            If true, `means` are initialized from a fit of the k-means clustering algorithm,
            otherwise (the default) they are randomized like the other parameters.
        **kwargs : dict
            Optional arguments passed to the k-means algorithm,
            specifically to the implementation `sklearn.cluster.KMeans`.

        See Also
        --------
        MFA.init_params : corresponding method for unconstrained mixture-of-factor-analyzer models
        """
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

        self.num_freeparam = self.num_freeparam + num_components * (1 + p) - 1

        return return_dict

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `PGMM.init_params`,
            and compatible with the instance value of the `constraint` attribute.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be used for calculating the model log-likelihood.

        Notes
        -----
        This methods expands the constrained parameters by repetition
        until they have the same dimensionality as the unconstrained MFA model.

        See Also
        --------
        MFA.unpack_params : corresponding method for MFA models
        PGMM.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        if self.constraint == "CCC":
            fac_loadings = np.array(
                [params["fac_loadings"]] * np.size(normalized_log_proportions)
            )
            error = np.array(
                [[params["error"][0]] * self.num_dim]
                * np.size(normalized_log_proportions)
            )
        if self.constraint == "UCC":
            fac_loadings = params["fac_loadings"]
            error = np.array(
                [[params["error"][0]] * self.num_dim]
                * np.size(normalized_log_proportions)
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
