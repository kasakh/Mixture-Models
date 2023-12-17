from .gmm import *


class GMM_Constrainted(GMM):
    """
    Class for Gaussian mixture models (GMMs) with common variance.

    Inherits from the GMM class, and has the same attributes and methods
    (in particular, it is instantiated the same way).
    """

    def init_params(self, num_components, scale=1.0, use_kmeans=False, **kwargs):
        """Initialize the constrained GMM with random parameters.

        Parameters
        ----------
        num_components : int
            Number of mixture components (i.e. clusters) to be fitted.
            Must be a positive integer :math:`\geq` number of input datapoints.

        scale : float, optional
            Scale parameter, defaults to 1.
            Corresponds roughly to the amount of heterogeneity between clusters.

        Returns
        -------
        init_params : dict
            Dictionary of named parameters. Consists of the following entries:

            log proportions
              Real vector of shape (num_components)
            means
              Real matrix of shape (num_components, D)
            sqrt_covs
              Real matrix of shape (D, D)

            where D = number of data dimensions.

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
        GMM.init_params : corresponding method for unconstrained GMMs
        sklearn.cluster.KMeans
        """
        self.num_clust_checker(num_components)
        D = self.num_dim
        self.num_freeparam = num_components * (1 + D) - 1 + 0.5 * (D * (1 + D))
        # rs = npr.seed(1)
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": self.kmeans(num_components, **kwargs)
            if use_kmeans
            else np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((D, D)) + np.eye(D),
        }

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `GMM_Constrainted.init_params`.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be used for calculating the model log-likelihood.

        See Also
        --------
        GMM_Constrainted.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        lower_triangles = np.array(
            [params["sqrt_covs"]] * np.size(normalized_log_proportions)
        )
        return normalized_log_proportions, params["means"], lower_triangles
