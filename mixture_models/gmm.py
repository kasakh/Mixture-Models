from .mixture_models_base import *
from .checkers import *
import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn
from sklearn.cluster import KMeans


class GMM(MM):
    """Class for Gaussian mixture models (GMMs).

    Inherits from the base MM class, and is instantiated the same way.
    It also has the following additional attributes:

    Attributes
    ----------
    num_freeparam : int
        Number of degrees of freedom (used for calculating AIC, BIC etc.)
    params_store : list
        Initialized after calling method `fit`.
        Contains the history of fitted parameters across iterations.

    Methods
    -------
    init_params(num_components, scale=1.0)
        Initializes random GMM parameters, for a given number of components.
    fit(init_params, opt_routine, **kargs)
        Runs optimization routine with the given initialization.
    labels(data, params)
        Returns 'hard' cluster assignments for given data, based on fitted parameters.

    See Also
    --------
    GMM_Constrainted : Child class with constrained common variance.
    TMM : Child class with a mixture of t-distributions instead of Gaussians.
    """

    def objective(self, params):
        """Calculates the negative log-likelihood for the GMM model."""
        kl_cov = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            kl_cov.append(cov_sqrt.T @ cov_sqrt)
        return (
            -1 * self.gmm_log_likelihood(params, self.data)
            - (0 * self.kl_div_tot(params["means"], kl_cov))
            - (-0.0 * self.kl_div_inverse_tot(params["means"], kl_cov))
        )

    def alt_gmm_log_likelihood(self, params, data):
        """Calculates the log-likelihood for the GMM model.

        Notes
        -----
        The input `params` is to be a dictionary of named parameters,
        of the same format as the return value of `GMM.init_params`.
        In terms of the parameter names `log_proportions`, `means` and `sqrt_covs`,
        the formula for the log-likelihood can be written as

        .. math:: \mathrm{logsumexp}\left(\mathtt{log_proportions}^\top f(x|\mathtt{means},\mathtt{sqrt_covs})\right)

        where `f` denots the vector of log-likelihoods for each component,
        with the `i`th component having a multivariate Gaussian distribution

        .. math:: f_i(x|\mu_i,\Sigma_i) = |2\pi\Sigma_i|^{-1/2} \cdot \exp\left\{-(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)/2\right}

        evaluated on datapoint x, where :math:`\mu_i = \mathtt{means}_i` and :math:`\Sigma_i = \mathtt{sqrt_covs}_i^\top \mathtt{sqrt_covs}_i`
        are the component's mean and covariance matrices respectively.
        (We parametrize the latter in terms of its (Cholesky) square root,
        because it is unconstrained by any positive definiteness assumptions,
        and hence more amenable to optimization.)

        See Also
        --------
        GMM.init_params
        """
        cluster_lls = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            cov = cov_sqrt.T @ cov_sqrt
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def alt_objective(self, params):
        """Calculates the negative log-likelihood for the GMM model."""
        return -self.alt_gmm_log_likelihood(params, self.data)

    def likelihood(self, params):
        """Calculates the model log-likelihood."""
        return -self.alt_objective(params)

    def init_params(self, num_components, scale=1.0, use_kmeans=False, **kwargs):
        """Initialize the GMM with random parameters.

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
              Vector of real matrices of shape (num_components, D, D)

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
        GMM.unpack_params : interface between the return value and other methods
        GMM.alt_gmm_log_likelihood : interprets each parameter in terms of
                                     its contribution to the log-likelihood
        GMM.alt_objective : loss function where the unpacked values are used
        sklearn.cluster.KMeans
        """
        self.num_clust_checker(num_components)
        D = self.num_dim

        self.num_freeparam = num_components * (1 + D + 0.5 * (D * (1 + D))) - 1
        # rs = npr.seed(1)
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": self.kmeans(num_components, **kwargs)
            if use_kmeans
            else np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((num_components, D, D)) + np.eye(D),
        }

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `GMM.init_params`.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be used for calculating the model log-likelihood.

        See Also
        --------
        GMM.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        return normalized_log_proportions, params["means"], params["sqrt_covs"]

    def params_checker(self, params, nonneg=True):
        """Verifies that the model parameters are valid.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters to be checked,
            of the same format as the return value of `GMM.init_params`.
        nonneg : bool
            Flag to control the check for covariance matrices,
            i.e. whether they are to be merely positive semidefinite (True)
            or if positive definiteness is required (False).

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Upon test failure.

        See Also
        --------
        GMM.init_params
        """
        D = self.num_dim
        proportions = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            check_dim(log_proportion, ()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean, (D,)) and check_finite(mean)
            check_dim(cov_sqrt, (D, D)) and check_finite(cov_sqrt)
            if not nonneg:
                check_posdef(cov_sqrt.T @ cov_sqrt)
        check_probdist(np.array(proportions))

    def aic(self, params):
        """Calculates the model AIC (Akaike Information Criterion)."""
        return 2 * self.num_freeparam + 2 * self.alt_objective(params)

    def bic(self, params):
        """Calculates the model BIC (Bayesian Information Criterion)."""
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.alt_objective(params)

    def kmeans(self, num_components, **kwargs):
        """Runs k-means on the data to obtain a reasonable initialization."""
        return KMeans(num_components, **kwargs).fit(self.data).cluster_centers_

    def labels(self, data, params):
        """Assigns clusters to data, based on given parameters.

        This cluster assignment is "hard", in the sense that it returns one cluster for each data point,
        unlike "soft" classifiers that return a probability distribution over clusters for each data point.

        Parameters
        ----------
        data : (..., N, D) ndarray
            D-dimensional input of (..., N) datapoints to be labelled.
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `GMM.init_params`.

        Returns
        -------
        labels : (..., N) ndarray
            A {1,...,num_components}-valued ndarray with as many elements as datapoints.
            Each value corresponds to a "hard" cluster assignment.

        See Also
        --------
        GMM.init_params
        """
        cluster_lls = []

        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            # calculate a stable version of mn

            cluster_lls.append(
                log_proportion + mvn.logpdf(data, mean, cov_sqrt.T @ cov_sqrt)
            )

        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        """Fits GMM parameters to the data of the model instance.

        This is done by calling the supplied optimization routine `opt_routine`
        with the supplied parameter initializations `init_params`.

        Parameters
        ----------
        init_params : dict
            Dictionary of named parameters, of the same format as
            the return value of `GMM.init_params`.
            This is used an initial input to the optimization routine.
        opt_routine : {'grad_descent','rms_prop','adam','Newton-CG'}
            The optimization routine to be called.
            Must be one of the strings listed above.
        **kargs : dict, optional
        Other parameters (passed to the `opt_routine` call).

        Returns
        -------
        None
            However, the attribute `params_store` is updated as a side effect.

        See Also
        --------
        GMM.alt_objective : loss function for the optimization

        """
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.alt_objective, init_params
        )

        def callback(flattened_params):
            params = unflatten(flattened_params)
            self.params_checker(params, nonneg=False)
            self.report_likelihood(self.likelihood(params))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store
