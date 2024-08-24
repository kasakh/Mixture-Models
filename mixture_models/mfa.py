from .mixture_models_base import *
from .checkers import *
import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn
from sklearn.cluster import KMeans


class MFA(MM):
    """Class for mixture-of-factor-analyzers (MFA) models.

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
        Initializes random MFA parameters, for a given number of components.
    fit(init_params, opt_routine, **kargs)
        Runs optimization routine with the given initialization.
    labels(data, params)
        Returns 'hard' cluster assignments for given data, based on fitted parameters.

    See Also
    --------
    PGMM : Child class allowing for constrained loading and/or error matrices.
    """

    def objective(self, params):
        """Calculates the negative log-likelihood for the MFA model."""
        return -self.fac_log_likelihood(params, self.data)

    def mvn_cov_logpdf(self, X, mu, cov):
        """Computes the multivariate normal log probability density function.

        Parameters
        ----------
        X : (..., N, D) ndarray
            Arguments at which the normal log-pdf is to be evaluated.
            Here D is the number of dimensions, and N the number of arguments.
        mu : (..., D) vector
            Mean vector. Must be broadcastable to the same shape as X.
        cov : (..., D, D) matrix
            Positive definite symmetric matrix. Must be conformable with `X-mu`.

        Returns
        -------
        mvn_logpdf : (..., N)
            Values of the D-dimensional normal log-pdf evaluated at X.

        Notes
        -----
        The multivariate normal probability density function is specified by

        .. math:: f(x|\mu,\Sigma) = |2\pi\Sigma|^{-1/2} \cdot \exp\left\{-(x-\mu)^T\Sigma^{-1}(x-\mu)/2\right}

        for argument :math:`x`, mean vector :math:`\mu` and covariance matrix :math:`\Sigma`.

        This method computes the above function and returns the logarithm :math:`\log f`.
        """
        return -0.5 * np.log(np.linalg.det(2 * np.pi * cov)) - 0.5 * np.sum(
            np.dot((X - mu), np.linalg.inv(cov)) * (X - mu), axis=1
        )

    def alt_objective(self, params):
        """Calculates the negative log-likelihood for the MFA model."""
        return -self.fac_log_likelihood(params, self.data)

    def likelihood(self, params):
        """Calculates the log-likelihood for the model."""
        return -self.alt_objective(params)

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

    def init_params(self, num_components, q, scale=1.0, use_kmeans=False, **kwargs):
        """Initialize the MFA with random parameters.

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
            Dictionary of named parameters. Consists of the following entries:

            log proportions
              Real vector of shape (num_components)
            means
              Real matrix of shape (num_components, p)
            fac_loadings
              Vector of real matrices of shape (num_components, p, q)
            error
              Real matrix of shape (num_components, p)

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
        MFA.unpack_params : interface between the return value and other methods
        MFA.fac_log_likelihood : interprets each parameter in terms of
                                 its contribution to the log-likelihood
        MFA.objective : loss function where the unpacked values are used
        sklearn.cluster.KMeans
        """
        self.num_clust_checker(num_components)
        p = self.num_dim
        self.num_freeparam = (
            num_components * (1 + p + p * q - 0.5 * q * (q - 1) + p) - 1
        )
        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": self.kmeans(num_components, **kwargs)
            if use_kmeans
            else np.random.randn(num_components, p) * scale,
            "fac_loadings": np.random.rand(num_components, p, q) * scale,
            "error": np.random.randn(num_components, p) * scale,
        }

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `MFA.init_params`.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be used for calculating the model log-likelihood.

        See Also
        --------
        MFA.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        return (
            normalized_log_proportions,
            params["means"],
            params["fac_loadings"],
            params["error"],
        )

    def params_checker(self, params):
        """Verifies that the model parameters are valid.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters to be checked,
            of the same format as the return value of `MFA.init_params`.

        Returns
        -------
        None

        Raises
        ------
        AssertionError
            Upon test failure.

        See Also
        --------
        MFA.init_params
        """
        p = self.num_dim
        q = np.shape(params["fac_loadings"])[-1]
        proportions = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            check_dim(log_proportion, ()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean, (p,)) and check_finite(mean)
            check_dim(cov_sqrt, (p, q)) and check_finite(cov_sqrt)
            check_dim(error, (p,)) and check_finite(mean)
        check_probdist(np.array(proportions))

    def fac_log_likelihood(self, params, data):
        """Calculates the log-likelihood for the MFA model.

        Notes
        -----
        The input `params` is to be a dictionary of named parameters,
        of the same format as the return value of `MFA.init_params`.
        In terms of the parameter names `log_proportions`, `means`, `cov_sqrt` and `error`,
        the formula for the log-likelihood can be written as

        .. math:: \mathrm{logsumexp}\left(\mathtt{log_proportions}^\top f(x|\mathtt{means},\mathtt{cov_sqrts},\mathtt{error})\right)

        where `f` denots the vector of log-likelihoods for each component,
        with the `i`th component having a multivariate Gaussian distribution

        .. math:: f_i(x|\mu_i,\Sigma_i) = |2\pi\Sigma_i|^{-1/2} \cdot \exp\left\{-(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)/2\right}

        evaluated on datapoint x, where :math:`\mu_i = \mathtt{means}_i` and :math:`\Sigma_i`
        are the component's mean and covariance matrices respectively,
        with the latter being defined in terms of the remaining parameters as

        .. math:: \Sigma_i = \mathtt{cov_sqrts}_i\mathtt{cov_sqrts}_i^\top + diag\left[\mathtt{error}_i \circ \mathtt{error}_i\right]

        where :math:`\circ` is the Hadamard product of the vector :math:`\mathtt{error}_i` with itself,
        and :math`diag[\dots]` forms a diagonal matrix with this vector as its main entries.

        See Also
        --------
        MFA.init_params
        """
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def fac_log_likelihood_alt(self, params, data):
        """Alternate expression for the log-likelihood of the MFA model.

        See Also
        --------
        MFA.fac_log_likelihood
        """
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(
                log_proportion + self.mvn_logpdf(data, mean, np.linalg.cholesky(cov).T)
            )
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

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
            the return value of `MFA.init_params`.

        Returns
        -------
        labels : (..., N) ndarray
            A {1,...,num_components}-valued ndarray with as many elements as datapoints.
            Each value corresponds to a "hard" cluster assignment.

        See Also
        --------
        MFA.init_params
        """
        cluster_lls = []

        for log_proportion, mean, cov_sqrt, error in zip(*self.unpack_params(params)):
            cov = (cov_sqrt @ cov_sqrt.T) + (np.diag(error) @ np.diag(error))
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))

        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        """Fits MFA parameters to the data of the model instance.

        This is done by calling the supplied optimization routine `opt_routine`
        with the supplied parameter initializations `init_params`.

        Parameters
        ----------
        init_params : dict
            Dictionary of named parameters, of the same format as
            the return value of `MFA.init_params`.
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
        MFA.objective : loss function for the optimization

        """
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.objective, init_params
        )

        def callback(flattened_params):
            params = unflatten(flattened_params)
            self.params_checker(params)
            self.report_likelihood(self.likelihood(params))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store
