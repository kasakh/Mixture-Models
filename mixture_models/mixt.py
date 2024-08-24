from .gmm import *
from autograd.scipy.special import gammaln
import autograd.numpy as np


class TMM(GMM):
    """
    Class for the mixture model of (multivariate) t-distributions.

    Inherits from the GMM class, and has the same attributes and methods
    (in particular, it is instantiated the same way).
    The differences lie in the parametrization and likelihood/PDF calculation.
    """

    def init_params(self, num_components, scale=1.0, use_kmeans=False, **kwargs):
        """Initialize the mixture-of-t model with random parameters.

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
            log_dofs
              Real vector of shape (num_components)

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
        TMM.likelihood : interprets each parameter in terms of
                         its contribution to the log-likelihood
        GMM.init_params : corresponding method for Gaussian mixture models
        sklearn.cluster.KMeans
        """
        self.num_clust_checker(num_components)
        D = self.num_dim

        self.num_freeparam = num_components * (1 + D + 0.5 * D * (D + 1) + 1) - 1

        return {
            "log proportions": np.random.randn(num_components) * scale,
            "means": self.kmeans(num_components, **kwargs)
            if use_kmeans
            else np.random.randn(num_components, D) * scale,
            "sqrt_covs": np.zeros((num_components, D, D)) + np.eye(D),
            "log_dofs": np.random.randn(num_components),
        }

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `TMM.init_params`.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be used for calculating the model log-likelihood.

        See Also
        --------
        TMM.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])
        dofs = np.exp(params["log_dofs"])
        return normalized_log_proportions, params["means"], params["sqrt_covs"], dofs

    def params_checker(self, params, nonneg=True):
        """Verifies that the model parameters are valid.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters to be checked,
            of the same format as the return value of `TMM.init_params`.
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
        TMM.init_params
        """
        D = self.num_dim
        proportions = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params)):
            check_dim(log_proportion, ()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean, (D,)) and check_finite(mean)
            check_dim(cov_sqrt, (D, D)) and check_finite(cov_sqrt)
            if not nonneg:
                check_posdef(cov_sqrt.T @ cov_sqrt)
            check_dim(dof, ()) and check_pos(dof, nonneg)
        check_probdist(np.array(proportions))

    def likelihood(self, params):
        """Calculates the log-likelihood for the mixture-of-t model.

        Notes
        -----
        The input `params` is to be a dictionary of named parameters,
        of the same format as the return value of `TMM.init_params`.
        In terms of the parameter names `log_proportions`, `means`, `sqrt_covs` and `log_dofs`,
        the formula for the log-likelihood can be written as

        .. math:: \mathrm{logsumexp}\left(\mathtt{log_proportions}^\top f(x|\mathtt{means},\mathtt{sqrt_covs},\mathtt{log_dofs})\right)

        where `f` denots the vector of log-likelihoods for each component,
        with the `i`th component having a multivariate t-distribution

        .. math:: f_i(x|\mu_i,\Sigma_i,\nu_i) = |\pi\nu_i\Sigma_i|^{-1/2} \cdot \frac{\Gamma\left((\nu_i+D)/2\right)}{\Gamma\left(\nu_i/2\right)} \cdot \left[1+(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)/\nu\right]^{-(\nu_i+D)/2}

        evaluated on datapoint x, where D is the number of data dimensions,
        and :math:`\mu_i = \mathtt{means}_i`, :math:`\Sigma_i = \mathtt{sqrt_covs}_i^\top \mathtt{sqrt_covs}_i`, :math:`\nu_i = \exp\left\{\mathtt{log_dofs}_i\right\}`
        are the component's mean, covariance matrix, and degrees of freedom respectively.
        (We parametrize :math:`\Sigma_i` in terms of its (Cholesky) square root,
        and :math:`\nu_i` in terms of their logarithm,
        because they are unconstrained by any assumptions of positivity (or positive definiteness)
        and are hence more amenable to optimization.)

        .. warning:: This implementation is known to be unstable when the number of dimensions D is high.

        See Also
        --------
        TMM.init_params
        """
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params)):
            D = len(mean)
            logpdf = (
                gammaln((dof + D) / 2)
                - gammaln(dof / 2)
                - 0.5 * np.log(np.linalg.det(np.pi * dof * cov_sqrt.T @ cov_sqrt))
                - 0.5
                * (dof + D)
                * np.log(
                    1
                    + np.sum(
                        ((self.data - mean) @ np.linalg.inv(cov_sqrt)) ** 2, axis=1
                    )
                    / dof
                )
            )
            cluster_lls.append(log_proportion + logpdf)
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def objective(self, params):
        """Calculates the negative log-likelihood for the model."""
        return -self.likelihood(params)

    def aic(self, params):
        """Calculates the model AIC (Akaike Information Criterion)."""
        return 2 * self.num_freeparam + 2 * self.objective(params)

    def bic(self, params):
        """Calculates the model BIC (Bayesian Information Criterion)."""
        return np.log(self.num_datapoints) * self.num_freeparam + 2 * self.objective(
            params
        )

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
            the return value of `TMM.init_params`.

        Returns
        -------
        labels : (..., N) ndarray
            A {1,...,num_components}-valued ndarray with as many elements as datapoints.
            Each value corresponds to a "hard" cluster assignment.

        See Also
        --------
        TMM.init_params
        """
        cluster_lls = []
        for log_proportion, mean, cov_sqrt, dof in zip(*self.unpack_params(params)):
            D = len(mean)
            logpdf = (
                gammaln((dof + D) / 2)
                - gammaln(dof / 2)
                - 0.5 * np.log(np.linalg.det(np.pi * dof * cov_sqrt.T @ cov_sqrt))
                - 0.5
                * (dof + D)
                * np.log(
                    1
                    + np.sum(((data - mean) @ np.linalg.inv(cov_sqrt)) ** 2, axis=1)
                    / dof
                )
            )
            cluster_lls.append(log_proportion + logpdf)
        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        """Fits TMM parameters to the data of the model instance.

        This is done by calling the supplied optimization routine `opt_routine`
        with the supplied parameter initializations `init_params`.

        Parameters
        ----------
        init_params : dict
            Dictionary of named parameters, of the same format as
            the return value of `TMM.init_params`.
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
        TMM.objective : loss function for the optimization

        """
        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.objective, init_params
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
