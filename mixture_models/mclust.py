from .mixture_models_base import *
from .checkers import *
import autograd.numpy as np
from sklearn.cluster import KMeans


class Mclust(MM):
    """
    Class for Gaussian mixture models (GMMs) as specified in the R package `mclust` (Fraley and Raftery, 2003).

    Instances of this class are GMMs with constraints on their component covariance matrices,
    similar in concept to the class `GMM.Constrainted`
    (which may be viewed as a special case of this class corresponding to the constraint 'EEE').

    Parameters
    ----------
    data : matrix
        Input data to be fitted. Must be a real matrix with finite values,
        e.g. missing values (NA, NaN), +Inf, -Inf are all forbidden.
    constraint : {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'}
        Name of the model to be initialized.
        The interpretation of these identifiers follows the `mclust` package.

    Attributes
    ----------
    constraint : {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'}
        The constraint specified on initialization.
    num_freeparam : int
        Number of degrees of freedom (used for calculating AIC, BIC etc.)
    params_store : list
        Initialized after calling method `fit`.
        Contains the history of fitted parameters across iterations.
    data, num_dim, num_datapoints
        Other attributes inherited from the base mixture model class `MM`.

    Methods
    -------
    init_params(num_components, scale=1.0)
        Initializes random Mclust parameters, for a given number of components.
    fit(init_params, opt_routine, **kargs)
        Runs optimization routine with the given initialization.
    labels(data, params)
        Returns 'hard' cluster assignments for given data, based on fitted parameters.

    See Also
    --------
    GMM_Constrainted

    Notes
    -----
    `mclust` is a freely available software package for R, which is in turn
    an open-source environment for statistical computing [1]_,
    and can be obtained from the cited URL.

    The `constraint` attribute identifies a GMM
    whose specification is documented in Section 2 of the paper [2]_.
    A copy of this paper is available at the URL [3]_.

    References
    ----------

    .. [1] R Core Team (2023). R: A language and environment for statistical
       computing. R Foundation for Statistical Computing, Vienna, Austria.
       URL https://www.R-project.org/.

    .. [2] Fraley, Chris, and Adrian E. Raftery. "Enhanced model-based clustering,
       density estimation, and discriminant analysis software: MCLUST."
       Journal of Classification 20.2 (2003): 263-286.

    .. [3] https://sites.stat.washington.edu/raftery/Research/PDF/fraley2003.pdf
    """

    def __init__(self, data, constraint="VVV"):
        self.constraint_set = {
            "EII",
            "VII",
            "EEI",
            "VEI",
            "EVI",
            "VVI",
            "EEE",
            "VEE",
            "EVE",
            "VVE",
            "EEV",
            "VEV",
            "EVV",
            "VVV",
        }
        if constraint in self.constraint_set:
            self.constraint = constraint
        else:
            raise ValueError(
                "constraint should be one of {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'} "
            )

        super().__init__(data)

    def init_params(self, num_components, scale=1.0, use_kmeans=False, **kwargs):
        """
        Initializes the Mclust model with random parameters.

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
            Dictionary of named parameters, whose entries depend on
            the `constraint` attribute set upon initialization.

            Consists of the following entries:

            log proportions
              Real vector of shape (num_components)
            means
              Real matrix of shape (num_components, D)
            log volumes
              Real vector of shape (1) or (num_components), depending on constraint
            log shapes
              Real array of shape () or (D) or (num_components, D), depending on constraint
            orientations
              Real array of shape () or (D,D) or (num_components, D, D), depending on constraint

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
        Mclust.unpack_params : interface between the return value and other methods
        Mclust.likelihood : interprets each parameter in terms of
                            its contribution to the log-likelihood
        Mclust.objective : loss function where the unpacked values are used
        sklearn.cluster.KMeans
        """
        self.num_clust_checker(num_components)
        D = self.num_dim
        return_dict = {}
        return_dict["log proportions"] = np.random.randn(num_components) * scale
        if use_kmeans:
            return_dict["means"] = self.kmeans(num_components, **kwargs)
        else:
            return_dict["means"] = np.random.randn(num_components, D) * scale
        self.num_freeparam = num_components * (1 + D) - 1

        if self.constraint[0] == "E":
            return_dict["log volumes"] = np.random.randn(1) * scale
            self.num_freeparam += 1
        elif self.constraint[0] == "V":
            return_dict["log volumes"] = np.random.randn(num_components) * scale
            self.num_freeparam += num_components

        if self.constraint[1] == "I":
            return_dict["log shapes"] = np.random.randn(0)
        elif self.constraint[1] == "E":
            return_dict["log shapes"] = np.random.randn(D) * scale
            self.num_freeparam += D - 1
        elif self.constraint[1] == "V":
            return_dict["log shapes"] = np.random.randn(num_components, D) * scale
            self.num_freeparam += num_components * (D - 1)

        if self.constraint[2] == "I":
            return_dict["orientations"] = np.random.randn(0)
        elif self.constraint[2] == "E":
            return_dict["orientations"] = np.random.randn(D, D)
            self.num_freeparam += D * (D - 1) * 0.5
        elif self.constraint[2] == "V":
            return_dict["orientations"] = np.random.randn(num_components, D, D)
            self.num_freeparam += num_components * D * (D - 1) * 0.5

        return return_dict

    def unpack_params(self, params):
        """Expands a dictionary of named parameters into a tuple.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters, of the same format as
            the return value of `Mclust.init_params`,
            and compatible with the instance value of the `constraint` attribute.

        Returns
        -------
        expanded_params : tuple
            A tuple of expanded model parameters,
            as can be passed to `Mclust.likelihood` for calculation.

        Notes
        -----
        The constraint-dependent parameters `log volumes`, `log shapes` and `orientations`
        are expanded by repetition (as necessary depending on the `constraint`)
        so as to form an eigendecomposition of the component covariance matrices,
        as required by the `Mclust` specification.
        Specifically, the covariance matrix of the i-th component is to be given by

        .. math:: \Sigma_i = \mathtt{volume}_i \times \mathtt{orientation}_i\mathtt{shape}_i\mathtt{orientation}_i^\top

        where the scale factors :math:`\mathtt{volume} > 0` are the exponential of `log volumes`,
        the positive diagonal matrix with unit determinant :math:`\mathtt{shape}`
        is constructed from the normalized exponential of the vector of `log shapes`,
        and the orthogonal matrix :math:`\mathtt{orientation}` is derived from `orientations`
        which is encoded as a skew-symmetric matrix using Cayley's parametrization [1]_.
        (Without loss of generality, we may assume that :math:`\mathtt{orientation}` is positive definite,
        otherwise just flip each column (i.e. eigenvector) corresponding to an eigenvalue of -1.)

        References
        ----------

        .. [1] https://planetmath.org/cayleysparameterizationoforthogonalmatrices

        See Also
        --------
        Mclust: contains the original reference for the eigendecomposition parametrization of the Mclust mixture model
        Mclust.init_params, Mclust.likelihood
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])

        num_components = len(normalized_log_proportions)
        D = self.num_dim

        if self.constraint[0] == "E":
            volumes = np.exp(params["log volumes"].repeat(num_components))
        elif self.constraint[0] == "V":
            volumes = np.exp(params["log volumes"])

        if self.constraint[1] == "I":
            shapes = np.zeros((num_components, D, D)) + np.eye(D)
        elif self.constraint[1] == "E":
            shapes = np.zeros((num_components, D, D)) + np.diag(
                np.exp(params["log shapes"] - np.sum(params["log shapes"]) / D)
            )
        elif self.constraint[1] == "V":
            shapes = np.array(
                [
                    np.diag(row)
                    for row in np.exp(
                        params["log shapes"] - np.sum(params["log shapes"]) / D
                    )
                ]
            )

        if self.constraint[2] == "I":
            orientations = np.zeros((num_components, D, D)) + np.eye(D)
        elif self.constraint[2] == "E":
            skew_symm = (
                params["orientations"] - np.transpose(params["orientations"])
            ) / 2
            orientations = np.zeros((num_components, D, D)) + (
                (np.eye(D) + skew_symm) @ np.linalg.inv(np.eye(D) - skew_symm)
            )
        elif self.constraint[2] == "V":
            skew_symm = (
                params["orientations"] - np.transpose(params["orientations"], (0, 2, 1))
            ) / 2
            orientations = (np.eye(D) + skew_symm) @ np.linalg.inv(
                np.eye(D) - skew_symm
            )

        return (
            normalized_log_proportions,
            params["means"],
            shapes,
            orientations,
            volumes,
        )

    def params_checker(self, params, nonneg=True):
        """Verifies that the model parameters are valid.

        Parameters
        ----------
        params : dict
            Dictionary of named parameters to be checked,
            of the same format as the return value of `Mclust.init_params`,
            and compatible with the instance value of the `constraint` attribute.
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
        Mclust.init_params
        """
        D = self.num_dim
        proportions = []
        for log_proportion, mean, shape, orientation, volume in zip(
            *self.unpack_params(params)
        ):
            check_dim(log_proportion, ()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean, (D,)) and check_finite(mean)
            check_dim(shape, (D, D)) and check_diagonal(shape) and check_pos(
                np.diag(shape), nonneg
            )
            check_dim(orientation, (D, D)) and check_orthogonal(orientation)
            check_dim(volume, ()) and check_pos(volume, nonneg)
        check_probdist(np.array(proportions))

    def likelihood(self, params):
        """Calculates the log-likelihood for the Mclust model.

        Notes
        -----
        The input `params` is to be a dictionary of named parameters,
        of the same format as the return value of `Mclust.init_params`,
        and compatible with the instance value of the `constraint` attribute.
        This method relies on the subroutine `unpack_params`
        to interpret the input dictionary of named parameters
        and assemble them into a tuple of parameters
        `log_proportions`, `means`, `shape`, `orientation` and `volume`.

        The formula for the log-likelihood can be written as

        .. math:: \mathrm{logsumexp}\left(\mathtt{log_proportions}^\top f(x|\mathtt{means},\mathtt{volume} \times \mathtt{orientation}\mathtt{shape} \mathtt{orientation}^\top)\right)

        where `f` denots the vector of log-likelihoods for each component,
        with the `i`th component having a multivariate Gaussian distribution

        .. math:: f_i(x|\mu_i,\Sigma_i) = |2\pi\Sigma_i|^{-1/2} \cdot \exp\left\{-(x-\mu_i)^T\Sigma_i^{-1}(x-\mu_i)/2\right}

        evaluated on datapoint x, where :math:`\mu_i = \mathtt{means}_i` and :math:`\Sigma_i`
        are the component's mean and covariance matrices respectively,
        with the latter being defined in terms of the remaining parameters as

        .. math:: \Sigma_i = \mathtt{volume}_i \times \mathtt{orientation}_i\mathtt{shape}_i\mathtt{orientation}_i^\top

        in other words, they form an eigendecomposition of :math:`\Sigma_i`.

        See Also
        --------
        Mclust.unpack_params : constructs the eigendecomposition
        Mclust.init_params
        """
        cluster_lls = []
        for log_proportion, mean, shape, orientation, volume in zip(
            *self.unpack_params(params)
        ):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(self.data, mean, cov))

        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def objective(self, params):
        """Calculates the negative log-likelihood for the Mclust model."""
        return -self.likelihood(params)

    def aic(self, params):
        """Calculates the model AIC (Akaike Information Criterion)."""
        return 2 * self.num_freeparam + 2 * self.objective(params)

    def bic(self, params):
        """Calculates the model BIC (Bayesian Information Criterion)."""
        return np.log(self.num_datapoints) * self.num_freeparam + 2 * self.objective(
            params
        )

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
            the return value of `Mclust.init_params`,
            and compatible with the instance value of the `constraint` attribute.

        Returns
        -------
        labels : (..., N) ndarray
            A {1,...,num_components}-valued ndarray with as many elements as datapoints.
            Each value corresponds to a "hard" cluster assignment.

        See Also
        --------
        Mclust.init_params
        """
        cluster_lls = []
        for log_proportion, mean, shape, orientation, volume in zip(
            *self.unpack_params(params)
        ):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))

        return np.argmax(np.array(cluster_lls).T, axis=1)

    def fit(self, init_params, opt_routine, **kargs):
        """Fits Mclust parameters to the data of the model instance.

        This is done by calling the supplied optimization routine `opt_routine`
        with the supplied parameter initializations `init_params`.

        Parameters
        ----------
        init_params : dict
            Dictionary of named parameters, of the same format as
            the return value of `Mclust.init_params`,
            and compatible with the instance value of the `constraint` attribute.
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
        Mclust.objective : loss function for the optimization

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
