from .mixture_models import *
from .checkers import *
import autograd.numpy as np

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
        self.constraint_set = {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'}
        if constraint in self.constraint_set:
            self.constraint = constraint
        else:
            raise ValueError(
                "constraint should be one of {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'} "
            )

        super().__init__(data)


    def init_params(self, num_components, scale=1.0):
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
            the instance
            
            Consists of the following entries:

            log proportions
              Real vector of shape (num_components)
            means
              Real matrix of shape (num_components, D)
            sqrt_covs
              Vector of real matrices of shape (num_components, D, D)
            
            where D = number of data dimensions.
        
        See Also
        --------
        GMM.unpack_params : interface between the return value and other methods
        GMM.alt_gmm_log_likelihood : interprets each parameter in terms of
                                     its contribution to the log-likelihood
        GMM.alt_objective : loss function where the unpacked values are used
        """
        '''
        'log volumes' = log lambda (or log lambda_k)
        'log shapes' = log (diag A) (or log diag A_k) (n.b. product of diagonal elements must = 1)
        'orientations' = Cayley parametrization of the orthogonal matrix D (or D_k) of eigenvectors
        (see https://planetmath.org/cayleysparameterizationoforthogonalmatrices for reference).
        WLOG we may assume all eigenvalues of D are +1 rather than -1;
        otherwise we could just replace D by DE, where E is the diagonal matrix of eigenvalues of D
        (i.e. diag(E) consists only of +/-1) without changing the covariance matrix.
        '''
        self.num_clust_checker(num_components)
        D = self.num_dim
        return_dict = {}
        return_dict["log proportions"] = np.random.randn(num_components) * scale
        return_dict["means"] = np.random.randn(num_components, D) * scale
        self.num_freeparam = num_components*(1+D)-1

        if self.constraint[0] == 'E':
            return_dict["log volumes"] = np.random.randn(1) * scale
            self.num_freeparam += 1
        elif self.constraint[0] == 'V':
            return_dict["log volumes"] = np.random.randn(num_components) * scale
            self.num_freeparam += num_components

        if self.constraint[1] == 'I':
            return_dict["log shapes"] = np.random.randn(0)
        elif self.constraint[1] == 'E':
            return_dict["log shapes"] = np.random.randn(D) * scale
            self.num_freeparam += D-1
        elif self.constraint[1] == 'V':
            return_dict["log shapes"] = np.random.randn(num_components, D) * scale
            self.num_freeparam += num_components*(D-1)

        if self.constraint[2] == 'I':
            return_dict["orientations"] = np.random.randn(0)
        elif self.constraint[2] == 'E':
            return_dict["orientations"] = np.random.randn(D,D)
            self.num_freeparam += D*(D-1)*0.5
        elif self.constraint[2] == 'V':
            return_dict["orientations"] = np.random.randn(num_components,D,D)
            self.num_freeparam += num_components*D*(D-1)*0.5
        
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
            as can be used for calculating the model log-likelihood.
        
        See Also
        --------
        Mclust.init_params
        """
        normalized_log_proportions = self.log_normalize(params["log proportions"])

        num_components = len(normalized_log_proportions)
        D = self.num_dim

        if self.constraint[0] == 'E':
            volumes = np.exp(params["log volumes"].repeat(num_components))
        elif self.constraint[0] == 'V':
            volumes = np.exp(params["log volumes"])

        if self.constraint[1] == 'I':
            shapes = np.zeros((num_components,D,D)) + np.eye(D)
        elif self.constraint[1] == 'E':
            shapes = np.zeros((num_components,D,D)) + np.diag(np.exp(params["log shapes"]-np.sum(params["log shapes"])/D))
        elif self.constraint[1] == 'V':
            shapes = np.array([np.diag(row) for row in np.exp(params["log shapes"]-np.sum(params["log shapes"])/D)])

        if self.constraint[2] == 'I':
            orientations = np.zeros((num_components,D,D)) + np.eye(D)
        elif self.constraint[2] == 'E':
            skew_symm = (params["orientations"]-np.transpose(params["orientations"]))/2
            orientations = np.zeros((num_components,D,D)) + ((np.eye(D) + skew_symm) @ np.linalg.inv(np.eye(D) - skew_symm))
        elif self.constraint[2] == 'V':
            skew_symm = (params["orientations"]-np.transpose(params["orientations"],(0,2,1)))/2
            orientations = (np.eye(D) + skew_symm) @ np.linalg.inv(np.eye(D) - skew_symm)

        return normalized_log_proportions, params["means"], shapes, orientations, volumes

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
        for log_proportion, mean, shape, orientation, volume in zip(*self.unpack_params(params)):
            check_dim(log_proportion,()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean,(D,)) and check_finite(mean)
            check_dim(shape,(D,D)) and check_diagonal(shape) and check_pos(np.diag(shape),nonneg)
            check_dim(orientation,(D,D)) and check_orthogonal(orientation)
            check_dim(volume,()) and check_pos(volume,nonneg)
        check_probdist(np.array(proportions))

    def likelihood(self, params):
        """Calculates the log-likelihood for the Mclust model.
        
        Notes
        -----
        This method relies on the subroutine `unpack_params`
        to interpret the input dictionary of named parameters
        and assemble them into a 
        The input `params` is to be a dictionary of named parameters,
        of the same format as the return value of `Mclust.init_params`,
        and compatible with the instance value of the `constraint` attribute.
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

        See Also
        --------
        Mclust.unpack_params : interprets the 
        Mclust.init_params
        """
        cluster_lls = []
        for log_proportion, mean, shape, orientation, volume in zip(*self.unpack_params(params)):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(self.data, mean, cov))

        return np.sum(logsumexp(np.vstack(cluster_lls),axis=0))

    def objective(self, params):
        """Calculates the negative log-likelihood for the Mclust model."""
        return -self.likelihood(params)


    def aic(self, params):
        """Calculates the model AIC (Akaike Information Criterion)."""
        return 2 * self.num_freeparam + 2 * self.objective(params)

    def bic(self, params):
        """Calculates the model BIC (Bayesian Information Criterion)."""
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.objective(params)



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
        for log_proportion, mean, shape, orientation, volume in zip(*self.unpack_params(params)):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(self.data, mean, cov))

        return np.argmax(np.array(cluster_lls).T,axis=1)


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
            self.params_checker(params,nonneg=False)
            likelihood = self.likelihood(params)
            if not np.isfinite(likelihood):
                raise ValueError("Log likelihood is {}".format(likelihood))
            print("Log likelihood {}".format(likelihood))
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store  
