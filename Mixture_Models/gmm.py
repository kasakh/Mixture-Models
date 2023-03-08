from .mixture_models import *
from .checkers import *
import autograd.numpy as np
import autograd.scipy.stats.multivariate_normal as mvn

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
    labels(data, params_store)
        Returns 'hard' cluster assignments for given data, based on fitted parameters.

    """

    def objective(self, params):
        kl_cov = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            kl_cov.append(cov_sqrt.T @ cov_sqrt)
        return (
            -1 * self.gmm_log_likelihood(params, self.data)
            - (0 * self.kl_div_tot(params["means"], kl_cov))
            - (-0.0 * self.kl_div_inverse_tot(params["means"], kl_cov))
        )
    
    def alt_gmm_log_likelihood(self, params, data):
        cluster_lls = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            cov = cov_sqrt.T @ cov_sqrt
            cluster_lls.append(log_proportion + mvn.logpdf(data, mean, cov))
        return np.sum(logsumexp(np.vstack(cluster_lls), axis=0))

    def alt_objective(self, params):
        return -self.alt_gmm_log_likelihood(params, self.data)

    def likelihood(self, params):
        return -self.alt_objective(params)

    def init_params(self, num_components, scale=1.0):
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
              Real matrixof shape (num_components, D, D)
            
            where D = number of data dimensions.
        
        See Also
        --------
        GMM.unpack_params : interface between the return value and other methods
        GMM.alt_objective : loss function where the unpacked values are used 
        """
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
    
    def params_checker(self, params, nonneg=True):
        D = self.num_dim
        proportions = []
        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params)):
            check_dim(log_proportion,()) and check_pos(-log_proportion)
            proportions.append(np.exp(log_proportion))
            check_dim(mean,(D,)) and check_finite(mean)
            check_dim(cov_sqrt,(D,D)) and check_finite(cov_sqrt)
            if not nonneg:
                check_posdef(cov_sqrt.T @ cov_sqrt)
        check_probdist(np.array(proportions))


    def aic(self, params):
        return 2 * self.num_freeparam + 2 * self.alt_objective(params)

    def bic(self, params):
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.alt_objective(params)

    def labels(self, data, params_store):
        """Assigns clusters to data, based on given parameters.

        This cluster assignment is "hard", in the sense that it returns one cluster for each data point,
        unlike "soft" classifiers that return a probability distribution over clusters for each data point.

        Parameters
        ----------
        data : (..., N, D) ndarray
            D-dimensional input of (..., N) datapoints to be labelled.
        params_store : dict
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

        for log_proportion, mean, cov_sqrt in zip(*self.unpack_params(params_store)):

            cluster_lls.append(log_proportion + self.mvn_logpdf(data, mean, cov_sqrt))

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
