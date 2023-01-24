from .mixture_models import *
import autograd.numpy as np

class Mclust(MM):
    def __init__(self, data, constraint="VVV"):
        self.constraint_set = {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'}
        if constraint in self.constraint_set:
            self.constraint = constraint
        else:
            raise ValueError(
                "constraint should be one of {'EII','VII','EEI','VEI','EVI','VVI','EEE','VEE','EVE','VVE','EEV','VEV','EVV','VVV'} "
            )

        self.data_checker(data)


    def init_params(self, num_components, scale=1.0):
        '''
        Mclust parametrization reference: https://sites.stat.washington.edu/raftery/Research/PDF/fraley2003.pdf#page=3
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
            shapes = np.zeros((num_components,D,D)) + np.diag(np.exp(params["log shapes"]-np.sum(params["shapes"])/D))
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



    def likelihood(self, params):
        cluster_lls = []
        for log_proportion, mean, shape, orientation, volume in zip(*self.unpack_params(params)):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(self.data, mean, cov))

        return np.sum(logsumexp(np.vstack(cluster_lls),axis=0))

    def objective(self, params):
        return -self.likelihood(params)


    def aic(self, params):
        return 2 * self.num_freeparam + 2 * self.objective(params)

    def bic(self, params):
        return np.log(
            self.num_datapoints
        ) * self.num_freeparam + 2 * self.objective(params)



    def labels(self, data, params):
        cluster_lls = []
        for log_proportion, mean, shape, orientation, volume in zip(*self.unpack_params(params)):
            cov = volume * orientation @ shape @ orientation.T
            cluster_lls.append(log_proportion + mvn.logpdf(self.data, mean, cov))

        return np.argmax(np.array(cluster_lls).T,axis=1)


    def fit(self, init_params, opt_routine, draw=None, **kargs):

        self.params_store = []
        flattened_obj, unflatten, flattened_init_params = flatten_func(
            self.objective, init_params
        )

        def callback(flattened_params):
            params = unflatten(flattened_params)
            print("Log likelihood {}".format(self.likelihood(params)))

            #
            self.params_store.append(params)

        self.optimize(
            flattened_obj, flattened_init_params, callback, opt_routine, **kargs
        )
        return self.params_store  
