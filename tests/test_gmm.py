from Mixture_Models import *


def make_data():
    return make_pinwheel(0.3, 0.05, 3, 100, 0.4)


def test_gmm_minimal_workflow():
    GMM_model = GMM(make_data())
    init_params = GMM_model.init_params(3,scale=0.5)
    #params_store = GMM_model.fit(init_params,"Newton-CG")
    for i, k in enumerate(init_params):
        assert k == ("log proportions", "means", "sqrt_covs")[i]
        assert np.shape(init_params[k]) == ((3,), (3, 2), (3, 2, 2))[i]
    # print(np.array([7.89039631e+00,9.66895221e+00,1.78736830e+08]))
    # assert np.array_equal(params_store['log_proportions'],np.array([7.89039631e+00,9.66895221e+00,1.78736830e+08]))
    # assert np.array_equal(params_store['means'], np.array([[-10.42522607, -8.24703923],[15.79200649,-1.1549147],[0.83587269,0.82873191]]))
    # assert np.array_equal(params_store['sqrt_precs'], np.array([[[7.54631258e+00,1.74371255e+00],[-3.12614448e+00,-7.13538201e+00]],[[1.24718316e+00,2.17712190e+00],[4.02759027e+00,-2.54640116e+00]],[[-1.45377366e+04,-3.84002320e+04],[1.97569616e+03,4.82101069e+03]]]))