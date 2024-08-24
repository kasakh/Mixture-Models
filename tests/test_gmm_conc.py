from mixture_models import *
from tests import utils


def init_GMM_Constrainted():
    return utils.init_MM(
        GMM_Constrainted,
        default_pinwheel_data,
        10,
        init_params_args={"num_components": 3, "scale": 0.5},
        expected={
            "log proportions": [0.66579325, 0.35763949, -0.77270015],
            "means": [
                [-0.00419192, 0.31066799],
                [-0.36004278, 0.13275579],
                [0.05427426, 0.00214572],
            ],
            "sqrt_covs": np.eye(2),
        },
    )


def test_gmm_conc_illustration():
    test_GMM, init_params = init_GMM_Constrainted()
    utils.check_fromfile(
        test_GMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0009, "mass": 0.8, "maxiter": 100},
        "expected_test_gmm_conc_illustration.json",
        ["likelihood", "aic", "bic"],
    )
