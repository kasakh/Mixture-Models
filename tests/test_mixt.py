from mixture_models import *
from tests import utils


def init_mixt():
    return utils.init_MM(
        TMM,
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
            "sqrt_covs": [np.eye(2), np.eye(2), np.eye(2)],
            "log_dofs": [-0.17460021, 0.43302619, 1.20303737],
        },
    )


def test_mixt_illustration():
    test_TMM, init_params = init_mixt()
    utils.check_fromfile(
        test_TMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0005, "mass": 0.9, "maxiter": 100},
        "expected_test_mixt_illustration.json",
        ["likelihood", "aic", "bic"],
    )
