from mixture_models import *
from tests import utils


def init_MFA():
    return utils.init_MM(
        MFA,
        default_pinwheel_data,
        10,
        init_params_args={
            "num_components": 3,
            "q": 1,
            "scale": 0.5,
        },  # q is the latent subspace
        expected={
            "log proportions": [0.66579325, 0.35763949, -0.77270015],
            "means": [
                [-0.00419192, 0.31066799],
                [-0.36004278, 0.13275579],
                [0.05427426, 0.00214572],
            ],
            "fac_loadings": [
                [[0.40631048], [0.30626303]],
                [[0.36087766], [0.14593803]],
                [[0.45888706], [0.35728789]],
            ],
            "error": [
                [-0.08730011, -0.56830111],
                [0.06756844, 0.7422685],
                [-0.53990244, -0.98886414],
            ],
        },
    )


def test_mfa_illustration():
    test_MFA, init_params = init_MFA()
    utils.check_fromfile(
        test_MFA,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0009, "mass": 0.95, "maxiter": 100},
        "expected_test_mfa_illustration.json",
        ["likelihood", "aic", "bic"],
    )
