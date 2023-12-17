from Mixture_Models import *
from tests import utils


constraint_specific_inits = {
    "UUU": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [
            [[-0.08730011], [0.21651309]],
            [[0.60151869], [-0.48253284]],
            [[0.51413704], [0.11431507]],
        ],
        "error": [
            [0.22256881, -0.56830111],
            [0.06756844, 0.7422685],
            [-0.53990244, -0.98886414],
        ],
    },
    "UUC": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [
            [[-0.08730011], [0.21651309]],
            [[0.60151869], [-0.48253284]],
            [[0.51413704], [0.11431507]],
        ],
        "error": [0.22256881, -0.56830111, 0.06756844],
    },
    "UCU": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [
            [[-0.08730011], [0.21651309]],
            [[0.60151869], [-0.48253284]],
            [[0.51413704], [0.11431507]],
        ],
        "error": [0.22256881, -0.56830111],
    },
    "UCC": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [
            [[-0.08730011], [0.21651309]],
            [[0.60151869], [-0.48253284]],
            [[0.51413704], [0.11431507]],
        ],
        "error": [0.22256881],
    },
    "CUU": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [[-0.08730011], [0.21651309]],
        "error": [
            [0.60151869, -0.48253284],
            [0.51413704, 0.11431507],
            [0.22256881, -0.56830111],
        ],
    },
    "CUC": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [[-0.08730011], [0.21651309]],
        "error": [0.60151869, -0.48253284, 0.51413704],
    },
    "CCU": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [[-0.08730011], [0.21651309]],
        "error": [0.60151869, -0.48253284],
    },
    "CCC": {
        "log proportions": [0.66579325, 0.35763949, -0.77270015],
        "means": [
            [-0.00419192, 0.31066799],
            [-0.36004278, 0.13275579],
            [0.05427426, 0.00214572],
        ],
        "fac_loadings": [[-0.08730011], [0.21651309]],
        "error": [0.60151869],
    },
}


def run_constraint(constraint):
    test_PGMM, init_params = utils.init_MM(
        PGMM,
        default_pinwheel_data,
        10,
        init_params_args={"num_components": 3, "q": 1, "scale": 0.5},
        expected=constraint_specific_inits[constraint],
        constraint=constraint,
    )  # q is the latent subspace
    utils.check_fromfile(
        test_PGMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0009, "mass": 0.95, "maxiter": 100},
        "expected_test_pgmm_" + constraint + "_illustration.json",
        ["likelihood", "aic", "bic"],
    )


def test_pgmm_illustration():
    for constraint in {"CCC", "CCU", "CUC", "CUU", "UCC", "UCU", "UUC", "UUU"}:
        run_constraint(constraint)
