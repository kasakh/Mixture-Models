from mixture_models import *
from tests import utils


def init_GMM():
    return utils.init_MM(
        GMM,
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
        },
    )


def check_mpkls(MM, params_store, expected):
    actual = []
    for params in params_store:
        kl_cov = []
        for log_proportion, mean, cov_sqrt in zip(*MM.unpack_params(params)):
            kl_cov.append(cov_sqrt.T @ cov_sqrt)
        actual.append(MM.print_mpkl(params["means"], kl_cov))
    utils.check_array_equal(actual, expected)


def test_gmm_illustration():
    test_GMM, init_params = init_GMM()
    params_store = utils.check_fromfile(
        test_GMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0005, "mass": 0.9, "maxiter": 100},
        "expected_test_gmm_illustration.json",
        ["likelihood", "aic", "bic"],
    )
    check_mpkls(
        test_GMM,
        params_store,
        [
            0.0,
            0.00022942331022557383,
            0.00065849782513272,
            0.0012577493548364682,
            0.0020036395698388443,
            0.0028844523252771204,
            0.0039061014528369853,
            0.005097590838421429,
            0.006515743453741507,
            0.008248575420026327,
            0.010416353131524891,
            0.01316897852701815,
            0.016677987847406817,
            0.021121290820669048,
            0.0266590941956788,
            0.03340061148303963,
            0.041363508652638714,
            0.05043164842887338,
            0.06032098016413756,
            0.07056677564368186,
            0.080545360444084,
            0.08953774585901231,
            0.09683086532420848,
            0.10183770273244175,
            0.1042066843501297,
            0.10388946185166037,
            0.1011469316171445,
            0.09775986979019824,
            0.10683811388353837,
            0.11646961082497898,
            0.12688791601266947,
            0.13842261089117236,
            0.15151692296449326,
            0.16675607966796435,
            0.1849138400152377,
            0.20702776338409157,
            0.23451954482465487,
            0.26938764408102944,
            0.31452018046912844,
            0.3742164060334241,
            0.4550864668146155,
            0.5676717711786832,
            0.7580533931086171,
            1.0593118478515477,
            1.5389213713192804,
            2.3546241004604553,
            3.862328746847337,
            6.9797992744094,
            14.58393076698857,
            39.35560551460964,
            171.60763084077564,
            147.69117154936654,
            61.03323522275949,
            30.014177456386264,
            20.198284925832866,
            19.004739839442173,
            24.90970970006203,
            40.43855534852633,
            69.5134927846708,
            120.16119618027558,
            162.6053572080931,
            112.25388122898579,
            82.07238864431855,
            82.33359327518978,
            88.37558305583838,
            84.21832257922327,
            89.37520756177814,
            124.94708962826738,
            231.04796508133546,
            262.44030761788747,
            132.8262247227446,
            85.8314176353328,
            74.97583934088611,
            87.88227977533904,
            139.1328580370184,
            224.70344871685367,
            107.85453121954919,
            72.46716309209836,
            71.24421295459022,
            97.3129266340971,
            159.77086095244124,
            153.96578077313393,
            171.432292215588,
            252.59044154292448,
            311.47740669448854,
            174.1024581490238,
            131.24202566137438,
            140.36749789277806,
            196.20950742632883,
            241.09418957839358,
            185.7942134371872,
            160.43576224987714,
            171.05835769159216,
            178.43981285864908,
            147.82599578399723,
            134.6625845158233,
            147.34856808400986,
            177.8239087113556,
            187.67862021907823,
            171.21401995601397,
        ],
    )


def test_gmm_different_optimizers_illustration():
    test_GMM, init_params = init_GMM()
    utils.check_fromfile(
        test_GMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0005, "mass": 0.9, "maxiter": 100, "tol": 1e-7},
        "expected_test_gmm_different_optimizers_illustration_0.json",
        ["likelihood"],
    )
    utils.check_fromfile(
        test_GMM,
        init_params,
        "rms_prop",
        {"learning_rate": 0.01, "gamma": 0.9, "maxiter": 100, "tol": 1e-7},
        "expected_test_gmm_different_optimizers_illustration_1.json",
        ["likelihood"],
    )
    utils.check_fromfile(
        test_GMM,
        init_params,
        "adam",
        {
            "learning_rate": 0.1,
            "beta1": 0.9,
            "beta2": 0.99,
            "maxiter": 100,
            "tol": 1e-7,
        },
        "expected_test_gmm_different_optimizers_illustration_2.json",
        ["likelihood"],
    )
    utils.check_fromfile(
        test_GMM,
        init_params,
        "Newton-CG",
        {"maxiter": 100, "tol": 1e-5},
        "expected_test_gmm_different_optimizers_illustration_3.json",
        ["likelihood"],
    )


def test_gmm_different_initializations_illustration():
    test_GMM, init_params = init_GMM()

    init_params["log proportions"] = np.log([0.1, 0.2, 0.7])
    from scipy.linalg import sqrtm

    init_params["sqrt_covs"][0] = sqrtm([[4, -0.75], [-0.75, 1]])

    utils.check_fromfile(
        test_GMM,
        init_params,
        "grad_descent",
        {"learning_rate": 0.0005, "mass": 0.9, "maxiter": 100, "tol": 1e-7},
        "expected_test_gmm_different_initializations_illustration.json",
        ["likelihood"],
    )


def test_other_datasets():
    datasets = ["wine"]
    expected_init_params = [
        {
            "log proportions": [0.66579325, 0.35763949, -0.77270015],
            "means": [
                [
                    -0.00419192,
                    0.31066799,
                    -0.36004278,
                    0.13275579,
                    0.05427426,
                    0.00214572,
                    -0.08730011,
                    0.21651309,
                    0.60151869,
                    -0.48253284,
                    0.51413704,
                    0.11431507,
                    0.22256881,
                ],
                [
                    -0.56830111,
                    0.06756844,
                    0.7422685,
                    -0.53990244,
                    -0.98886414,
                    -0.87168615,
                    0.13303508,
                    1.19248367,
                    0.56184563,
                    0.83631111,
                    0.04957461,
                    0.69899819,
                    -0.13562399,
                ],
                [
                    0.30660209,
                    -0.13365859,
                    -0.27465451,
                    0.06635415,
                    -0.23807101,
                    0.65423654,
                    0.09750664,
                    0.20010499,
                    -0.16881617,
                    0.62823613,
                    -0.36598475,
                    0.33011578,
                    -0.17543595,
                ],
            ],
            "sqrt_covs": [np.eye(13), np.eye(13), np.eye(13)],
        }
    ]
    for i, dataset in enumerate(datasets):
        test_GMM, init_params = utils.init_MM(
            GMM,
            data.load_csvdataset(dataset)[0],
            10,
            init_params_args={"num_components": 3, "scale": 0.5},
            expected=expected_init_params[i],
        )
        utils.check_fromfile(
            test_GMM,
            init_params,
            "Newton-CG",
            {"maxiter": 25, "tol": 1e-5},
            "expected_test_other_datasets_" + str(i) + ".json",
            ["likelihood", "aic", "bic"],
        )
