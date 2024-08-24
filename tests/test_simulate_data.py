from mixture_models import *
from tests import utils

from sklearn.mixture import GaussianMixture
import time


def run_experiment(
    method,
    n,
    p,
    K,
    prop_informative,
    balance,
    constrained,
    fit_constrained,
    num_trials=10,
    scale=5,
    use_kmeans=True,
    optim_params={"learning_rate": 0.0003, "mass": 0.9, "tol": 1e-06},
    export_results=False,
):
    if method not in ["AD", "EM"]:
        raise ValueError("method should be one of 'AD' or 'EM'")
    overall_start_time = time.time()
    num_iters = []
    timetaken = []
    ari = []
    avglogprop = []
    avgmean = []
    avgcov = []
    for trial in range(1, num_trials + 1):
        npr.seed(10 * trial)
        data, ground_truth, centers, covariances = simulate_data(
            n, p, K, constrained, balance, prop_informative, scale
        )

        start_time = time.time()
        try:
            if method == "AD":
                test_GMM = Mclust(data, "VVI") if fit_constrained else GMM(data)
                init_params = test_GMM.init_params(K, scale, use_kmeans)
                params = test_GMM.fit(init_params, "grad_descent", **optim_params)
            else:
                test_GMM = GaussianMixture(
                    n_components=K,
                    covariance_type="diag" if fit_constrained else "full",
                    tol=optim_params["tol"],
                ).fit(data)
        except:
            continue
        timetaken.append(time.time() - start_time)

        if method == "AD":
            num_iters.append(len(params))
            params = params[-1]
            labels = np.array(test_GMM.labels(data, params))
            if fit_constrained:
                fitlogprops, fitmeans, fitcovs = [], [], []
                for logprop, mean, shape, orientation, volume in zip(
                    *test_GMM.unpack_params(params)
                ):
                    fitlogprops.append(logprop)
                    fitmeans.append(mean)
                    fitcovs.append(volume * orientation @ shape @ orientation.T)
            else:
                fitlogprops, fitmeans, fitcovs = test_GMM.unpack_params(params)
                fitcovs = [np.dot(m, m.transpose()) for m in fitcovs]
        else:
            num_iters.append(test_GMM.n_iter_)
            labels = test_GMM.predict(data)
            fitlogprops = np.log(test_GMM.weights_)
            fitmeans = np.log(test_GMM.means_)
            fitcovs = np.log(test_GMM.covariances_)
            if fit_constrained:
                fitcovs = np.array([np.diag(x) for x in fitcovs])

        # We must first permute the labels, to avoid penalizing wrong permutations
        clusassg = np.argsort([sum(x * x for x in v) for v in fitmeans])
        labels = [clusassg[i] for i in labels]
        fitlogprops = [fitlogprops[i] for i in clusassg]
        fitmeans = [fitmeans[i] for i in clusassg]
        fitcovs = [fitcovs[i] for i in clusassg]

        # Average adjusted Rand index
        ari.append(adjusted_rand_score(labels, np.repeat(range(K), ground_truth)))
        # Average difference in log-proportions
        avglogprop.append(
            np.linalg.norm(fitlogprops - np.log(ground_truth) + np.log(n)) / K
        )
        # Average difference in means, across clusters
        avgmean.append(np.linalg.norm(fitmeans - centers) / K)
        # Average difference in covariance matrices (Frobenius norm), across clusters
        avgcov.append(np.linalg.norm(fitcovs - covariances) / K)
    overall_end_time = time.time()

    if export_results:
        # create a unique filepath
        filepath = (
            "_".join(
                [
                    str(x)
                    for x in [
                        method,
                        n,
                        p,
                        K,
                        prop_informative,
                        balance,
                        constrained,
                        fit_constrained,
                    ]
                ]
            )
            + ".json"
        )
        utils.create_run(
            {
                "method": method,
                "n": n,
                "p": p,
                "K": K,
                "prop_informative": prop_informative,
                "balance": balance,
                "constrained": constrained,
                "fit_constrained": fit_constrained,
                "scale": scale,
                "use_kmeans": use_kmeans,
                "optim_params": optim_params,
                "timetaken": timetaken,
                "num_iters": num_iters,
                "ari": ari,
                "avglogprop": avglogprop,
                "avgmean": avgmean,
                "avgcov": avgcov,
                "start_time": overall_start_time,
                "end_time": overall_end_time,
            },
            os.path.join("tests", filepath),
        )

    return timetaken, ari, avglogprop, avgmean, avgcov


def test_experiment():
    for method in ["AD", "EM"]:
        run_experiment(method, 128, 2, 2, 0.2, 4, False, False, export_results=True)
    assert True
