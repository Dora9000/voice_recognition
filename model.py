import numpy as np


def map_adaptation(gmm, data, max_iterations=300, likelihood_threshold=1e-20, relevance_factor=16):
    N = data.shape[0]
    D = data.shape[1]
    K = gmm.n_components

    mu_new = np.zeros((K, D))
    n_k = np.zeros((K, 1))

    mu_k = gmm.means_
    cov_k = gmm.covariances_
    pi_k = gmm.weights_

    old_likelihood = gmm.score(data)
    new_likelihood = 0
    iterations = 0
    while abs(old_likelihood - new_likelihood) > likelihood_threshold and iterations < max_iterations:
        iterations += 1
        old_likelihood = new_likelihood
        z_n_k = gmm.predict_proba(data)
        n_k = np.sum(z_n_k, axis=0)
        for i in range(K):
            temp = np.zeros((1, D))
            for n in range(N):
                temp += z_n_k[n][i] * data[n, :]
            mu_new[i] = (1 / max(n_k[i], 1e-20)) * temp

        adaptation_coefficient = n_k / (n_k + relevance_factor)
        for k in range(K):
            mu_k[k] = (adaptation_coefficient[k] * mu_new[k]) + ((1 - adaptation_coefficient[k]) * mu_k[k])
        gmm.means_ = mu_k

        log_likelihood = gmm.score(data)
        new_likelihood = log_likelihood
        #print(log_likelihood)
    return gmm
