"""Offline scientific references. No production or precision qualification implied."""
import numpy as np


def kalman(y, variance, observed, F, G, Q, C):
    """Scalar observation Joseph filter; x0 precedes the first transition.

    Missing values are never read by arithmetic. Returns per-time log density,
    filtered means/covariances and predictions, all in the input matrix dtype.
    """
    dtype = G.dtype
    n, d = F.shape
    m = np.zeros(d, dtype=dtype)
    P = C.copy()
    means, covs, predictions, predcovs = [], [], [], []
    terms = np.zeros(n, dtype=dtype)
    eye = np.eye(d, dtype=dtype)
    for t in range(n):
        m = G @ m
        P = G @ P @ G.T + Q
        predictions.append(m.copy()); predcovs.append(P.copy())
        if observed[t]:
            if not np.isfinite(y[t]) or not np.isfinite(variance[t]) or variance[t] < 0:
                raise ValueError('Observed values and nonnegative variances must be finite')
            v = y[t] - F[t] @ m
            s = F[t] @ P @ F[t] + variance[t]
            if not np.isfinite(s) or s <= 0:
                raise ValueError('Nonpositive/nonfinite innovation variance; no numerical floor allowed')
            k = P @ F[t] / s
            m = m + k * v
            A = eye - np.outer(k, F[t])
            P = A @ P @ A.T + np.outer(k, k) * variance[t]
            terms[t] = -dtype.type(.5) * (np.log(dtype.type(2*np.pi)*s) + v*v/s)
        means.append(m.copy()); covs.append(P.copy())
    return tuple(np.asarray(x) for x in (terms, means, covs, predictions, predcovs))


def dense_condition(y, variance, observed, F, G, Q, C):
    """Independent innovation-loading construction of the complete Gaussian.

    Includes all cross-time state covariance blocks. Intended for tiny fixtures.
    """
    n, d = F.shape
    loading = np.zeros((n*d, (n+1)*d), dtype=np.float64)
    noise = np.zeros(((n+1)*d, (n+1)*d), dtype=np.float64)
    noise[:d, :d] = C
    for t in range(n):
        noise[(t+1)*d:(t+2)*d, (t+1)*d:(t+2)*d] = Q
        for j in range(t+2):
            power = t+1 if j == 0 else t+1-j
            loading[t*d:(t+1)*d, j*d:(j+1)*d] = np.linalg.matrix_power(G, power)
    state_cov = loading @ noise @ loading.T
    H = np.zeros((n, n*d))
    for t in range(n):
        H[t, t*d:(t+1)*d] = F[t]
    H = H[observed]
    V = H @ state_cov @ H.T + np.diag(variance[observed])
    yy = y[observed]
    if not len(yy):
        return 0., np.zeros(n*d), state_cov
    sign, logdet = np.linalg.slogdet(V)
    if sign <= 0:
        raise ValueError('Degenerate observation distribution')
    cross = state_cov @ H.T
    mean = cross @ np.linalg.solve(V, yy)
    covariance = state_cov - cross @ np.linalg.solve(V, cross.T)
    lp = -.5*(len(yy)*np.log(2*np.pi) + logdet + yy @ np.linalg.solve(V, yy))
    return lp, mean, covariance


def rts(filtered, G):
    _, means, covs, predictions, predcovs = filtered
    means, covs = means.copy(), covs.copy()
    for t in range(len(means)-2, -1, -1):
        J = np.linalg.solve(predcovs[t+1], G @ filtered[2][t]).T
        means[t] += J @ (means[t+1] - predictions[t+1])
        covs[t] += J @ (covs[t+1] - predcovs[t+1]) @ J.T
    return means, covs


def simulate(F, G, Q, C, variance, *, seed):
    """Generate in fixed model units; never normalize by the realized range.

    Inputs specify generative units and covariance. PSD process noise allows
    exact deterministic components without jitter or variance floors.
    """
    rng=np.random.default_rng(seed)
    x=rng.multivariate_normal(np.zeros(len(G)),C,check_valid='raise')
    states=[];y=[]
    for t,f in enumerate(F):
        x=G@x+rng.multivariate_normal(np.zeros(len(G)),Q,check_valid='raise')
        states.append(x.copy());y.append(f@x+rng.normal(scale=np.sqrt(variance[t])))
    return np.asarray(y),np.asarray(states)


def simulation_smoother(filtered,G,*,seed,n_draws):
    """Offline backward conditional simulation with separate retained-draw RNG.

    No clipping/jitter. A failed covariance factorization remains a failure.
    """
    rng=np.random.default_rng(seed)
    _,m,P,a,R=filtered
    n,d=m.shape
    draws=np.empty((n_draws,n,d))
    draws[:,-1]=rng.multivariate_normal(m[-1],P[-1],size=n_draws,check_valid='raise')
    for t in range(n-2,-1,-1):
        J=np.linalg.solve(R[t+1],G@P[t]).T
        cond=P[t]-J@R[t+1]@J.T
        center=m[t]+(draws[:,t+1]-a[t+1])@J.T
        draws[:,t]=center+rng.multivariate_normal(np.zeros(d),cond,size=n_draws,check_valid='raise')
    return draws
