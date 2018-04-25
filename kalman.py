import numpy as np
import matplotlib.pyplot as plt


def kalman(data, var_estimate = 0.1 ** 2):
    n_iter = len(data)
    sz = (n_iter, )

    Q = 1e-5  # process variance

    # allocate space for arrays
    xhat = np.zeros(sz)  # a posteri estimate of x
    P = np.zeros(sz)  # a posteri error estimate
    xhatminus = np.zeros(sz)  # a priori estimate of x
    Pminus = np.zeros(sz)  # a priori error estimate
    K = np.zeros(sz)  # gain or blending factor

    R = var_estimate  # estimate of measurement variance, change to see effect

    xhat[0] = data[0]
    P[0] = 1.0

    for k in range(1, n_iter):
        # time update
        xhatminus[k] = xhat[k - 1]
        Pminus[k] = P[k - 1] + Q

        # measurement update
        K[k] = Pminus[k] / (Pminus[k] + R)
        xhat[k] = xhatminus[k] + K[k] * (data[k] - xhatminus[k])
        P[k] = (1 - K[k]) * Pminus[k]

    return xhat


if __name__ == '__main__':
    data = np.random.normal(-0.37727, 0.1, size=50)
    smoothed_data = kalman(data)

    plt.figure()
    plt.plot(data, 'k+', label='noisy measurements')
    plt.plot(smoothed_data, 'b-', label='a posteri estimate')
    plt.axhline(-0.37727, color='g', label='truth value')
    plt.legend()
    plt.title('Estimate vs. iteration step', fontweight='bold')
    plt.xlabel('Iteration')
    plt.ylabel('Voltage')
    plt.show()
