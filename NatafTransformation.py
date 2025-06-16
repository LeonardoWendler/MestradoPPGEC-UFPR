# -*- coding: utf-8 -*-
"""
Created on Sun Aug 25 00:15:59 2024

@author: User
"""

import numpy as np
from scipy.stats import norm, lognorm, uniform

def transform_y_to_x(L, y, means, stdvs, distributions, getJac):

    # Number of random variables
    numRV = len(y)

    # Transform from y to z
    z = L.dot(y)

    # Transform from z to x
    x = np.zeros(numRV)
    for j in range(numRV):

        if distributions[j] == "Normal":

            x[j] = z[j] * stdvs[j] + means[j]

        elif distributions[j] == "Lognormal":

            mu = np.log(means[j]) - 0.5 * np.log(1 + (stdvs[j] / means[j]) * (stdvs[j] / means[j]))
            sigma = np.sqrt(np.log((stdvs[j] / means[j]) * (stdvs[j] / means[j]) + 1))
            x[j] = np.exp(z[j] * sigma + mu)

        elif distributions[j] == "Uniform":

            halfspan = np.sqrt(3) * stdvs[j]
            a = means[j] - halfspan
            x[j] = uniform.ppf(norm.cdf(z[j]), a, 2 * halfspan)

        else:
            print("Error: Distribution type not found!")


    # Calculate Jacobian dxdy
    dxdy = np.zeros((numRV, numRV))
    if getJac:

        # First calculate diag[dxdz]
        dxdz = np.zeros((numRV, numRV))

        for j in range(numRV):

            if distributions[j] == "Normal":

                dxdz[j, j] = stdvs[j]

            elif distributions[j] == "Lognormal":

                mu = np.log(means[j]) - 0.5 * np.log(1 + (stdvs[j] / means[j]) * (stdvs[j] / means[j]))
                sigma = np.sqrt(np.log((stdvs[j] / means[j]) * (stdvs[j] / means[j]) + 1))
                dxdz[j, j] = sigma * np.exp(z[j] * sigma + mu)

            elif distributions[j] == "Uniform":

                halfspan = np.sqrt(3) * stdvs[j]
                a = means[j] - halfspan
                f = uniform.pdf(x[j], a, 2 * halfspan)
                phi = norm.pdf(z[j])
                dxdz[j, j] = phi / f

            else:
                print("Error: Distribution type not found!")

        # Notice that dG/dy = dg/dx * dx/dz * dz/dy can be multiplied in opposite order if transposed
        dxdy = (np.transpose(L)).dot(dxdz)

        return [x, dxdy]

    else:

        return x