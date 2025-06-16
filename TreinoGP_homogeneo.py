# -*- coding: utf-8 -*-
"""
Created on Wed Apr  2 21:46:30 2025

@author: Leona
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.gaussian_process import GaussianProcessRegressor as GPR
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import train_test_split
import sklearn.metrics as skmetric
import pandas as pd
import scipy as sc
from scipy.stats import qmc
from ANSYS_functions import evaluate_uniform
import time
import joblib

"Dados para' treino"
sampler = qmc.LatinHypercube(1,seed=199)
sample_init = sampler.random(n=int(40))
sample = qmc.scale(sample_init, 5., 60.)
results = np.zeros((sample.shape[0]))
timecount = 0
for i in range(0,sample.shape[0]):
    ini=time.time()
    results[i] = evaluate_uniform(sample[i][0])
    timecount+=time.time()-ini
    print("Status: ",(i+1),"/", sample.shape[0], "Elapsed time: ",timecount," s", "\n Remaining time: ", (timecount/(i+1))*(sample.shape[0]-i-1)," s")

"Setup do metamodelo"
# Seed dos dados train homogeneo 9980
data = pd.read_excel("KriggingTrain_Homogeneous.xlsx",header=0)
X = data.iloc[:,0:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()
datatest = pd.read_excel("KriggingTest_Homogeneous.xlsx",header=0)
X_test = datatest.iloc[:,0:-1].to_numpy()
y_test = datatest.iloc[:,-1].to_numpy()
for i in range(0,sample.shape[0]):
    ini=time.time()
    results[i] = evaluate_uniform(X[i][0])
    timecount+=time.time()-ini
    print("Status: ",(i+1),"/", X.shape[0], "Elapsed time: ",timecount," s", "\n Remaining time: ", (timecount/(i+1))*(X.shape[0]-i-1)," s")


# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=12)

X_train = X
y_train = y
# Definir kernel (RBF com variância e length-scale)
kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e4))

# Objective function

def obj_func(theta, eval_gradient=True):
    # theta: current kernel hyperparameters
    # Computes: -log marginal likelihood under current kernel
    lml, grad = gp.log_marginal_likelihood(
        theta, eval_gradient=True, clone_kernel=False
    )
    return -lml, -grad

# Custom optimizer with increased max iterations
def custom_optimizer(obj_func, initial_theta, bounds):

    opt_res = sc.optimize.minimize(
        obj_func,
        initial_theta,
        bounds=bounds,
        method="L-BFGS-B",
        jac=True,
        options={'maxiter': 500}
        # options={'maxiter': 500, 'disp': True},  # Increase iterations and show logs
    )
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min

# Criar modelo
gp = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True, optimizer = custom_optimizer)
# Treinar
gp.fit(X_train, y_train)

# Prever com incerteza
y_predtr, y_std = gp.predict(X_train, return_std=True)
# joblib.dump(gp, "gpruniformmodel.pkl")
# Avaliação
y_predtest, y_stdtest = gp.predict(X_test, return_std=True) #Dados de teste

#Calculo metricas
# R2_train = skmetric.r2_score(y_train,y_predtr)
R2_test = skmetric.r2_score(y_test,y_predtest)
MSE_train = skmetric.root_mean_squared_error(y, y_predtr)
MSE_test = skmetric.root_mean_squared_error(y_test, y_predtest)
# MAE_train = skmetric.mean_absolute_error(y_train,y_predtr)
MAE_test = skmetric.mean_absolute_error(y_test,y_predtest)
# MAXE_train = skmetric.max_error(y_train,y_predtr)
MAXE_test = skmetric.max_error(y_test,y_predtest)

# Visualização (apenas previsão vs real)
plt.figure(figsize=(6, 4))
plt.errorbar(range(len(y_pred)), y_pred, yerr=y_std, fmt='o', label='Previsão ± std')
plt.plot(range(len(y_test)), y_test, 'x', label='Valor real')
plt.legend()
plt.title("Previsão com Gaussian Process")
plt.tight_layout()
plt.grid(True)
plt.show()

y_predtr, y_stdtr = gp.predict(X_train, return_std=True)

R2 = skmetric.r2_score(y_train,y_predtr)
print(skmetric.root_mean_squared_error(y_pred, y_test))
print(skmetric.mean_absolute_error(y_pred, y_test))
print(skmetric.max_error(y_pred, y_test))