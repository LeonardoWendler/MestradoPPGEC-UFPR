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
from DiscRandomField import sample_RandomField
import time
from ANSYS_functions import evaluate_LHS
import joblib

"Dados para campo aleatorio"
incid = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="Incidencia").to_numpy()
x_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndcoord").to_numpy()[:,0]
y_points = pd.read_excel('elem_data_fileatt.xlsx',header=0,sheet_name="RCIndcoord").to_numpy()[:,1]

"Dados para' treino"
rng = np.random.default_rng(879104)
sample = (60.05 - 5 )*rng.random((400, 24))
results = np.zeros((sample.shape[0]))
timecount = 0
for i in range(0,sample.shape[0]):
    ini=time.time()
    results[i] = evaluate_LHS(sample[i][:])
    timecount+=time.time()-ini
    print("Status: ",(i+1),"/", sample.shape[0], "Elapsed time: ",timecount," s", "\n Remaining time: ", (timecount/(i+1))*(sample.shape[0]-i-1)," s")

"Setup do metamodelo"    
data = pd.read_excel("KriggingTrain_Heterogeneous.xlsx",header=None)
datatest = pd.read_excel("KriggingTest_Heterogeneous.xlsx",header=None,sheet_name='Teste1')
datatest2 = pd.read_excel("KriggingTest_Heterogeneous.xlsx",header=None,sheet_name='seed=879104')

X = data.iloc[:,0:-1].to_numpy()
y = data.iloc[:,-1].to_numpy()
X_test = datatest.iloc[:,0:-1].to_numpy()
y_test = datatest.iloc[:,-1].to_numpy()
X_test2 = datatest2.iloc[:,0:-1].to_numpy()
y_test2 = datatest2.iloc[:,-1].to_numpy()

# Dividir em treino e teste
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=122)

# Definir kernel (RBF com variância e length-scale)
# thetas = np.array([1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,1.,
#                    1.,1.,1.,1.,1.,1.,1.,1.,1.,1.])
thetas = np.array([1.])

kernel = C(1.0, (1e-3, 1e5)) * RBF(length_scale=thetas, length_scale_bounds=(1e-2, 1e4))

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
        # options={'maxiter': 100}
        options={'maxiter': 100, 'disp': True},  # Increase iterations and show logs
    )
    theta_opt, func_min = opt_res.x, opt_res.fun
    return theta_opt, func_min

# Criar modelo
gp = GPR(kernel=kernel, n_restarts_optimizer=10, alpha=1e-6, normalize_y=True, optimizer = custom_optimizer)

# Treinar
gp.fit(X, y)

# Prever com incerteza
y_predtest, y_std = gp.predict(X_test, return_std=True) #Dados de teste
y_predtest2, y_std2 = gp.predict(X_test2, return_std=True) #Dados de teste
y_predtr, y_stdtr = gp.predict(X, return_std=True) #Dados de treino

#Calculo metricas
# MAE_train = skmetric.mean_absolute_error(y_train,y_predtr)
MAE_test = skmetric.mean_absolute_error(y_test,y_predtest)
MAE_test2 = skmetric.mean_absolute_error(y_test2,y_predtest2)
# MSE_train = skmetric.root_mean_squared_error(y, y_predtr)
MSE_test = skmetric.root_mean_squared_error(y_test, y_predtest)
MSE_test2 = skmetric.root_mean_squared_error(y_test2, y_predtest2)
# MAXE_train = skmetric.max_error(y_train,y_predtr)
MAXE_test = skmetric.max_error(y_test,y_predtest)
MAXE_test2 = skmetric.max_error(y_test2,y_predtest2)
# R2_train = skmetric.r2_score(y_train,y_predtr)
R2_test = skmetric.r2_score(y_test,y_predtest)
R2_test2 = skmetric.r2_score(y_test2,y_predtest2)

#Print avaliacoes
print("Kernel otimizado:", gp.kernel_)
# print("MSE treino:", MSE_train)
print("MSE teste:", MSE_test)
# print("MAE treino:", MAE_train)
print("MAE teste:", MAE_test)
# print("MAX Absolute Error treino:", MAXE_train)
print("MAX Absolute Error teste:", MAXE_test)
# print("R² Treino:", R2_train)
print("R² Teste:", R2_test)

joblib.dump(gp, "gprmodelv2.pkl")

# Visualização (apenas previsão vs real)
# plt.figure(figsize = (16*1/2.54,9*1/2.54), dpi=300)
# plt.errorbar(range(len(y_predtest)), y_predtest, yerr=y_std, fmt='o', label='Previsão ± std')
# plt.plot(range(len(y_test)), y_test, 'x', label='Valor real')
# plt.legend()
# plt.title("Previsão com Gaussian Process")
# plt.tight_layout()
# plt.show()

# Visualizacao dados treino vs previsoes modelo
# fig, ax = plt.subplots(figsize = (16*1/2.54,9*1/2.54), dpi=300)
# ax.scatter(y_predtr*0.25, y_train*0.25 ,marker=".", label = "Previsões no conjunto de dados de treino")
# ax.annotate(f'R² = {R2_train:.2f}', xy=(0.05, 0.8), xycoords='axes fraction',
#             bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.2))
# ax.set_xlabel(r'Resistência a Flambagem - MEF (kN/m)')
# ax.set_ylabel(r'Previsões do metamodelo (kN/m)')
# plt.legend()
# plt.show()
# Visualizacao dados teste vs previsoes modelo
# fig, ax = plt.subplots(figsize = (16*1/2.54,9*1/2.54), dpi=300)
# ax.scatter(y_predtest*0.25, y_test*0.25 ,marker=".", label = "Previsões no conjunto de dados de treino")
# ax.annotate(f'R² = {R2_test:.2f}', xy=(0.05, 0.8), xycoords='axes fraction',
#             bbox=dict(boxstyle='square,pad=0.3', facecolor='white', alpha=0.2))
# ax.set_xlabel(r'Resistência a Flambagem - MEF (kN/m)')
# ax.set_ylabel(r'Previsões do metamodelo (kN/m)')
# plt.legend()
# plt.show()