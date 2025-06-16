# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:36:55 2024

@author: User
"""
# import warnings
# warnings.filterwarnings('ignore')
import subprocess
import os
import re
from string import Template
import time
import numpy as np
import math as m
import pandas as pd
from DiscRandomField import sample_RandomField

def Ortoelastic_prop(M_m, p_m, p_f, E_m, E_f, G_m, G_f, v_m, v_f):
    
    '''
    M_m: resin content (percentual de 0 a 1)
    p_m: resin density (kg/m3)
    p_f: fiber density (kg/m3)
    E_m: matrix Young elastic modulus (GPa)
    E_f: fiber Young elastic modulus (GPa)
    G_m: matrix Shear elastic modulus (GPa)
    G_f: fiber Shear elastic modulus (GPa)
    v_m: matrix poisson ratio
    v_f: fiber poisson ratio
    '''
    V_f = (1-M_m)*p_m/(M_m*p_f+(1-M_m)*p_m)
    E_1 =  E_f*V_f+E_m*(1-V_f)
    v_12 = v_f*V_f+v_m*(1-V_f) #v_12 = v_13
    eta_2 = 0.2/(1-v_m)*(1.1 - m.sqrt(E_m/E_f) + 3.5*E_m/E_f)*(1 + 0.22*V_f)
    E_2 = E_f*E_m*(V_f+eta_2*(1-V_f))/(E_m*V_f+E_f*eta_2*(1-V_f))
    n_12 = 0.28 + m.sqrt(E_m/E_f) 
    G_12 = G_f*G_m*(V_f + n_12*(1-V_f))/(G_m*V_f + G_f*n_12*(1 - V_f)) #G_12 = G_13
    v_23 = v_12 #v_12 = v_13
    G_23 = E_2/(2*((1+v_23)))
#    v_21 = v_12*E_2/E_1
    # print("Valor E_1: {:.2f} GPa".format(E_1))
    # print("Valor E_2: {:.2f} GPa".format(E_2))
    # print("Valor G_12: {:.2f} GPa".format(G_12))
    # print("Valor G_23: {:.2f} GPa".format(G_23))
    return E_1, E_2, G_12, G_23, v_12
# TBELAST = np.zeros((RCvec.shape[0]))
# data_sorted = np.sort(RCvec)
# TBELAST = Ortoelastic_prop(data_sorted*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
# test = Ortoelastic_prop(27*0.01, 940, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
# test = Ortoelastic_prop(32.5106*0.01, 940, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
# test = Ortoelastic_prop(34.3271*0.01, 940, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)

'''
lambdas ---- 50, 75, 100, 125, 150, 200
'''
'''
INTERPOLACAO LINEAR DOS TEORES DE RESINA
'''

def linear_interp(SEC_ID, data, incid):
    
    '''
    SEC_ID: ID da secao desejada, ex: 5 para I 200x100x9.5, ...
    x0: coordenadas x do pontos conhecidos
    y0: coordenadas y do pontos desconhecidos
    z0: valores dos pontos conhecidos
    xi: coordenadas x do pontos interpolados
    yi: coordenadas y do pontos interpolados
    data: dados dos teores de resina
    incid: incidencia dos elementos [Num elem,SecID,CGx,CGy,CGz]
    '''

    def IWD(xi, yi):
        cross_sec = pd.DataFrame(np.array([[100., 70., 6.], [102., 51., 6.4], [120.,70.,8.],
                                             [152.,	76., 6.35], [152., 80., 10.], [200., 100., 9.5]]),
                       columns = ['d','bf','t'])
        d, bf, t =  cross_sec.iloc[SEC_ID,0], cross_sec.iloc[SEC_ID,1],  cross_sec.iloc[SEC_ID,2]
        coords = pd.DataFrame(np.array([[-bf/4, d/2-t/2], [0, d/2-t/2], [bf/4, d/2-t/2],
                                        [0, d/4], [0, 0], [0, -d/4],
                                        [-bf/4, -d/2+t/2], [0, -d/2+t/2], [bf/4, -d/2+t/2]]),
                       index=['TF-1','TF-M','TF-2','W-1','W-2','W-3','BF-1','BF-M','BF-2'],
                       columns = ['x','y'])

        z0 = data
        obs = np.vstack((coords.iloc[:,0],coords.iloc[:,1])).T
        interp = np.vstack((xi, yi)).T
        d0 = np.subtract.outer(obs[:,0], interp[:,0])
        d1 = np.subtract.outer(obs[:,1], interp[:,1])
        dist = np.hypot(d0, d1)
        verif = np.less_equal(dist, np.ones((1,dist.shape[1]))*10**(-6))

        try:
            
            weights = 1.0 / dist**2
            # Make weights sum to one
            weights /= weights.sum(axis=0)
            # Multiply the weights for each interpolated point by all observed Z-values
            zi = np.dot(weights.T, z0)
            
            return zi
            # if verif.any():
                
            #     zi = data.iloc[SEC_ID,np.where(np.any(verif==True, axis=1))[0][0]]
            # else:
                
            #     weights = 1.0 / dist
            #     # Make weights sum to one
            #     weights /= weights.sum(axis=0)
            #     # Multiply the weights for each interpolated point by all observed Z-values
            #     zi = np.dot(weights.T, z0)[0]
            # return zi
        
        except:

            return data.iloc[SEC_ID,np.where(np.any(verif==True, axis=1))[0][0]]
    
    results = IWD(incid[:,2],incid[:,3])
    return results

def assert_prop(incid, dkle_sample):
    '''
    incid: Matriz com dados [Elem numb, SECID, X, Y, RCID] **RCID = ID do resin content
    dkle_sample: Vetor do Resin Content do random field em ordem crescente do RCID
    '''
    
    resin_conts = np.zeros((incid[:,0].shape[0]))
    resin_conts = dkle_sample[(incid[:,4]-1).astype(np.int64)]
    
    return resin_conts

def create_from_template(template_filename, output_filename, keyword_values):
    """
    Create a new file by replacing keywords in a given template file
    """
    
    template_file = open(template_filename, "r")
    # Good Practice: always close a file
    try:
        template_content = template_file.read()
        template = Template(template_content)
        content = template.substitute(keyword_values)
    finally:
        template_file.close()
    
    output_file = open(output_filename, "w")
    try:
        output_file.write(content)
    finally:
        output_file.close()

#"C:\Program Files\ANSYS Inc\ANSYS Student\v241\ansys\bin\winx64\MAPDL.exe"
#"C:/Program Files/ANSYS Inc/ANSYS Student/v241/ansys/bin/winx64/MAPDL.exe"

#################### DEPRECATED ###################
def run_ansys(input_filename, output_filename):
    # job_name = "vm3-optimz"
    '''
    USAGE: D:\Program Files\ANSYS Inc\v241\ANSYS\bin\winx64\ANSYS.EXE
                    [-d device name] [-j job name]
                    [-b list|nolist] [-m scratch memory(mb)]
                    [-s read|noread] [-g] [-db database(mb)]
                    [-p product] [-l language]
                    [-dyn] [-np #] [-nt #]
                    [-dvt] [-dis] [-machines list]
                    [-i inputfile] [-o outputfile]
                    [-ser port] [-scport port]
                    [-scname couplingname] [-schost hostname]
                    [-smp] [-mpi intelmpi|openmpi|msmpi]
                    [-dir working_directory ]
                    [-acc nvidia|amd|intel] [-na #]
    '''
    # cwd=os.path.dirname(input_filename)

    # cmd_params = "D:/Program Files/ANSYS Inc/v241/ansys/bin/winx64/MAPDL.exe"+ " -b " + " -i " + input_filename + " -o "+output_filename + " -np 4 " + "-dir D:/Ansys_models"
    cmd_params = "C:/Program Files/ANSYS Inc/ANSYS Student/v251/ansys/bin/winx64/MAPDL.exe"+ " -b " + " -i " + input_filename + " -o "+output_filename + " -np 4 " + "-dir C:/Users/Leona/Documents/Mestrado/Python"
    # r"C:\Users\Leona\ufprenv\Scripts\activate.bat",
    # "C:\Users\Leona\ufprenv\Scripts\activate.bat",
    # "D:\Program Files\ANSYS Inc\v241\ansys\bin\winx64\MAPDL.exe"
    subprocess.call(cmd_params)



##### GRPC PYANSYS #########

# def run_ansys(mapdl,input_filename):
#     mapdl.clear()
#     mapdl.input(input_filename)
#     eigenvalue  = mapdl.get('Fcrit','ACTIVE',0,'SET','FREQ')
#     return eigenvalue

def evaluate_uniform(RC2):
    variables = {}
    TBELAST_1 = Ortoelastic_prop(RC2*0.01, 1300, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
    variables["MODULUSX1"] = TBELAST_1[0]*1000
    variables["MODULUSY1"] = TBELAST_1[1]*1000
    variables["SHEARXY1"] = TBELAST_1[2]*1000
    variables["SHEARYZ1"] = TBELAST_1[3]*1000    
    variables["POISSONXY1"] = TBELAST_1[4]
    create_from_template("mefbuck_flexural_uniformv2.tpl", "mefbuck_flexural_uniformv2.inp", variables)
    # ini = time.time()
    # eigenvalue = run_ansys(mapdl,"mefbuck_flexural.inp")
    # fim = time.time()
    run_ansys("mefbuck_flexural_uniformv2.inp","mefbuck_flexural_uniformv2.out")
    output_file = open("mefbuck_flexural_uniformv2.out", "r")
    try:
        template_content = output_file.read()
    finally:
        output_file.close()
    eigenvalue = float(re.search(" *GET  FCRIT     FROM  ACTI  ITEM=SET  FREQ  VALUE=.*?([\d\.]+)", template_content).group(1))
    # fim2 = time.time()
    # print("Tempo total iteracao {}/{}: {}".format(i+1,len(load),fim-ini)+"\n"+"Fcrit= {}".format(eigenvalue[i])+"\n")
    return eigenvalue

def evaluate_uniformtest(RC2):
    variables = {}
    TBELAST_1 = Ortoelastic_prop(RC2*0.01, 1300, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
    variables["MODULUSX1"] = TBELAST_1[0]*1000
    variables["MODULUSY1"] = TBELAST_1[1]*1000
    variables["SHEARXY1"] = TBELAST_1[2]*1000
    variables["SHEARYZ1"] = TBELAST_1[3]*1000    
    variables["POISSONXY1"] = TBELAST_1[4]
    create_from_template("testeuniform.tpl", "testeuniform.inp", variables)
    # ini = time.time()
    # eigenvalue = run_ansys(mapdl,"mefbuck_flexural.inp")
    # fim = time.time()
    run_ansys("testeuniform.inp","testeuniform.out")
    output_file = open("testeuniform.out", "r")
    try:
        template_content = output_file.read()
    finally:
        output_file.close()
    eigenvalue = float(re.search(" *GET  FCRIT     FROM  ACTI  ITEM=SET  FREQ  VALUE=.*?([\d\.]+)", template_content).group(1))
    # fim2 = time.time()
    # print("Tempo total iteracao {}/{}: {}".format(i+1,len(load),fim-ini)+"\n"+"Fcrit= {}".format(eigenvalue[i])+"\n")
    return eigenvalue

'TESTE EVALUATE UNIFORM'
# RC2 = 26.762962962962973

# test = evaluate_uniform(RC2, 5)

'EVALUATE GENERIC'

def evaluateGENERIC(RC1, RC2, RC3, RC4, RC5, RC6, RC7, RC8, RC9, SEC_ID):
    # data = pd.read_excel('resincontent.xlsx',header=0)
    incid = pd.read_excel('elem_data_fileatt.xlsx',header=0).to_numpy()
    cross_sec = pd.DataFrame(np.array([[100., 70., 6.], [102., 51., 6.4], [120.,70.,8.],
                                         [152.,	76., 6.35], [152., 80., 10.], [200., 100., 9.5]]),
                   columns = ['d','bf','t'])
    d, bf, t =  cross_sec.iloc[SEC_ID,0], cross_sec.iloc[SEC_ID,1],  cross_sec.iloc[SEC_ID,2]
    # RC3 = np.array(RC3)
    # eigenvalue = np.zeros(len(RC3))
    # for i in range(0,len(RC3)):
    variables = {}
    variables["d_beam"] = d
    variables["bf_beam"] = bf
    variables["t_beam"] = t
    variables["length"] = 4000
    # data = np.array([RC2[i], RC3[i], RC4[i], RC5[i], RC6[i], RC7[i], RC8[i], RC9[i], RC10[i]])
    data = np.array([RC1, RC2, RC3, RC4, RC5, RC6, RC7, RC8, RC9])
    data_sorted = np.sort(data)
    TBELAST_1 = Ortoelastic_prop(data_sorted[0]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_2 = Ortoelastic_prop(data_sorted[1]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_3 = Ortoelastic_prop(data_sorted[2]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_4 = Ortoelastic_prop(data_sorted[3]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_5 = Ortoelastic_prop(data_sorted[4]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_6 = Ortoelastic_prop(data_sorted[5]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_7 = Ortoelastic_prop(data_sorted[6]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_8 = Ortoelastic_prop(data_sorted[7]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    TBELAST_9 = Ortoelastic_prop(data_sorted[8]*0.01, 940, 2600, 3.4, 81, 1.3, 33, 0.35, 0.22)
    variables["FIBCONT1"] = data_sorted[0]
    variables["MODULUSX1"] = TBELAST_1[0]*1000
    variables["MODULUSY1"] = TBELAST_1[1]*1000
    variables["SHEARXY1"] = TBELAST_1[2]*1000
    variables["POISSONXY1"] = TBELAST_1[3]
    variables["FIBCONT2"] = data_sorted[1]
    variables["MODULUSX2"] = TBELAST_2[0]*1000
    variables["MODULUSY2"] = TBELAST_2[1]*1000
    variables["SHEARXY2"] = TBELAST_2[2]*1000
    variables["POISSONXY2"] = TBELAST_2[3]
    variables["FIBCONT3"] = data_sorted[2]
    variables["MODULUSX3"] = TBELAST_3[0]*1000
    variables["MODULUSY3"] = TBELAST_3[1]*1000
    variables["SHEARXY3"] = TBELAST_3[2]*1000
    variables["POISSONXY3"] = TBELAST_3[3]
    variables["FIBCONT4"] = data_sorted[3]
    variables["MODULUSX4"] = TBELAST_4[0]*1000
    variables["MODULUSY4"] = TBELAST_4[1]*1000
    variables["SHEARXY4"] = TBELAST_4[2]*1000
    variables["POISSONXY4"] = TBELAST_4[3]
    variables["FIBCONT5"] = data_sorted[4]
    variables["MODULUSX5"] = TBELAST_5[0]*1000
    variables["MODULUSY5"] = TBELAST_5[1]*1000
    variables["SHEARXY5"] = TBELAST_5[2]*1000
    variables["POISSONXY5"] = TBELAST_5[3]
    variables["FIBCONT6"] = data_sorted[5]
    variables["MODULUSX6"] = TBELAST_6[0]*1000
    variables["MODULUSY6"] = TBELAST_6[1]*1000
    variables["SHEARXY6"] = TBELAST_6[2]*1000
    variables["POISSONXY6"] = TBELAST_6[3]
    variables["FIBCONT7"] = data_sorted[6]
    variables["MODULUSX7"] = TBELAST_7[0]*1000
    variables["MODULUSY7"] = TBELAST_7[1]*1000
    variables["SHEARXY7"] = TBELAST_7[2]*1000
    variables["POISSONXY7"] = TBELAST_7[3]
    variables["FIBCONT8"] = data_sorted[7]
    variables["MODULUSX8"] = TBELAST_8[0]*1000
    variables["MODULUSY8"] = TBELAST_8[1]*1000
    variables["SHEARXY8"] = TBELAST_8[2]*1000
    variables["POISSONXY8"] = TBELAST_8[3]
    variables["FIBCONT9"] = data_sorted[8]
    variables["MODULUSX9"] = TBELAST_9[0]*1000
    variables["MODULUSY9"] = TBELAST_9[1]*1000
    variables["SHEARXY9"] = TBELAST_9[2]*1000
    variables["POISSONXY9"] = TBELAST_9[3]
    Res_cont = linear_interp(SEC_ID, data, incid)
    for j in range(0,len(incid[:,0])):
        variables["FC_e"+str(int(incid[j,0]))] = Res_cont[j]

    create_from_template("mefbuck_flexural_v2.tpl", "mefbuck_flexural_v2.inp", variables)
    # ini = time.time()
    # eigenvalue = run_ansys(mapdl,"mefbuck_flexural.inp")
    # fim = time.time()
    run_ansys("mefbuck_flexural_v2.inp","mefbuck_flexural_v2.out")
    output_file = open("mefbuck_flexural_v2.out", "r")
    try:
        template_content = output_file.read()
    finally:
        output_file.close()
    eigenvalue = float(re.search(" *GET  FCRIT     FROM  ACTI  ITEM=SET  FREQ  VALUE=.*?([\d\.]+)", template_content).group(1))
    # fim2 = time.time()
    # print("Tempo total iteracao {}/{}: {}".format(i+1,len(load),fim-ini)+"\n"+"Fcrit= {}".format(eigenvalue[i])+"\n")
    return eigenvalue

'EVALUATE COM RANDOM FIELD'

def evaluateRF(RCvec, x, y, incid, meanval, stdv):

    dkle_sample = sample_RandomField(RCvec, x, y, meanval, stdv)
    Res_cont = assert_prop(incid, dkle_sample)
    TBELAST = np.zeros((Res_cont.shape[0]))
    data = Res_cont.copy()
    data_sorted = np.sort(data)
    TBELAST = Ortoelastic_prop(data_sorted*0.01, 1300, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
    variables = {}
    for j in range(Res_cont.shape[0]):
        variables["FIBCONT"+str(int(j+1))] = data_sorted[j]
        variables["MODULUSX"+str(int(j+1))] = TBELAST[0][j]*1000
        variables["MODULUSY"+str(int(j+1))] = TBELAST[1][j]*1000
        variables["SHEARXY"+str(int(j+1))] = TBELAST[2][j]*1000
        variables["SHEARYZ"+str(int(j+1))] = TBELAST[3][j]*1000
        variables["POISSONXY"+str(int(j+1))] = TBELAST[4][j]        
        variables["FC_e"+str(int(j+1))] = Res_cont[j]

    create_from_template("mefbuck_flexural_v5.tpl", "mefbuck_flexural_v5.inp", variables)
    global timecount
    ini = time.perf_counter()
    run_ansys("mefbuck_flexural_v5.inp","mefbuck_flexural_v5.out")
    output_file = open("mefbuck_flexural_v5.out", "r")
    try:
        template_content = output_file.read()
    finally:
        output_file.close()
    eigenvalue = float(re.search(" *GET  FCRIT     FROM  ACTI  ITEM=SET  FREQ  VALUE=.*?([\d\.]+)", template_content).group(1))
    fim = time.perf_counter()
    timecount+=fim-ini
    # print("Tempo total iteracao {}/{}: {}".format(i+1,len(load),fim-ini)+"\n"+"Fcrit= {}".format(eigenvalue[i])+"\n")
    print("Resin contents: ", dkle_sample)
    print("Buckling load: ", eigenvalue)
    print("Elapsed time: ", fim-ini)
    return eigenvalue

def evaluate_LHS(RCvec):

    Res_cont = RCvec
    TBELAST = np.zeros((Res_cont.shape[0]))
    data = Res_cont.copy()
    data_sorted = np.sort(data)
    TBELAST = Ortoelastic_prop(data_sorted*0.01, 1300, 2600, 3.7, 81, 1.4, 33, 0.35, 0.22)
    variables = {}
    for j in range(Res_cont.shape[0]):
        variables["FIBCONT"+str(int(j+1))] = data_sorted[j]
        variables["MODULUSX"+str(int(j+1))] = TBELAST[0][j]*1000
        variables["MODULUSY"+str(int(j+1))] = TBELAST[1][j]*1000
        variables["SHEARXY"+str(int(j+1))] = TBELAST[2][j]*1000
        variables["SHEARYZ"+str(int(j+1))] = TBELAST[3][j]*1000
        variables["POISSONXY"+str(int(j+1))] = TBELAST[4][j]        
        variables["FC_e"+str(int(j+1))] = Res_cont[j]

    create_from_template("mefbuck_flexural_v5.tpl", "mefbuck_flexural_v5.inp", variables)
    # ini = time.time()
    # eigenvalue = run_ansys(mapdl,"mefbuck_flexural.inp")
    # fim = time.time()
    run_ansys("mefbuck_flexural_v5.inp","mefbuck_flexural_v5.out")
    output_file = open("mefbuck_flexural_v5.out", "r")
    try:
        template_content = output_file.read()
    finally:
        output_file.close()
    eigenvalue = float(re.search(" *GET  FCRIT     FROM  ACTI  ITEM=SET  FREQ  VALUE=.*?([\d\.]+)", template_content).group(1))
    # fim2 = time.time()
    # print("Tempo total iteracao {}/{}: {}".format(i+1,len(load),fim-ini)+"\n"+"Fcrit= {}".format(eigenvalue[i])+"\n")
    # print("Resin contents: ", dkle_sample)
    # print("Buckling load: ", eigenvalue)
    return eigenvalue

'''
ANSYS APDL MACRO
*DIM,LABEL,CHAR,1,1
*DIM,VALUE,,1,1
LABEL(1,1) = 'Load_fac'
*VFILL,VALUE(1,1),DATA,Fcrit
/COM
/COM
*VWRITE,LABEL(1,1),VALUE(1,1)
(1X,A8,F14.0)
!/OUT,buckbeam,D:\Ansys_models,out
'''