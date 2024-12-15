import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
from scipy.linalg import qr
import pickle
from tqdm import tqdm
import pywt
#from random import randint
import random

BAD = ['VT', 'VFb', 'VFt'] # KIS,S
project_dir = r'C:\Users\jodge\Documents\School\Summer24\tinymlcontest2022_demo_example'
dir_path = r'C:\Users\jodge\Documents\School\Summer24\tinyml_contest_data_training'
directory_files = os.listdir(dir_path)

training_script = os.path.join(project_dir, 'training_save_deep_models.py')
testing_script = os.path.join(project_dir, 'testing_performances.py')
models_dir = os.path.join(project_dir, 'models')
results_dir = os.path.join(project_dir, 'results')

"""
Energy captured: ranks needed
 5%: 14  10%: 29  15%: 46  20%: 64   25%:  84
30%: 106 35%: 129 40%: 154 45%: 182  50%:  215
55%: 252 60%: 297 65%: 354 70%: 443  75%:  552
80%: 670 85%: 798 90%: 935 95%: 1084 100%: 1250 
"""

label_path = os.path.join(project_dir, 'label_list.csv')
df_labels = pd.read_csv(label_path, sep=',', header=0)
label_dict = {k : v for k,v in df_labels.values}

def readObj(filename):
    print(f"reading from {filename}")
    with open(filename, 'rb') as file:
       return pickle.load(file)
def saveObj(obj, filename):
    print(f"saving to {filename}")
    with open(filename, 'wb') as file:
        pickle.dump(obj, file)        
def getStyle(file_name):
    return 'b-' if file_name.split('-')[1] not in BAD else 'r-'

def setCorrectModel(path, wanted = "original"):
    if wanted == "original":
        #os.rename(os.path.join(path, 'model_1.py.og'), os.path.join(path,'model_1.py'))
        return f"ren {os.path.join(path, 'model_1.py.og')} model_1.py"
    elif wanted == "avgpool":
        #os.rename(os.path.join(path, 'model_1.py.avgpool'), os.path.join(path,'model_1.py'))
        return f"ren {os.path.join(path, 'model_1.py.avgpool')} model_1.py"
        
def resetModel(models_dir, used = "original"):
    if used == "original":
        #os.rename(models_dir, os.path.join(models_dir, 'model_1.py.og'))
        return f"ren {os.path.join(models_dir, 'model_1.py')} model_1.py.og"
    elif used == "avgpool":
        #os.rename(models_dir, os.path.join(models_dir,'model_1.py.avgpool'))
        return f"ren {os.path.join(models_dir, 'model_1.py')} model_1.py.avgpool"
    else:
        print("BAD!!!!!!!!!!!!!")

####################### SVD ######################
####################### SVD ######################
####################### SVD ######################

def prepareData(problemSetup): #for SVD
    #problemSetupsDir = r'C:\Users\jodge\Documents\School\Summer24\playground\data'
    #problemSetup = readObj(os.path.join(problemSetupsDir, 'full.pkl'))
    print(problemSetup.r, problemSetup.p) # make sure it's working
        
    for fi in tqdm(directory_files[:]): #special iterator that makes a progress bar
        signal = np.loadtxt(os.path.join(dir_path,fi))
        mse, recon = measureAndReconstruct(problemSetup, signal)

        #FOR THE TRUNCATED REPRESENTATION ###############################
        #trunc = problemSetup.C @ temp # not going to do this anymore, we want to preserve the natural ordering / time information

        #with zeroes for where we didn't want info so the signal "looks" like the original
        #trunc = indices_to_zero = np.setdiff1d(np.arange(trunc.size), pivots)
        #np.put(trunc, indices_to_zero, 0)
        
        trunc = signal[sorted(problemSetup.pivots)]

        outReconName = os.path.join(r'C:\Users\jodge\Documents\School\Summer24\recon_data', 'R' + fi) 
        outTruncName = os.path.join(r'C:\Users\jodge\Documents\School\Summer24\trunc_data', 'T' + fi)
        np.savetxt(outReconName, recon, fmt='%.7f')
        np.savetxt(outTruncName, trunc, fmt='%.7f')
    
    #print("DONE")
    return r, p # ive been using p (which should be r+1) for file naming. idk why i chose that convention.

class SVD:
    def __init__(self, U, S, VT):
        self.U = U
        self.S = S
        self.VT = VT

class ProblemSetup:
  def __init__(self, r, p, truncSVD, Q, R, pivots, C): #fullSVD,
      self.r = r # ranks to use
      self.p = p # number of sensors
      #self.fullSVD = fullSVD
      self.truncSVD = truncSVD
      self.Q = Q
      self.R = R
      self.pivots = pivots
      self.C = C
      
def getFullTrainSignalMatrix(): #all files with each as a column
    pkl_name = r"fullTrainMat.pkl"
    pkl_loc = os.path.join("data", pkl_name)

    if os.path.exists(pkl_loc):
        answer = readObj(pkl_loc)
        return answer
    else:
        print(f"{pkl_loc} not found - see other notebooks for generation")
        
def processData(trainingData, label, r = 400, p = 500): # r = ranks desired, p = num sensors. bug when r=p, math is hard.
    pkl_name = f"{label}_r{r}_p{p}.pkl"
    pkl_loc = os.path.join("data", pkl_name)

    if os.path.exists(pkl_loc):
        answer = readObj(pkl_loc)
        return answer
    
    U, S, VT = svd(trainingData)
    #full_SVD = SVD(U, S, VT)
    
    #reshape SVD according to r
    U_hat, S_hat, VT_hat = U[:,:r], S[:r,:r], VT[:r,:]
    trunc_SVD = SVD(U_hat, S_hat, VT_hat)
    
    Q, R, pivots = None, None, None
    if (p == r):
        Q, R, pivots = qr(U, pivoting = True) # or maybe just U! - we did this change!
    elif (p > r): # oversampled
        Q, R, pivots = qr(U_hat @ U_hat.T, pivoting = True) # or maybe just U
    else:
        for _ in range(100):
            print("ERROR p < r")
            
    pivots = pivots[:p]
    
    # Create C matrix
    C = np.zeros((p, getFullTrainSignalMatrix().shape[0]))
    #print(C.shape, pivots.shape)
    C[np.arange(p), pivots] = 1

    problemSetup = ProblemSetup(r, p,  trunc_SVD, Q, R, pivots, C) #full_SVD,
    #filename = os.path.join("data", label + ".pkl")
    saveObj(problemSetup, pkl_loc)
    
    return problemSetup
    
def svd(x): # = getFullTrainSignalMatrix()
    U, S, VT = np.linalg.svd(x, full_matrices=True) #full_matrices=False
    S = np.diag(S)
    return (U, S, VT)

def measureAndReconstruct(problemSetup, signal):
    # Measure a signal
    C, U_hat, p, r, pivots =  problemSetup.C, problemSetup.truncSVD.U, problemSetup.p, problemSetup.r, problemSetup.pivots
    y = C @ signal
    
    # Solve for coefficients
    U_k_reduced = U_hat[:, :p][pivots, :] # suggested to try
    
    if p == r:
        a = np.linalg.solve(C @ U_hat, y)
    else:
        a = np.linalg.pinv(C @ U_hat) @ y

    x_reconstructed = U_hat @ a
    mseFinal = np.mean((signal - x_reconstructed) ** 2)
    return mseFinal, x_reconstructed


def getFileInfo(file_name):
    parts = file_name.split('-')
    pn = parts[0][1:]
    cat = parts[1]
    return pn, cat


####################### Plotting ######################
####################### Plotting ######################
####################### Plotting ######################

def getRandomCategorySignal(cat = 'SR'):
    if cat not in label_dict.keys():
        return None
    idx = random.randint(0, len(directory_files))
    file_name = directory_files[idx]
    is_cat = file_name.split('-')[1] == cat
    if is_cat:
        file_path = os.path.join(dir_path, file_name)
        x1 = np.loadtxt(file_path)
        return x1, file_name
    else:
        return getRandomCategorySignal(cat)
        
def getRandomAlignedSignal(good = True): #dnd joke function name
    idx = random.randint(0, len(directory_files))
    file_name = directory_files[idx]
    is_good = file_name.split('-')[1] not in BAD
    if is_good == good:
        file_path = os.path.join(dir_path, file_name)
        x1 = np.loadtxt(file_path)
        return x1, file_name
    else:
        return getRandomAlignedSignal(good)
