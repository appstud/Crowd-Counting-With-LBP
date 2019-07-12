import scipy.io as sio
import cv2
import argparse
import utils as utils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pdb
from skimage import feature
import numpy as np
import os
import time 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import make_scorer

def computeError(y, y_pred):
    return np.sqrt(np.mean((np.sum(y_pred,1)-np.sum(y,1))**2))
    

def main(args):
    

    ###Read descriptors and ground truth for train set
    matFile=sio.loadmat(os.path.join(args.data_root,"train"))
    descriptorsTrain=matFile["lbp_descriptors"]
    labelsTrain=matFile["labels"]

    ###Read descriptors and ground truth for test set
    matFile2=sio.loadmat(os.path.join(args.data_root,"test"))
    descriptorsTest=matFile2["lbp_descriptors"]
    labelsTest=matFile2["labels"]
    
    print("descriptors train shape:"+ str(descriptorsTrain.shape))
    print("labels train shape:"+ str(labelsTrain.shape)+"\n")
    print("descriptors test shape:"+ str(descriptorsTest.shape))
    print("labels test shape:"+ str(labelsTest.shape)+"\n")
    
    Grid_Dict = {"alpha": [1e-13,1e-5,1e-4,1e-3,1e-2],"gamma": np.logspace(-3, 2, 10)}
    krr_Tuned = GridSearchCV(KernelRidge(kernel='rbf'), cv=4 ,param_grid=Grid_Dict, scoring=make_scorer(computeError,greater_is_better=False),refit=True)
    krr_Tuned.fit(descriptorsTrain, labelsTrain)

    
    predictionsTrain=krr_Tuned.predict(descriptorsTrain)
    MSE_train=computeError(labelsTrain, predictionsTrain)


    ########testing###########  
    predictionsTest=krr_Tuned.predict(descriptorsTest)
    MSE_test=computeError(labelsTest, predictionsTest)
    

    print("  minTestError:"+str( MSE_test),  " minTrainError:"+ str(MSE_train), "Best alpha:"+ str(krr_Tuned.best_params_['alpha']), "Best rbf param:"+ str(krr_Tuned.best_params_['gamma']))

    
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../ShanghaiTech/ShanghaiTech/part_B', type=str)
    args = parser.parse_args()

    main(args)
    
