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
from extractDescriptorsAndGt import LocalBinaryPatterns,getNumberOfPointsInsideRectangle,extractDataForOneImage
from trainAndTestLBP import computeError


def drawImageWithPredictions(image_color,nbPointGT,predictions,widthOfPatch,heightOfPatch):
    predictionImage=np.copy(image_color)
    for i in range(args.numberOfRectanglePerColumn):
        for j in range(args.numberOfRectanglePerRow):
            rectangle={"anchor":(i*heightOfPatch,j*widthOfPatch),"width":widthOfPatch,"height":heightOfPatch}
            cv2.rectangle(predictionImage,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),1)
    
            cv2.putText(image_color, str(max(0,nbPointGT[i,j])), (int((j+0.5)*(widthOfPatch)),int((i+0.5)*(heightOfPatch))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), thickness=3,lineType=cv2.LINE_AA)
            cv2.rectangle(image_color,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),1)
            

            cv2.putText(predictionImage, str(max(0,int(predictions[i,j]))), (int((j+0.5)*(widthOfPatch)),int((i+0.5)*(heightOfPatch))), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0),thickness=3, lineType=cv2.LINE_AA) 
            #cv2.imshow("prediction_image",predictionImage)
    return image_color, predictionImage
               
    
def main(args):
    
    images_list, gts_list = utils.get_data_list(args.data_root, mode='test')
    try:
        pathToSaveImage=os.path.join(args.data_root,"imagesWithPredictions")
        os.makedirs(pathToSaveImage)
        
    except:
        print("Folder already exist!")
    
    matFile=sio.loadmat(os.path.join(args.data_root,"train"))
    descriptorsTrain=matFile["lbp_descriptors"]
    labelsTrain=matFile["labels"]

    matFile=sio.loadmat(os.path.join(args.data_root,"test"))
    descriptorsTest=matFile["lbp_descriptors"]
    labelsTest=matFile["labels"]
    
    alpha=args.alpha
    gamma=args.gamma
    
    KRR = KernelRidge(kernel='rbf', alpha=alpha, gamma=gamma)
    KRR.fit(descriptorsTrain, labelsTrain)
    
    predictionsTrain=KRR.predict(descriptorsTrain)
    MSE_train=computeError(labelsTrain, predictionsTrain)

    predictionsTest=KRR.predict(descriptorsTest)
    MSE_test=computeError(labelsTest, predictionsTest)
    
    print("minTestError:"+str(MSE_test), "minTrainError:"+ str(MSE_train))
    
 
    for img_idx in range(len(images_list)):
        
        matFile=sio.loadmat(gts_list[img_idx])
        train_head_points = matFile['image_info'][0][0][0][0][0]
        image_color = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)

        ###Convert to grayscale if the image is not in grayscale already
        height,width,numChannels=image_color.shape
       
       
        widthOfPatch=int(width/float(args.numberOfRectanglePerRow))
        heightOfPatch=int(height/float(args.numberOfRectanglePerColumn))

        lbp=LocalBinaryPatterns(numPoints=8, radius=1)
        lbpWholeImage=[]
        
                
        # Load the image and ground truth
        rawImage = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)
        matFile=sio.loadmat(gts_list[img_idx])
        headPoints = matFile['image_info'][0][0][0][0][0]            
        nbPointInImage,lbpWholeImage=extractDataForOneImage(rawImage,headPoints,lbp,args.numberOfRectanglePerRow,args.numberOfRectanglePerColumn)
        
        for x,y in list(headPoints):
            cv2.circle(image_color, (np.int(x),np.int(y)),2,(0, 255, 0),-1)
        
        """cv2.rectangle(image_color,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),3)
        for x,y in list(points):
            cv2.circle(image_color, (np.int(x),np.int(y)), 2, (0, 255, 0), -1)
        cv2.imshow("image",train_image_color)
        k=cv2.waitKey(1)"""
        
        
        predictions=(KRR.predict(lbpWholeImage.reshape(1,-1))).reshape(args.numberOfRectanglePerColumn,args.numberOfRectanglePerRow)
        predictionImage=np.copy(image_color)
        nbPointInImage=np.array(nbPointInImage).reshape(args.numberOfRectanglePerColumn,args.numberOfRectanglePerRow)
        image_color,predictionImage=drawImageWithPredictions(image_color,nbPointInImage,predictions,widthOfPatch,heightOfPatch)

        predictionAndGroundTruthImage=255*np.ones([predictionImage.shape[0],2*predictionImage.shape[1]+30,3])
        
        ###cv2.imshow("image",image_color)

        predictionAndGroundTruthImage[:,0:predictionImage.shape[1],:]=image_color
        predictionAndGroundTruthImage[:,-predictionImage.shape[1]:,:]=predictionImage
        ###cv2.imshow("both",np.uint8(predictionAndGroundTruthImage))
        ###k=cv2.waitKey(1)
        #print(os.path.join(pathToSaveImage,str(img_idx)+'.jpg'))
        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'.jpg'), predictionImage)
        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'_gts.jpg'), image_color)
        cv2.imwrite(os.path.join(pathToSaveImage,str(img_idx)+'_both.jpg'), predictionAndGroundTruthImage) 


                
                


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../ShanghaiTech/ShanghaiTech/part_B', type=str)
    parser.add_argument('--numberOfRectanglePerRow', default=16, type=int)
    parser.add_argument('--numberOfRectanglePerColumn', default=12, type=int)
    parser.add_argument('--alpha', default=0.01, type=float)
    parser.add_argument('--gamma', default=0.003593813, type=float)

    args = parser.parse_args()

    main(args)
    
