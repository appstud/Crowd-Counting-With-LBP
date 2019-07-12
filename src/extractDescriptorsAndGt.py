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

class LocalBinaryPatterns:
    def __init__(self, numPoints, radius):
        # store the number of points and radius
        self.numPoints = numPoints
        self.radius = radius

    def describe(self, image, eps=1e-7):
        # compute the Local Binary Pattern representation
        # of the image, and then use the LBP representation
        # to build the histogram of patterns
        lbp = feature.local_binary_pattern(image, self.numPoints,
                self.radius, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(),
                bins=np.arange(0, 60))

        # normalize the histogram
        hist = hist.astype("float")
        hist /= (hist.sum() + eps)

        # return the histogram of Local Binary Patterns
        return hist

    
def getNumberOfPointsInsideRectangle(points,rectangle={"anchor":(0,0),"width":200,"height":200}):
    ### points: a 2D array of points, each line is a sample.
    ### rectangle: a dictionary that defines a rectangle by its top left corner, width and height.
    ### returns the points inside the rectangle and their number.
    
    mask=(points[:,0]<(rectangle["anchor"][1]+rectangle["width"])) & (points[:,0]> rectangle["anchor"][1])
    mask=(mask) & (points[:,1]<(rectangle["anchor"][0]+rectangle["height"])) & (points[:,1]> rectangle["anchor"][0])
    
    return np.sum(mask),points[mask,:]

def drawPointsOnImage(image,points):
    ###image: RGB image to draw points on
    ###points: the points to be drawn
    ###returns: the image after the drawing operation
    
    for x,y in list(points):
        cv2.circle(image, (np.int(x),np.int(y)), 1, (0, 255, 0), -1)
    return image

def extractDataForOneImage(image,points,lbp,numberOfRectanglePerRow,numberOfRectanglePerColumn,showImg):
    
    
    ###Convert to grayscale if the image is not in grayscale already
    height,width,numChannels=image.shape
    if(numChannels==3):
        grayImage=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        grayImage=rawImage
    nbPointInImage=[]
    widthOfPatch=int(width/float(numberOfRectanglePerRow))
    heightOfPatch=int(height/float(numberOfRectanglePerColumn))
    allHistInImage=[]
    
    for i in range(numberOfRectanglePerColumn):
        for j in range(numberOfRectanglePerRow):
            ###Get the points inside a patch and their numbers
            nbPoint,headPointsInPatch=getNumberOfPointsInsideRectangle(points,rectangle={"anchor":(i*heightOfPatch,j*widthOfPatch),"width":widthOfPatch,"height":heightOfPatch})
            
            ###Add it to a vector that contains the number of points in a each patch
            nbPointInImage.append(nbPoint)

            ###Draw the boundaries of the current patch on the image
            cv2.rectangle(image,(j*widthOfPatch,i*heightOfPatch),(j*widthOfPatch+widthOfPatch,i*heightOfPatch+heightOfPatch),(0,0,255),3)
            
            ###Draw the points available in the current patch on the image
            rawImage=drawPointsOnImage(image,headPointsInPatch)

            ###Show the image
            if (showImg):
                cv2.imshow("image",rawImage)
                k=cv2.waitKey(1)

            ###Calculate the LBP uniform histogram descriptor for the current patch
            hist=lbp.describe(grayImage[i*heightOfPatch:(i+1)*heightOfPatch,j*widthOfPatch:(j+1)*widthOfPatch], eps=1e-7)

            ###Append current 59D descriptor to descriptors of previous patches in a list
            allHistInImage.append(hist)
            
    ###Concatenate all descriptors of all patches into one descriptor vector  
    allHistInImage=np.concatenate(allHistInImage, axis=0)

    return nbPointInImage,allHistInImage




def extractDescriptorsAndGroundTruthMatrix(args,mode="train"):
    #Get the list of images.
    images_list, gts_list = utils.get_data_list(args.data_root, mode=mode)
    allPoints=None
    allHist=None
    lbp=LocalBinaryPatterns(numPoints=8, radius=1)
    for img_idx in range(len(images_list)):
        
        # Load the image and ground truth
        rawImage = np.asarray(cv2.imread(images_list[img_idx]), dtype=np.uint8)
        matFile=sio.loadmat(gts_list[img_idx])
        if img_idx%10 == 0:
            print("Image processed : "+str(img_idx)+"/"+str(len(images_list)))
        headPoints = matFile['image_info'][0][0][0][0][0]
            
        nbPointInImage,allHistInImage=extractDataForOneImage(rawImage,headPoints,lbp,args.numberOfRectanglePerRow,args.numberOfRectanglePerColumn,args.showImg)
        
        ###Concatenate all descriptors into one matrix and all vectors of number of points into one matrix   
        if(allPoints is None):
            allPoints=np.array(nbPointInImage)
        else:
            allPoints=np.vstack((allPoints,nbPointInImage))

        if(allHist is None):
            allHist=allHistInImage

        else:
            allHist=np.vstack((allHist,allHistInImage))
        
    ###Save the extracted descriptors matrix and ground truth matrix
    sio.savemat(os.path.join(args.data_root,mode+".mat"),{"lbp_descriptors":allHist,"labels":allPoints})
      
    print(mode+ "set descriptors shape:" +str(allHist.shape))
    print(mode+ "set ground truth shape:" +str(allPoints.shape))

def main(args):
    ###Extract descriptors and ground truth for the train set
    extractDescriptorsAndGroundTruthMatrix(args,mode="train")

    ###Extract descriptors and ground truth for the test set
    extractDescriptorsAndGroundTruthMatrix(args,mode="test")
    



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', default='../ShanghaiTech/part_B', type=str)
    parser.add_argument('--numberOfRectanglePerRow', default=16, type=int)
    parser.add_argument('--numberOfRectanglePerColumn', default=12, type=int)
    parser.add_argument('--showImg', default=0, type=int)

    args = parser.parse_args()

    main(args)
