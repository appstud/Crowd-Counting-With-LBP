# Crowd-Counting-With-LBP
## Prerequisites
- Python 3.6
- SciPy library
- SciKit image library
- OpenCV library
- SKlearn library

## Quick start

### Download the ShanghaiTech Database by looking into the repository and put it in the root folder

    https://github.com/desenzhou/ShanghaiTechDataset
    

### Run the extractDescriptorsAndGt.py file to extract lbp descriptor and groundtruth and save them in mat files
    python extractDescriptorsAndGt.py --data_root="../ShanghaiTech/ShanghaiTech/part_B" --showImg=1

### Run the trainAndTestLBP.py file to train and test different parameters from the ridge regression (alpha and gamma)
    python trainAndTestLBP.py --data_root="../ShanghaiTech/ShanghaiTech/part_B"

### Run predictionsLBP.py file to train the ridge regression with suitable parameters and save results of the predictions as images
    python  predictionsLBP.py --data_root="../ShanghaiTech/ShanghaiTech/part_B" --alpha=0.001 --gamma=0.0036
