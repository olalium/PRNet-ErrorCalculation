# Error calculation for PRNet (front, side and average)
Code for calculating the error of PRNet compared to ground truth using point-to-point euclidean distance divided by the interocular distance as metric. User can choose the angle of the input image and number of images to evaluate.

## Getting Started

### Prerequisites
* Python 3.7
* Numpy, skimage, scipy
* matplotlib for showing results
* TensorFlow >= 1.4
* dlib

Clone the project repository. 
```bash
https://github.com/olalium/PRNet-ErrorCalculation.git
``` 

### Installing PRNet

1. clone the PRNet repository in the `src/` folder
```bash
git clone https://github.com/YadiraF/PRNet
``` 
2. download the PRN trained model(links found in PRNet readme) and put it in `PRNet/Data/net-data`

### Installing ICP implementation

1. clone the ICP repositoriy implemented by Clay Flannigan into the `src/` folder.
```bash
git clone https://github.com/ClayFlannigan/icp
``` 
