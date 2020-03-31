Here are the steps one has to follow to run this algorithm:

## 0. Requirements to run this algorithm
Tensorflow 1.12

## 1. Dataset
You need a training set and a test set. This algorithm works with meshes, so the train and test set need to be made of meshes (off files are supported).
Also, make sure your meshes are aligned to one axis . They can be rotated along this axis if you use data augmentation.
Basically, train and test sets need to be aligned to the same axis.

## 2. Preprocess data
Go to MATLAB_TOOLS folder, adapt the code there to your dataset.
This will compute the eigen functions of the intrinsic Laplacian on the meshes and store them in a «spectral/» folder.

## 3. Create a dataset and config class for you train and test set
Create a file with the name of your training set for instance, with the parameters you want for your model. link the paths so the algo knows where to find the data it requires to run properly.

## 4. Link paths and run train.py
You should be ready to train your model ! Use your new dataset class in train.py, your model will be stored in the «results/» folder.

## 5. Test your model by running test.py
Take the log name of your model inside the «results/» folder, and put it in your test.py file. Change the test set to the one you want. The test results will be .npy files, storing the functional maps, and output descriptors, in the «test/» folder.

## 6. Evaluation of the results
Go to «eval_scripts/» folder. You need ground truth between the pairs of shapes you want to evaluate. The current methods evaluates the error by computing the geodesic distance between the ground truth point and the point your model predicted on the target shape, for each point of the source shape. That requires the Geodesic Distance Matrix of the target shape, that you need to store and link to the evaluation script.

## 7. Have fun
You can now get a score ! play with the KPConv architecture and see if you can get better results.
