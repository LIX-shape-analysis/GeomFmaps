# Deep Geometric Maps: Robust Feature Learning for Shape Correspondence

This is our implementation of Deep Geometric Maps, a Shape Matching Network that builds comprehensive features from point cloud data, projects them in spectral basis to compute accurate functional maps. Here is a [link to the paper](https://arxiv.org/abs/2003.14286).

![TEASER](https://raw.githubusercontent.com/LIX-shape-analysis/GeomFmaps/master/images/TEASER.png "TEASER")

This code was written by Nicolas Donati, although a large part of it is taken from the code of [KPConv](https://github.com/HuguesTHOMAS/KPConv). Indeed, we use KPConv as a feature extractor in our method.

Here are the steps to follow to run this algorithm:

## Prerequisites
* CUDA and CuDNN (changing the code to run on CPU should require few changes)
* Python 3.7
* Tensorflow 1.12 (there may be some issues with more recent versions of Tensorflow, as discussed in KPConv)

Clone this repository:
``` bash
git clone https://github.com/LIX-shape-analysis/GeomFmaps.git
cd GeomFmaps
```

## Setup

* Datasets : you will need a training set and a test set. This algorithm works with meshes, so the train and test set need to be made of meshes (off files are supported).
Also, make sure your meshes are **aligned to one axis**. They can be rotated along this axis if you use data augmentation. Naturally, train and test sets need to be aligned to the same axis.

* Preprocessing : go to MATLAB_TOOLS folder, adapt the code there to your dataset. The matlab script will compute the eigen functions of the intrinsic Laplacian on the meshes and store them in a «spectral/» folder.

* Add a new config Class : Create a file with the name of your training (and test) set, with the parameters you want for your model in the «config/» folder. Some are already included (FAUST, SCAPE and SHREC re-meshed for instance).

* Add a new dataset Class : Create a file with the name of your training (and test) set, with the parameters (especially the paths to your data) for your dataset in the «dataset/» folder. Some are already included (FAUST, SCAPE and SHREC re-meshed for instance).

## Training
You should be ready to train your model ! Use your new dataset class in train.py, your model will be stored in the «results/» folder. You can now run :
``` bash
python train.py
```

## Testing
Take the log name of your model inside the «results/» folder, and put it in your test.py file. Also change the dataset class to match that of your test set. You can now run :
``` bash
python test.py
```
The test results will be stored in .npy files, containing the functional maps, and output descriptors, in the «test/» folder.

## Evaluation of the results
Go to «eval_scripts/» folder. **You need ground truth between the pairs of shapes you want to evaluate**. As described in the paper, the error corresponds to the geodesic distance between the ground truth point and the point your model predicted on the target shape, for each point of the source shape. That requires the Geodesic Distance Matrix of the target shape, that you need to store and link to the evaluation script. You can then run :
``` bash
cd eval_scripts
python test.py
```

## Have fun
You can now get a score ! play with the KPConv architecture and see if you can get better results.

## Citation
If you use our work, please cite our paper.
```
@article{donati2020deepGeoMaps,
  title={Deep Geometric Maps: Robust Feature Learning for Shape Correspondence},
  author={Donati, Nicolas and Sharma, Abhishek and Ovsjanikov, Maks},
  journal={CVPR},
  year={2020}
}
```

## Contact
If you have any problem about this implementation, please feel free to contact via:

nicolas.donati AT polytechnique DOT edu
