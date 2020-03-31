Here are the steps one has to follow to run this algorithm:

0) requirements to run this algorithm :
trimesh, tensorflow, etc...

1) get a training set and a test set. This algorithm works with meshes, so the train and test set need to be made of meshes (we support off and ply files). Also, right now this algorithm can only deal with meshes that are aligned to one axis (they can be rotated along this axis). Basically, that means you need to make sure train and test sets are aligned to the same axis if you want to learn the matching.

2) preprocess data. Run preprocess_all_data (matlab code ?).
this will compute some functions over your meshes, called eigen functions of the intrinsic laplacian.
they are stored under "spectral" folders and will be used for the loss, mainly. to each of these eigenfunctions corresponds an eigenvalue. That is why we store, evecs, evals, and evecs_trans (which is the Moore pseudo-inverse of evecs.. can we remove it ? store only weight matrix ?).

3) create a file with the name of your training set for instance, with the parameters you want for your model. link the paths so the algo knows where to find the data it requires to run properly.

4) you should be ready to train your model ! create a script with your running file. Your model will be stored in Results. You can check out the functional maps it produces while training to see if they look reasonnable.

5) prepare the paths to your test set and test your model. The results will be stored in tests.

6) evaluation of the tests. this requires to adapt the evaluation files already present in the code. The current method evaluates the error by computing the geodesic distance between the ground truth point and the point your model predicted, for each point of the source shape. That requires the Geodesic Distance Matrix, that you need to store and link to the evaluation script.

7) you can now get a score ! play with the KPConv architecture and see if you can get better results.
