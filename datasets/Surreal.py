# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling Surreal Dataset (training and testing)
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#      Nicolas DONATI - 01/01/2020


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Basic libs
import tensorflow as tf
import numpy as np

# Dataset parent class
from datasets.common import Dataset
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling

# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#

def grid_subsampling(points, features=None, labels=None, sampleDl=0.1, verbose=0):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features
    :param points: (N, 3) matrix of input points
    :param features: optional (N, d) matrix of features (floating number)
    :param labels: optional (N,) matrix of integer labels
    :param sampleDl: parameter defining the size of grid voxels
    :param verbose: 1 to display
    :return: subsampled points, with features and/or labels depending of the input
    """

    if (features is None) and (labels is None):
        return cpp_subsampling.compute(points, sampleDl=sampleDl, verbose=verbose)
    elif (labels is None):
        return cpp_subsampling.compute(points, features=features, sampleDl=sampleDl, verbose=verbose)
    elif (features is None):
        return cpp_subsampling.compute(points, classes=labels, sampleDl=sampleDl, verbose=verbose)
    else:
        return cpp_subsampling.compute(points, features=features, classes=labels, sampleDl=sampleDl, verbose=verbose)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Class Definition
#       \***************/
#


class SurrealDataset(Dataset):
    """
    Class to handle any subset of 5000 shapes of the surreal dataset introduced in 3D coded (for comparison in exp2)
    this dataset is composed of 6890-points shapes, so the spectral data is relatively heavy.
    """

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, config):
        Dataset.__init__(self, 'surreal')

        ####################
        # Dataset parameters
        ####################

        # Type of task conducted on this dataset
        # self.network_model = 'shape_matching'  # this is the only type of model here but it comes from KPConc code

        ##########################
        # Parameters for the files
        ##########################

        # Path of the folder containing files
        self.dataset_name = 'surreal'
        self.path = '../../../media/donati/Data1/Datasets/shapes_surreal/'
        self.data_folder = 'off_2/'
        self.spectral_folder = 'spectral_full/'
        self.txt_file = 'surreal5000_training.txt'

        ####################################################
        ####################################################
        ####################################################
        # decide the number of shapes to keep in the training set (exp 2 setting)
        self.split = config.split
        self.num_train = config.num_train  # -1 for all

        # Number of eigenvalues kept for this model fmaps
        self.neig = config.neig
        self.neig_full = config.neig_full

        # Number of thread for input pipeline
        self.num_threads = config.input_threads

    # Utility methods
    # ------------------------------------------------------------------------------------------------------------------

    def get_batch_gen(self, config):
        """
        A function defining the batch generator for each split. Should return the generator, the generated types and
        generated shapes
        :param split: string in "training", "validation" or "test" (here we just keep training)
        :param config: configuration file
        :return: gen_func, gen_types, gen_shapes
        """

        ################
        # Def generators
        ################

        def random_balanced_gen():
            print('trying to generate batch series with ', self.num_train, 'shapes')

            # Initiate concatenation lists
            tp_list = []  # points
            tev_list = []  # eigen vectors
            tevt_list = []  # transposed eigen vectors
            tv_list = []  # eigen values
            tevf_list = []  # full eigen vectors for ground truth maps
            ti_list = []  # cloud indices

            batch_n = 0
            i_batch = 0

            gen_indices = np.random.permutation(int(self.num_train))  # initiate indices for the generator
            # if we had to test on this dataset we would need to introduce a test/val case with non-shuffled indices
            # print(gen_indices.shape, config.batch_num)
            # if config.split == 'test':
            #     print('test setting here not fully supported')
            #     n_shapes = self.num_test  # has to be defined
            #     gen_indices = []
            #     for i in range(n_shapes - 1):
            #         for j in range(i + 1, n_shapes):
            #             gen_indices += [i, j]  # put all the pairs in order
            #     gen_indices = np.array(gen_indices)


            # Generator loop
            for p_i in gen_indices:

                # Get points and other input data
                new_points = self.input_points[p_i]
                new_evecs = self.input_evecs[p_i][:, :self.neig]
                new_evecs_trans = self.input_evecs_trans[p_i][:self.neig, :]
                new_evals = self.input_evals[p_i][:self.neig]

                new_evecs_full = self.input_evecs_full[p_i][:, :self.neig]

                n = new_points.shape[0]

                if i_batch == config.batch_num:

                    yield (np.concatenate(tp_list, axis=0),
                           np.concatenate(tev_list, axis=0),
                           np.concatenate(tevt_list, axis=1),
                           np.concatenate(tv_list, axis=1),
                           np.concatenate(tevf_list, axis=0),
                           np.array(ti_list, dtype=np.int32),
                           np.array([tp.shape[0] for tp in tp_list]))

                    tp_list = []
                    tev_list = []
                    tevt_list = []
                    tv_list = []
                    tevf_list = []
                    ti_list = []

                    batch_n = 0
                    i_batch = 0

                # Add data to current batch
                tp_list += [new_points]
                tev_list += [new_evecs]
                tevt_list += [new_evecs_trans]
                tv_list += [new_evals]
                tevf_list += [new_evecs_full]
                ti_list += [p_i]

                # Update batch size
                batch_n += n
                i_batch += 1

            # yield the rest if necessary (it will not be a full batch and could lead to mistakes because of
            # shape matching needing pairs !!!!)
            yield (np.concatenate(tp_list, axis=0),
                   np.concatenate(tev_list, axis=0),
                   np.concatenate(tevt_list, axis=1),
                   np.concatenate(tv_list, axis=1),
                   np.concatenate(tevf_list, axis=0),
                   np.array(ti_list, dtype=np.int32),
                   np.array([tp.shape[0] for tp in tp_list]))

        ##################
        # Return generator
        ##################

        # Generator types and shapes
        gen_types = (tf.float32, tf.float32, tf.float32, tf.float32, tf.float32, tf.int32, tf.int32)
        gen_shapes = ([None, 3], [None, self.neig],
                      [self.neig, None], [self.neig, None], [None, self.neig], [None], [None])

        return random_balanced_gen, gen_types, gen_shapes

    def get_tf_mapping(self, config):

        def tf_map(stacked_points, stacked_evecs, stacked_evecs_trans,
                   stacked_evals, stacked_evecs_full, obj_inds, stack_lengths):
            """
            From the input point cloud, this function compute all the point clouds at each conv layer, the neighbors
            indices, the pooling indices and other useful variables.
            :param stacked_points: Tensor with size [None, 3] where None is the total number of points
            :param stack_lengths: Tensor with size [None] where None = number of batch // number of points in a batch
            """

            # Get batch indice for each point
            batch_inds = self.tf_get_batch_inds(stack_lengths)

            # Augment input points
            stacked_points, scales, rots = self.tf_augment_input(stacked_points,
                                                                 batch_inds,
                                                                 config)

            # First add a column of 1 as feature for the network to be able to learn 3D shapes
            stacked_features = tf.ones((tf.shape(stacked_points)[0], 1), dtype=tf.float32)

            # Then use positions or not
            if config.in_features_dim == 1:
                pass
            elif config.in_features_dim == 3:
                stacked_features = tf.concat((stacked_features, stacked_points), axis=1)
            else:
                raise ValueError('Only accepted input dimensions are 1, 3 (with or without XYZ)')

            # Get the whole input list
            input_list = self.tf_shape_matching_inputs(config,
                                                       stacked_points,
                                                       stacked_features,
                                                       stack_lengths,
                                                       batch_inds)

            # Add scale and rotation for testing
            input_list += [scales, rots, obj_inds]
            input_list += [stack_lengths]  # in order further on to multiply element-wise in the stack
            input_list += [stacked_evecs, stacked_evecs_trans, stacked_evals]
            input_list += [stacked_evecs_full]

            return input_list

        return tf_map





