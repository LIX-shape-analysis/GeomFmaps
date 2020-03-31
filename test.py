#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Callable script to test any model on any dataset
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#      Nicolas DONATI - 03/2020
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#

# Common libs
import time
import os
import numpy as np

# My libs
from utils.config import Config
from configs.SurrealConfig import SurrealConfig
from configs.FAUST_rConfig import FAUST_rConfig
from configs.SCAPE_rConfig import SCAPE_rConfig
from configs.SHRECConfig import SHRECConfig
from configs.SHREC_rConfig import SHREC_rConfig

from utils.tester import ModelTester
from models.KPCNN_FM_model import KernelPointCNN_FM

# Datasets
from datasets.Surreal import SurrealDataset
from datasets.FAUST_remeshed import FAUST_r_Dataset
from datasets.SCAPE_remeshed import SCAPE_r_Dataset
from datasets.SHREC import SHREC_Dataset
from datasets.SHREC_remeshed import SHREC_r_Dataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utility functions
#       \***********************/
#


def test_caller(path, step_ind):

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Disable warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    test_dataset = 'SHREC_r'  # 'FAUST_r'  # 'SCAPE_r'  # 'SHREC'

    # Load model parameters at train time (but we also need a test config)
    #config_train = Config()
    #config_train.load(path)

    ##################################
    # Change model parameters for test
    ##################################

    # Change parameters for the test here. For example, you can stop augmenting the input data.

    #config.augment_noise = 0.0001
    #config.augment_color = 1.0
    #config.validation_size = 500
    #config.batch_num = 10

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    config = FAUST_rConfig()  # default setting
    dataset = FAUST_r_Dataset(config)
    if test_dataset == 'FAUST_r':
        print('default setting')
        # config = FAUST_rConfig()
        # dataset = FAUST_r_Dataset(config)
    elif test_dataset == 'SCAPE_r':
        config = SCAPE_rConfig()
        dataset = SCAPE_r_Dataset(config)
    elif test_dataset == 'SHREC':
        config = SHRECConfig()
        dataset = SHREC_Dataset(config)
    elif test_dataset == 'SHREC_r':
        config = SHREC_rConfig()
        dataset = SHREC_r_Dataset(config)
    else:
        raise ValueError('dataset not supported')

    config.epoch_steps = 0  # to avoid it should be None
    config.load(path)  # get the exact same parameters for the network (which will be re-loaded afterwards)
    # print(config.neig, config.epoch_steps)

    # re-define some config parameters for testing
    print('rotations at training were set to :', config.augment_rotation)
    config.augment_rotation = 'none'
    config.batch_num = 1
    # config.lam_reg = None  # no need here to force funnel shape so much (can be 1e-4 or None ?)

    config.split = 'test'
    dataset.split = 'test'
    dataset.num_train = -1  # we want all test data
    #config.first_subsampling_dl = 0.48
    #config.neig = 70; config.neig_full = 70 
    dataset.neig = 100; dataset.neig_full = 100
    dataset.load_subsampled_clouds(config.first_subsampling_dl)
    print(config.neig, dataset.neig)
    # Initialize input pipelines to get batch generator
    dataset.init_test_input_pipeline(config)


    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    model = KernelPointCNN_FM(dataset.flat_inputs, dataset.flat_inputs_2, config)

    # Find all snapshot in the chosen training folder
    snap_path = os.path.join(path, 'snapshots')
    snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

    # Find which snapshot to restore
    print('restoring snapshot')
    chosen_step = np.sort(snap_steps)[step_ind]
    chosen_snap = os.path.join(path, 'snapshots', 'snap-{:d}'.format(chosen_step))
    print(chosen_snap)

    # Create a tester class
    print('Model tester')
    model.fld_name = dataset.dataset_name
    tester = ModelTester(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ############
    # Start test
    ############

    print('Start Test')
    print('**********\n')

    tester.test_shape_matching(model, dataset)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#


if __name__ == '__main__':

    ##########################
    # Choose the model to test
    ##########################

    #chosen_log = 'results/Log_2020-03-16_16-01-48'  # Log_2020-03-12_16-23-10'  # 'Log_2020-02-13_16-09-59'  # surreal100
    
    #chosen_log = 'results/Log_2020-03-04_17-26-11'  # 'Log_2020-02-27_14-55-10'  # surreal5k
    #chosen_log = 'results/Log_2020-03-04_21-01-24'  # s2k
    #chosen_log = 'results/Log_2020-03-10_16-51-30'  # 'Log_2020-03-06_19-04-32'  # s500
    #chosen_log = 'results/Log_2020-03-10_16-51-51'  # 'Log_2020-03-07_13-34-24'  # s100
    #chosen_log = 'results/Log_2020-03-05_08-34-57'  # s5k NoReg
    #chosen_log = 'results/Log_2020-03-08_07-58-35'  #FAUSTr
    #chosen_log = 'results/Log_2020-03-09_08-35-52'  #SCAPEr
    
    chosen_log = 'results/Log_2020-03-28_17-51-04'
    chosen_log = 'results/Log_2020-03-29_20-41-26'
 
    #
    #   You can also choose the index of the snapshot to load (last by default)
    #

    chosen_snapshot = -1

    #
    #   Eventually, you can choose to test your model on the validation set
    #

    on_val = False

    #
    #   If you want to modify certain parameters in the Config class, for example, to stop augmenting the input data,
    #   there is a section for it in the function "test_caller" defined above.
    #

    # Check if log exists
    if not os.path.exists(chosen_log):
        raise ValueError('The given log does not exists: ' + chosen_log)

    # Let's go
    test_caller(chosen_log, chosen_snapshot)



