# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#      Nicolas DONATI - 01/01/2020


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Common libs
import time
import os
import numpy as np

# configs
from configs.SurrealConfig import SurrealConfig
from configs.FAUST_rConfig import FAUST_rConfig
from configs.SCAPE_rConfig import SCAPE_rConfig
from configs.SHRECConfig import SHRECConfig

# Custom libs
from utils.config import Config
from utils.trainer import ModelTrainer
from models.KPCNN_FM_model import KernelPointCNN_FM

# Dataset
from datasets.Surreal import SurrealDataset
from datasets.FAUST_remeshed import FAUST_r_Dataset
from datasets.SCAPE_remeshed import SCAPE_r_Dataset
from datasets.SHREC import SHREC_Dataset


# ----------------------------------------------------------------------------------------------------------------------
#
#           Main Call
#       \***************/
#

if __name__ == '__main__':

    ##########################
    # Initiate the environment
    ##########################

    # Choose which gpu to use
    GPU_ID = '0'
    #GPU_ID = '2'

    # Set GPU visible device
    os.environ['CUDA_VISIBLE_DEVICES'] = GPU_ID

    # Enable/Disable warnings (set level to '0'/'3')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'

    ###########################
    # Load the model parameters
    ###########################

    training_dataset = 'surreal'  # 'SCAPE_r'  #'FAUST_r'  #'surreal'

    ##############
    # Prepare Data
    ##############

    print()
    print('Dataset Preparation')
    print('*******************')

    # Initiate dataset configuration
    config = SurrealConfig()  # default configuration
    dataset = SurrealDataset(config)
    if training_dataset == 'surreal':
        print('default setting')
        # config = SurrealConfig()
        # dataset = SurrealDataset(config)
    elif training_dataset == 'FAUST_r':
        config = FAUST_rConfig()
        dataset = FAUST_r_Dataset(config)
    elif training_dataset == 'SCAPE_r':
        config = SCAPE_rConfig()
        dataset = SCAPE_r_Dataset(config)
    else:
        raise ValueError('dataset not supported')

    # Create subsample clouds of the models
    dl0 = config.first_subsampling_dl
    dataset.load_subsampled_clouds(dl0)  # No initial subsampling if -1
    #config.first_subsampling_dl = config.second_subsampling_dl
    # Initialize input pipelines
    dataset.init_input_pipeline(config)

    # Test the input pipeline alone with this debug function
    #dataset.check_input_pipeline_timing(config)

    ##############
    # Define Model
    ##############

    print('Creating Model')
    print('**************\n')
    t1 = time.time()

    # Model class
    #model = KernelPointFCNN(dataset.flat_inputs, config)
    config.path = dataset.path  # set the config path to get spectral data
    model = KernelPointCNN_FM(dataset.flat_inputs, dataset.flat_inputs_2, config)

    # Choose here if you want to start training from a previous snapshot
    previous_training_path = None
    step_ind = -1

    if previous_training_path:

        # Find all snapshot in the chosen training folder
        snap_path = os.path.join(previous_training_path, 'snapshots')
        snap_steps = [int(f[:-5].split('-')[-1]) for f in os.listdir(snap_path) if f[-5:] == '.meta']

        # Find which snapshot to restore
        chosen_step = np.sort(snap_steps)[step_ind]
        chosen_snap = os.path.join(previous_training_path, 'snapshots', 'snap-{:d}'.format(chosen_step))

    else:
        chosen_snap = None

    # Create a trainer class
    trainer = ModelTrainer(model, restore_snap=chosen_snap)
    t2 = time.time()

    print('\n----------------')
    print('Done in {:.1f} s'.format(t2 - t1))
    print('----------------\n')

    ################
    # Start training
    ################

    print('Start Training')
    print('**************\n')

    trainer.train(model, dataset)





