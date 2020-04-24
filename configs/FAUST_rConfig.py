# Custom libs
from utils.config import Config

class FAUST_rConfig(Config):
    """
    Override the parameters you want to modify for this dataset
    """

    ####################
    # Dataset parameters
    ####################

    # Dataset name
    dataset = 'FAUST_r'

    # Type of task performed on this dataset (also overwritten)
    network_model = None

    # Number of CPU threads for the input pipeline
    input_threads = 8

    #########################
    # Architecture definition
    #########################

    # Define layers (this can be tuned)
    # ND : for humans, you don't want to put too many layers because then the last will have too few points
    architecture = ['simple',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    'resnetb_strided',
                    'resnetb',
                    #'resnetb_strided',
                    #'resnetb',
                    #'resnetb_strided',
                    #'nearest_upsample',
                    #'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary',
                    'nearest_upsample',
                    'unary'
                    ]

    # KPConv specific parameters
    num_kernel_points = 15
    first_subsampling_dl = 0.03  # No initial subsampling if -1

    # Density of neighborhoods for deformable convs (which need bigger radiuses). For normal conv we use KP_extent
    density_parameter = 1.0

    # Behavior of convolutions in ('constant', 'linear', gaussian)
    KP_influence = 'linear'
    KP_extent = 1.0

    # Aggregation function of KPConv in ('closest', 'sum', could it just be a MAX-POOL ?)
    convolution_mode = 'sum'

    # Choice of input features (1 for constant signal, 3 for coordinates. can be tuned it dataset/common)
    in_features_dim = 3
    first_features_dim = 128

    # Functional map parameters
    neig = 70  # number of eigenvectors for the functional map. needs to be lower than that of preprocessing
    neig_full = neig  # number of eigenvectors for spectral ground truth in loss computation
    lam_reg = 1e-3  # laplacian regularization. Higher than that leeds to too much importance in Laplacian commut.

    ####################
    # number of shapes #
    ####################
    split = 'train'  # will be turned to test when needed
    num_train = -1  # -1 for all. It will crash if this is more than the actual number of training shapes

    #####################
    # Training parameters
    #####################

    # epoch management
    max_epoch = 2000
    epoch_steps = None  # Number of steps per epochs (If None, 1 epoch = computing the whole dataset.)
    batch_num = 4  # * 2 because of this source -> target structure

    # Learning rate management
    learning_rate = 1e-3
    momentum = 0.98
    lr_decays = {i: 0.99**(1/80) for i in range(1, max_epoch)}  # can be tuned
    grad_clip_norm = 100.0   # don't touch

    # Batch normalization parameters
    use_batch_norm = True
    batch_norm_momentum = 0.98

    # Number of epoch between each snapshot
    snapshot_gap = 500

    #################
    # Augmentations #
    #################

    augment_rotation = 'vertical'  # only rotation supported by our method (right now y-axis)
    augment_scale_min = 0.9
    augment_scale_max = 1.1

    # Whether to use loss balanced according to object number of points
    batch_averaged_loss = False

    # Do we need to save convergence
    saving = True
    saving_path = None
