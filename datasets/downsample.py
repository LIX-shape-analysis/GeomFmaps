# Basic libs
import pickle

# PLY and OFF reader and MAT
import scipy.io as sio
from utils.off import *

# OS functions
from os.path import exists, join

# Subsampling extension
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling


def downsample_data(dataset, subsampling_parameter):
    """
    Subsample point clouds and load into memory
    """

    if 0 < subsampling_parameter <= 0.01:
        raise ValueError('subsampling_parameter too low (should be over 1 cm')

    #################
    # Getting files #
    #################

    # Restart timer
    t0 = time.time()

    # Load wanted points if possible
    print('\nLoading points')

    pic = False  # to know if there has been a load with pickle
    big = False  # to know whether the dataset is too big to handle
    filename = ''
    filename1 = ''
    filename2 = ''

    n_data = dataset.num_data_kept
    if n_data > 3000:
        big = True
        print('BIG dataset ; splitting ...')
        filename1 = join(dataset.path, dataset.path_pickle,
                         dataset.split + '_{:.3f}_record_part1.pkl'.format(subsampling_parameter))
        filename2 = join(dataset.path, dataset.path_pickle,
                         dataset.split + '_{:.3f}_record_part2.pkl'.format(subsampling_parameter))

        if exists(filename1):
            with open(filename1, 'rb') as file:
                print('from pickle')
                dataset.input_points, \
                dataset.input_evecs, \
                dataset.input_evecs_trans, \
                dataset.input_evals, \
                dataset.input_evecs_full = pickle.load(file)

        if exists(filename2):
            with open(filename2, 'rb') as file:
                pts, evs, evts, evas, evfs = pickle.load(file)
                dataset.input_points += pts
                dataset.input_evecs += evs
                dataset.input_evecs_trans += evts
                dataset.input_evals += evas
                dataset.input_evecs_full += evfs
                print(dataset.input_evecs[0].shape)
                print(len(dataset.input_points), '/', dataset.num_data, 'shapes have been loaded')
                pic = True  # pickle has been loaded

    else:
        filename = join(dataset.path, dataset.path_pickle,
                        dataset.split + '_{:.3f}_record.pkl'.format(subsampling_parameter))

        if exists(filename):
            with open(filename, 'rb') as file:
                print('from pickle')
                dataset.input_points, \
                dataset.input_evecs, \
                dataset.input_evecs_trans, \
                dataset.input_evals, \
                dataset.input_evecs_full = pickle.load(file)
                print(len(dataset.input_points), '/', dataset.num_data, 'shapes have been loaded')
                pic = True  # pickle has been loaded

    # Else compute them from original points
    if not pic:

        kept_indices = range(dataset.num_data_kept)
        if dataset.num_data_kept != dataset.num_data:  # keeping only part of the data
            print('we keep only part of the data here')
            kept_indices = np.arange(int(dataset.num_data_kept))
            filename_indices = join(dataset.path, dataset.path_pickle, 'indices.txt')
            np.savetxt(filename_indices, kept_indices, fmt='%i')

        # Collect training file names
        names = np.loadtxt(join(dataset.path, dataset.txt_file), dtype=np.str)[kept_indices]

        # Collect point clouds
        for i, cloud_name in enumerate(names):

            if i % 100 == 0:
                print(i, cloud_name)

            # Read points
            txt_file = join(dataset.path, dataset.data_folder, cloud_name)
            # distinguish between off and pts
            if cloud_name[-3:] == 'pts':
                if i == 0: print('dealing with pts files')
                data = np.loadtxt(txt_file, delimiter=' ', dtype=np.float32)
                #data = data - np.mean(data, 0)  # centering the points ?
                if i % 100 == 0: print(np.mean(data, 0))
            elif cloud_name[-3:] == 'off':
                if i == 0: print('dealing with off files')
                data, _ = readOFF(txt_file)
                data = data.astype(np.float32)
            
            # vts files in case of FAUST and SCAPE remesh
            if dataset.dataset_name == 'FAUST_r' or dataset.dataset_name == 'SCAPE_r':
                print('in a FAUST/SCAPE vts setting ! ! ! !')
                vts_folder = '../../../media/donati/Data1/Datasets/'+ dataset.dataset_name +'/corres/'
                vts = np.loadtxt(vts_folder+cloud_name[:-4]+'.vts', dtype = np.int32) - 1

            # Read spectral data
            txt_file = dataset.path + dataset.spectral_folder + cloud_name[:-4] + '.mat'  # name of spectral container
            spc_data = sio.loadmat(txt_file)#, verify_compressed_data_integrity=False)

            evecs_full = spc_data['target_evecs'][:, :dataset.neig_full]
            evecs = evecs_full[:, :dataset.neig]
            evecs_trans = spc_data['target_evecs_trans'][:dataset.neig, :]
            evals = spc_data['target_evals'][:dataset.neig]
            concat_features = np.concatenate([evecs, evecs_trans.T], axis=1).astype(np.float32)

            # Subsample them
            if subsampling_parameter > 0:
                points, concat_features = grid_subsampling(data[:, :3],
                                                           features=concat_features,
                                                           sampleDl=subsampling_parameter)
            else:
                points = data[:, :3]

            # Add to list
            dataset.input_points += [points]
            dataset.input_evecs += [concat_features[:, :dataset.neig]]
            dataset.input_evecs_trans += [concat_features[:, dataset.neig:].T]
            dataset.input_evals += [evals]
            
            if dataset.dataset_name == 'FAUST_r' or dataset.dataset_name == 'SCAPE_r':
                dataset.input_evecs_full += [evecs_full[vts]]  # No sampling on this one
            else:
                dataset.input_evecs_full += [evecs_full]

            # Save for later use
            if big:
                if i == dataset.num_data / 2:
                    print('saving part1')
                    # Save for later use the first part
                    with open(filename1, 'wb') as file:
                        pickle.dump((dataset.input_points,
                                     # dataset.input_normals,
                                     dataset.input_evecs,
                                     dataset.input_evecs_trans,
                                     dataset.input_evals,
                                     dataset.input_evecs_full), file)

                    dataset.input_points = []
                    # dataset.input_normals = []
                    dataset.input_evecs = []
                    dataset.input_evecs_trans = []
                    dataset.input_evals = []
                    dataset.input_evecs_full = []

        if big:
            filename = filename2
            print('saving part2')

        with open(filename, 'wb') as file:
            pickle.dump((dataset.input_points,
                         # dataset.input_normals,
                         dataset.input_evecs,
                         dataset.input_evecs_trans,
                         dataset.input_evals,
                         dataset.input_evecs_full), file)
        print('saved !')

    lengths = [p.shape[0] for p in dataset.input_points]
    sizes = [le * 4 * 400 for le in lengths]
    print('{:.1f} MB loaded in {:.1f}s'.format(np.sum(sizes) * 1e-6, time.time() - t0))

    return dataset

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
