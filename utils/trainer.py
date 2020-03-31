#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the training of any model
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Hugues THOMAS - 11/06/2018
#


# ----------------------------------------------------------------------------------------------------------------------
#
#           Imports and global variables
#       \**********************************/
#


# Basic libs
import tensorflow as tf
import numpy as np
import pickle
import os
from os import makedirs, remove
from os.path import exists, join
import time
import psutil
import sys

# PLY reader
from utils.ply import read_ply, write_ply
import matplotlib.pyplot as plt

# Metrics
from utils.metrics import IoU_from_confusions
from sklearn.metrics import confusion_matrix

# ----------------------------------------------------------------------------------------------------------------------
#
#           Trainer Class
#       \*******************/
#


class ModelTrainer:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

        # Add training ops
        self.add_train_ops(model)

        # Tensorflow Saver definition
        my_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='KernelPointNetwork')
        self.saver = tf.train.Saver(my_vars, max_to_keep=100)

        # Create a session for running Ops on the Graph.
        on_CPU = False
        if on_CPU:
            cProto = tf.ConfigProto(device_count={'GPU': 0})
        else:
            cProto = tf.ConfigProto()
            cProto.gpu_options.allow_growth = True
        self.sess = tf.Session(config=cProto)

        # Init variables
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if (restore_snap is not None):
            exclude_vars = ['softmax', 'head_unary_conv', '/fc/']
            restore_vars = my_vars
            for exclude_var in exclude_vars:
                restore_vars = [v for v in restore_vars if exclude_var not in v.name]
            restorer = tf.train.Saver(restore_vars)
            restorer.restore(self.sess, restore_snap)
            print("Model restored.")

    def add_train_ops(self, model):
        """
        Add training ops on top of the model
        """

        ##############
        # Training ops
        ##############

        with tf.variable_scope('optimizer'):

            # Learning rate as a Variable so we can modify it
            self.learning_rate = tf.Variable(model.config.learning_rate, trainable=False, name='learning_rate')

            # Create the gradient descent optimizer with the given learning rate.
            print('given momentum for learning rate decay : ', model.config.momentum)
            optimizer = tf.train.MomentumOptimizer(self.learning_rate, model.config.momentum)

            # Training step op
            gvs = optimizer.compute_gradients(model.loss)

            if model.config.grad_clip_norm > 0:

                # Get gradient for deformable convolutions and scale them
                scaled_gvs = []
                for grad, var in gvs:
                    if 'offset_conv' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    if 'offset_mlp' in var.name:
                        scaled_gvs.append((0.1 * grad, var))
                    else:
                        scaled_gvs.append((grad, var))
                print('stacking gradients')

                # Clipping each gradient independantly
                # capped_gvs = [(tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for grad, var in scaled_gvs]
                capped_gvs = [(grad if grad is None else tf.clip_by_norm(grad, model.config.grad_clip_norm), var) for
                              grad, var in scaled_gvs]

                # Clipping the whole network gradient (problematic with big network where grad == inf)
                # capped_grads, global_norm = tf.clip_by_global_norm([grad for grad, var in gvs], self.config.grad_clip_norm)
                # vars = [var for grad, var in gvs]
                # capped_gvs = [(grad, var) for grad, var in zip(capped_grads, vars)]

                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(capped_gvs)

            else:
                extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
                with tf.control_dependencies(extra_update_ops):
                    self.train_op = optimizer.apply_gradients(gvs)

        return

    # Training main method
    # ------------------------------------------------------------------------------------------------------------------

    def train(self, model, dataset, debug_NaN=False):
        """
        Train the model on a particular dataset.
        """

        if debug_NaN:
            # Add checking ops
            self.check_op = tf.add_check_numerics_ops()

        # Parameters log file
        if model.config.saving:
            model.parameters_log()

        # Save points of the kernel to file
        self.save_kernel_points(model, 0)

        if model.config.saving:
            # Training log file
            with open(join(model.saving_path, 'training.txt'), "w") as file:
                file.write('Steps out_loss reg_loss point_loss train_accuracy time memory\n')

            # Killing file (simply delete this file when you want to stop the training)
            if not exists(join(model.saving_path, 'running_PID.txt')):
                with open(join(model.saving_path, 'running_PID.txt'), "w") as file:
                    file.write('Launched with PyCharm')

        # Train loop variables
        t0 = time.time()
        self.training_step = 0
        self.training_epoch = 0
        mean_dt = np.zeros(2)
        last_display = t0
        epoch_n = 1
        mean_epoch_n = 0

        # Initialise iterator with train data
        self.sess.run(dataset.train_init_op)

        # Start loop
        while self.training_epoch < model.config.max_epoch:

            try:
                # Run one step of the model.
                t = [time.time()]
                ops = [self.train_op,
                       model.output_loss,
                       model.regularization_loss,
                       model.offsets_loss,
                       model.fmaps,
                       model.desc,
                       model.desc_2,
                       #model.output_features,
                       #model.output_features_2,
                       model.inputs['batch_inds'],
                       model.inputs_2['batch_inds'],
                       model.inputs['points']]

                # If NaN appears in a training, use this debug block
                if debug_NaN and False:
                    all_values = self.sess.run(ops + [self.check_op] + list(dataset.flat_inputs), {model.dropout_prob: 0.5})
                    L_out, L_reg, L_p, probs, labels, acc = all_values[1:7]
                    if np.isnan(L_reg) or np.isnan(L_out):
                        input_values = all_values[8:]
                        self.debug_nan(model, input_values, probs)
                        a = 1/0

                else:
                    _, L_out, L_reg, L_p, probs, desc, desc_2, bi, bi2, pts = self.sess.run(ops, {model.dropout_prob: 0.5})

                t += [time.time()]

                ### saving map images ################ (for debugging or silply checking network Fmap)
                fig_C = plt.figure()
                ax_C = fig_C.add_subplot(111)
                im_C = ax_C.imshow(probs[0])
                fig_C.colorbar(im_C, orientation='horizontal', shrink=0.8)
                ax_C.set_title('functionnal map')
                images_directory = 'results/images/'
                if not exists(images_directory):
                    makedirs(images_directory)
                fig_C.savefig(images_directory + 'maps' + '_it=' + str(self.training_epoch) + '_funmap.png')
                plt.close('all')
                ###############################

                ######### saving maps and descs and batch_indices for quick eval #######
                if False and self.training_epoch % 100 == 99:  # only necessary when debugging
                    savepath = join(model.saving_path, 'Matches', 'it_' + str(self.training_epoch))
                    if not os.path.isdir(savepath):
                        print('matches_dir=%s' % savepath)
                        os.makedirs(savepath)
                    np.save(join(savepath, 'fmaps.npy'), probs)
                    np.save(join(savepath, 'desc1.npy'), desc)
                    np.save(join(savepath, 'desc2.npy'), desc_2)
                    #np.save(join(savepath, 'of1.npy'), of)
                    #np.save(join(savepath, 'of2.npy'), of2)
                    np.save(join(savepath, 'bi1.npy'), bi)
                    np.save(join(savepath, 'bi2.npy'), bi2)
                    np.save(join(savepath, 'pts.npy'), pts)

                # Stack prediction for training confusion
                if model.config.network_model == 'classification':
                    self.training_preds = np.hstack((self.training_preds, np.argmax(probs, axis=1)))
                    self.training_labels = np.hstack((self.training_labels, labels))
                t += [time.time()]

                # Average timing
                mean_dt = 0.95 * mean_dt + 0.05 * (np.array(t[1:]) - np.array(t[:-1]))

                # Console display (only one per second)
                if (t[-1] - last_display) > 1.0:
                    last_display = t[-1]
                    #message = 'Step {:08d} L_out={:5.3f} L_reg={:5.3f} L_p={:5.3f} Acc={:4.2f} ' \
                    message = 'Step {:08d} L_out={:5.3f}'  # L_reg={:5.3f} ' \
                              #'batchsize={:02d} ---{:8.2f} ms/batch (Averaged)'
                    print(message.format(self.training_step,
                                         L_out))
                                         #L_reg,
                                         #L_p,
                                         #acc,
                                         #batch_size,
                                         #1000 * mean_dt[0],
                                         #1000 * mean_dt[1]))

                # Log file (txt inside results folder)
                if model.config.saving:
                    process = psutil.Process(os.getpid())
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        #message = '{:d} {:.3f} {:.3f} {:.3f} {:.2f} {:.2f} {:.1f}\n'
                        message = '{:d} {:.3f} {:.3f} {:.3f} {:.2f} {:.1f}\n'  #TODO: clean log message
                        file.write(message.format(self.training_step,
                                                  L_out,
                                                  L_reg,
                                                  L_p,
                                                  t[-1] - t0,
                                                  process.memory_info().rss * 1e-6))

                # Check kill signal (running_PID.txt deleted)
                if model.config.saving and not exists(join(model.saving_path, 'running_PID.txt')):
                    break

                #if model.config.dataset.startswith('ShapeNetPart') or model.config.dataset.startswith('ModelNet'):
                if model.config.epoch_steps and epoch_n > model.config.epoch_steps:
                    raise tf.errors.OutOfRangeError(None, None, '')

            except tf.errors.OutOfRangeError:

                if model.config.saving:
                    with open(join(model.saving_path, 'training.txt'), "a") as file:
                        file.write('NEW EPOCH')
                print('NEW EPOCH : ', self.training_epoch, 'learning rate :', self.learning_rate.eval(session=self.sess))

                # End of train dataset, update average of epoch steps
                mean_epoch_n += (epoch_n - mean_epoch_n) / (self.training_epoch + 1)
                epoch_n = 0
                self.int = int(np.floor(mean_epoch_n))
                model.config.epoch_steps = int(np.floor(mean_epoch_n))
                if model.config.saving:
                    model.parameters_log()

                # Snapshot
                if model.config.saving and (self.training_epoch + 1) % model.config.snapshot_gap == 0:

                    # Tensorflow snapshot
                    snapshot_directory = join(model.saving_path, 'snapshots')
                    if not exists(snapshot_directory):
                        makedirs(snapshot_directory)
                    self.saver.save(self.sess, snapshot_directory + '/snap', global_step=self.training_step + 1)

                    # Save points
                    self.save_kernel_points(model, self.training_epoch)

                # Update learning rate
                if self.training_epoch in model.config.lr_decays:
                    op = self.learning_rate.assign(tf.multiply(self.learning_rate,
                                                               model.config.lr_decays[self.training_epoch]))
                    self.sess.run(op)

                # Increment
                self.training_epoch += 1

                # Validation
                if (self.training_epoch % 100 == 20) and False:  # not supported at the moment
                    print('validation step')
                    self.shape_matching_validation_error(model, dataset)

                # Reset iterator on training data
                self.sess.run(dataset.train_init_op)

            except tf.errors.InvalidArgumentError as e:

                print('Caught a NaN error :')
                print(e.error_code)
                print(e.message)
                print(e.op)
                print(e.op.name)
                print([t.name for t in e.op.inputs])
                print([t.name for t in e.op.outputs])

                a = 1/0  # to provoke error if the error did not make the code crash

            # Increment steps
            self.training_step += 1
            epoch_n += 1

        # Remove File for kill signal
        if exists(join(model.saving_path, 'running_PID.txt')):
            remove(join(model.saving_path, 'running_PID.txt'))
        self.sess.close()

    # Validation methods
    # ------------------------------------------------------------------------------------------------------------------

    def shape_matching_validation_error(self, model, dataset):  # not supported .. For now only saves some training data
        """
        Validation method for single object segmentation models
        """
        ##########
        # Initiate
        ##########

        # Choose validation smoothing parameter (0 for no smothing, 0.99 for big smoothing)
        val_smooth = 0.95

        # Initialise iterator with train data
        self.sess.run(dataset.val_init_op)

        #####################
        # Network predictions
        #####################

        maps = []
        desc1 = []
        desc2 = []
        bi1 = []
        bi2 = []

        mean_dt = np.zeros(2)
        last_display = time.time()

        for i0 in range(model.config.validation_size):
            try:
                # Run one step of the model.
                t = [time.time()]
                ops = (model.fmaps,
                       model.desc,
                       model.desc_2,
                       model.inputs['batch_inds'],
                       model.inputs_2['batch_inds'])

                C, C_t, A, B, b1, b2 = self.sess.run(ops, {model.dropout_prob: 0.5})
                t += [time.time()]

                # Save validation data
                # ***************************************

                savepath = join(model.saving_path, 'val_data', 'Matches', 'it_' + str(self.training_epoch))

                if not os.path.isdir(savepath):
                    print('matches_dir=%s' % savepath)
                    os.makedirs(savepath)

                maps += [C]
                desc1 += [A]
                desc2 += [B]
                bi1 += [b1]
                bi2 += [b2]

            except tf.errors.OutOfRangeError:
                break

            np.save(join(savepath, 'fmaps.npy'), maps)
            np.save(join(savepath, 'desc1.npy'), desc1)
            np.save(join(savepath, 'desc2.npy'), desc2)
            np.save(join(savepath, 'bi1.npy'), bi1)
            np.save(join(savepath, 'bi2.npy'), bi2)
        return


    # Saving methods
    # ------------------------------------------------------------------------------------------------------------------

    def save_kernel_points(self, model, epoch):
        """
        Method saving kernel point disposition and current model weights for later visualization
        """

        if model.config.saving:

            # Create a directory to save kernels of this epoch
            kernels_dir = join(model.saving_path, 'kernel_points', 'epoch{:d}'.format(epoch))
            if not exists(kernels_dir):
                makedirs(kernels_dir)

            # Get points
            all_kernel_points_tf = [v for v in tf.global_variables() if 'kernel_points' in v.name
                                    and v.name.startswith('KernelPoint')]
            all_kernel_points = self.sess.run(all_kernel_points_tf)

            # Get Extents
            if False and 'gaussian' in model.config.convolution_mode:
                all_kernel_params_tf = [v for v in tf.global_variables() if 'kernel_extents' in v.name
                                        and v.name.startswith('KernelPoint')]
                all_kernel_params = self.sess.run(all_kernel_params_tf)
            else:
                all_kernel_params = [None for p in all_kernel_points]

            # Save in ply file
            for kernel_points, kernel_extents, v in zip(all_kernel_points, all_kernel_params, all_kernel_points_tf):

                # Name of saving file
                ply_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.ply'
                ply_file = join(kernels_dir, ply_name)

                # Data to save
                if kernel_points.ndim > 2:
                    kernel_points = kernel_points[:, 0, :]
                if False and 'gaussian' in model.config.convolution_mode:
                    data = [kernel_points, kernel_extents]
                    keys = ['x', 'y', 'z', 'sigma']
                else:
                    data = kernel_points
                    keys = ['x', 'y', 'z']

                # Save
                write_ply(ply_file, data, keys)

            # Get Weights
            all_kernel_weights_tf = [v for v in tf.global_variables() if 'weights' in v.name
                                    and v.name.startswith('KernelPointNetwork')]
            all_kernel_weights = self.sess.run(all_kernel_weights_tf)

            # Save in numpy file
            for kernel_weights, v in zip(all_kernel_weights, all_kernel_weights_tf):
                np_name = '_'.join(v.name[:-2].split('/')[1:-1]) + '.npy'
                np_file = join(kernels_dir, np_name)
                np.save(np_file, kernel_weights)

    # Debug methods  ---  /!\ /!\ /!\ not updated for shape Matching algo
    # ------------------------------------------------------------------------------------------------------------------

    def show_memory_usage(self, batch_to_feed):    # not updated for shape Matching algo

            for l in range(self.config.num_layers):
                neighb_size = list(batch_to_feed[self.in_neighbors_f32[l]].shape)
                dist_size = neighb_size + [self.config.num_kernel_points, 3]
                dist_memory = np.prod(dist_size) * 4 * 1e-9
                in_feature_size = neighb_size + [self.config.first_features_dim * 2**l]
                in_feature_memory = np.prod(in_feature_size) * 4 * 1e-9
                out_feature_size = [neighb_size[0], self.config.num_kernel_points, self.config.first_features_dim * 2**(l+1)]
                out_feature_memory = np.prod(out_feature_size) * 4 * 1e-9

                print('Layer {:d} => {:.1f}GB {:.1f}GB {:.1f}GB'.format(l,
                                                                   dist_memory,
                                                                   in_feature_memory,
                                                                   out_feature_memory))
            print('************************************')

    def debug_nan(self, model, inputs, logits):  # not updated for shape Matching algo
        """
        NaN happened, find where
        """

        print('\n\n------------------------ NaN DEBUG ------------------------\n')

        # First save everything to reproduce error
        file1 = join(model.config.saving_path, 'all_debug_inputs.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(inputs, f1)

        # First save all inputs
        file1 = join(model.config.saving_path, 'all_debug_logits.pkl')
        with open(file1, 'wb') as f1:
            pickle.dump(logits, f1)

        # Then print a list of the trainable variables and if they have nan
        print('List of variables :')
        print('*******************\n')
        all_vars = self.sess.run(tf.global_variables())
        for v, value in zip(tf.global_variables(), all_vars):
            nan_percentage = 100 * np.sum(np.isnan(value)) / np.prod(value.shape)
            print(v.name, ' => {:.1f}% of values are NaN'.format(nan_percentage))


        print('Inputs :')
        print('********')

        #Print inputs
        nl = model.config.num_layers
        for layer in range(nl):

            print('Layer : {:d}'.format(layer))

            points = inputs[layer]
            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            nan_percentage = 100 * np.sum(np.isnan(points)) / np.prod(points.shape)
            print('Points =>', points.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(neighbors)) / np.prod(neighbors.shape)
            print('neighbors =>', neighbors.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(pools)) / np.prod(pools.shape)
            print('pools =>', pools.shape, '{:.1f}% NaN'.format(nan_percentage))
            nan_percentage = 100 * np.sum(np.isnan(upsamples)) / np.prod(upsamples.shape)
            print('upsamples =>', upsamples.shape, '{:.1f}% NaN'.format(nan_percentage))

        ind = 4 * nl
        features = inputs[ind]
        nan_percentage = 100 * np.sum(np.isnan(features)) / np.prod(features.shape)
        print('features =>', features.shape, '{:.1f}% NaN'.format(nan_percentage))
        ind += 1
        batch_weights = inputs[ind]
        ind += 1
        in_batches = inputs[ind]
        max_b = np.max(in_batches)
        print(in_batches.shape)
        in_b_sizes = np.sum(in_batches < max_b - 0.5, axis=-1)
        print('in_batch_sizes =>', in_b_sizes)
        ind += 1
        out_batches = inputs[ind]
        max_b = np.max(out_batches)
        print(out_batches.shape)
        out_b_sizes = np.sum(out_batches < max_b - 0.5, axis=-1)
        print('out_batch_sizes =>', out_b_sizes)
        ind += 1
        point_labels = inputs[ind]
        ind += 1
        if model.config.dataset.startswith('ShapeNetPart_multi'):
            object_labels = inputs[ind]
            nan_percentage = 100 * np.sum(np.isnan(object_labels)) / np.prod(object_labels.shape)
            print('object_labels =>', object_labels.shape, '{:.1f}% NaN'.format(nan_percentage))
            ind += 1
        augment_scales = inputs[ind]
        ind += 1
        augment_rotations = inputs[ind]
        ind += 1

        print('\npoolings and upsamples nums :\n')

        #Print inputs
        for layer in range(nl):

            print('\nLayer : {:d}'.format(layer))

            neighbors = inputs[nl + layer]
            pools = inputs[2*nl + layer]
            upsamples = inputs[3*nl + layer]

            max_n = np.max(neighbors)
            nums = np.sum(neighbors < max_n - 0.5, axis=-1)
            print('min neighbors =>', np.min(nums))

            max_n = np.max(pools)
            nums = np.sum(pools < max_n - 0.5, axis=-1)
            print('min pools =>', np.min(nums))

            max_n = np.max(upsamples)
            nums = np.sum(upsamples < max_n - 0.5, axis=-1)
            print('min upsamples =>', np.min(nums))


        print('\nFinished\n\n')
        time.sleep(0.5)



































