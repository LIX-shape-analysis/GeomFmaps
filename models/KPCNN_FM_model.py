#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Segmentation model
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
from os import makedirs
from os.path import exists, join
import time
import tensorflow as tf
import numpy as np
import scipy.io as sio
import sys

# Convolution functions
from models.network_blocks import assemble_CNN_FM_blocks, shape_matching_head, shape_matching_head_2

#from losses import penalty_bijectivity, penalty_ortho, penalty_desc_orientation,\
#    penalty_laplacian_commutativity, penalty_desc_commutativity
from losses.supervised_penalty import *


# ----------------------------------------------------------------------------------------------------------------------
#
#           Model Class
#       \*****************/
#


class KernelPointCNN_FM:

    def __init__(self, flat_inputs, flat_inputs_2, config):
        """
        Initiate the model
        :param flat_inputs: List of input tensors (flatten)
        :param config: configuration class
        """

        # Model parameters
        self.config = config

        # Path of the result folder
        if self.config.saving:
            if self.config.saving_path == None:
                self.saving_path = time.strftime('results/Log_%Y-%m-%d_%H-%M-%S', time.gmtime())
            else:
                self.saving_path = self.config.saving_path
            if not exists(self.saving_path):
                makedirs(self.saving_path)

        ########
        # Inputs
        ########

        # Sort flatten inputs in a dictionary
        with tf.variable_scope('inputs'):
            self.inputs = dict()
            self.inputs_2 = dict()

            ########
            # Source
            ########

            self.inputs['points'] = flat_inputs[:config.num_layers]
            self.inputs['neighbors'] = flat_inputs[config.num_layers:2 * config.num_layers]
            self.inputs['pools'] = flat_inputs[2 * config.num_layers:3 * config.num_layers]
            self.inputs['upsamples'] = flat_inputs[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs['features'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_weights'] = flat_inputs[ind]
            ind += 1
            self.inputs['in_batches'] = flat_inputs[ind]
            ind += 1
            self.inputs['out_batches'] = flat_inputs[ind]
            ind += 1

            self.inputs['augment_scales'] = flat_inputs[ind]
            ind += 1
            self.inputs['augment_rotations'] = flat_inputs[ind]
            ind += 1
            self.inputs['batch_inds'] = flat_inputs[ind]
            ind += 1
            self.inputs['stack_lengths'] = flat_inputs[ind]
            ind += 1
            self.inputs['evecs'] = flat_inputs[ind]  # Spectral
            ind += 1
            self.inputs['evecs_trans'] = flat_inputs[ind]
            ind += 1
            self.inputs['evals'] = flat_inputs[ind]
            ind += 1
            self.inputs['evecs_full'] = flat_inputs[ind]

            ########
            # Target
            ########

            self.inputs_2['points'] = flat_inputs_2[:config.num_layers]
            self.inputs_2['neighbors'] = flat_inputs_2[config.num_layers:2 * config.num_layers]
            self.inputs_2['pools'] = flat_inputs_2[2 * config.num_layers:3 * config.num_layers]
            self.inputs_2['upsamples'] = flat_inputs_2[3 * config.num_layers:4 * config.num_layers]
            ind = 4 * config.num_layers
            self.inputs_2['features'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['batch_weights'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['in_batches'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['out_batches'] = flat_inputs_2[ind]
            ind += 1

            self.inputs_2['augment_scales'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['augment_rotations'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['batch_inds'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['stack_lengths'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['evecs'] = flat_inputs_2[ind]  # Spectral
            ind += 1
            self.inputs_2['evecs_trans'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['evals'] = flat_inputs_2[ind]
            ind += 1
            self.inputs_2['evecs_full'] = flat_inputs_2[ind]

            # Dropout placeholder
            self.dropout_prob = tf.placeholder(tf.float32, name='dropout_prob')

            ##########

        ########
        # Layers
        ########

        # Create layers
        with tf.variable_scope('KernelPointNetwork'):
            # get the descriptors
            self.output_features, self.output_features_2 = assemble_CNN_FM_blocks(self.inputs,
                                                                                  self.inputs_2,
                                                                                  self.config,
                                                                                  self.dropout_prob)

            # prepare output for loss computation (pass into spectral)
            self.desc = features_to_spectral(self.output_features, self.inputs['evecs_trans'],
                                             self.inputs['stack_lengths'], config)
            self.desc_2 = features_to_spectral(self.output_features_2, self.inputs_2['evecs_trans'],
                                               self.inputs_2['stack_lengths'], config)

            # get the fmaps in a 3d Tensor

            if config.lam_reg == None:
                self.fmaps = shape_matching_head(self.desc, self.desc_2, config)
                self.fmaps_t = shape_matching_head(self.desc_2, self.desc, config)
            else:
                self.fmaps = shape_matching_head_2(self.desc, self.desc_2, self.inputs['evals'], self.inputs_2['evals'], config)
                self.fmaps_t = shape_matching_head_2(self.desc_2, self.desc, self.inputs_2['evals'], self.inputs['evals'], config)

        ########
        # Losses
        ########

        with tf.variable_scope('loss'):

            # self.output_loss = shape_matching_loss(output_features,
            #                                        output_features_2,
            #                                        spectral,
            #                                        spectral_2)

            # Add regularization
            #self.n_batch = tf.shape(self.inputs['stack_lengths'])[0]  # isn't that fixed ? it is
            self.n_batch = self.config.batch_num
            print("n_batch", self.inputs['stack_lengths'])
            # Or_op1 = tf.reshape(self.inputs['orOps'], [self.n_batch, config.neig, config.neig])
            # Or_op2 = tf.reshape(self.inputs_2['orOps'], [self.n_batch, config.neig, config.neig])

            # self.output_loss = global_fmap_loss(self.fmaps, self.fmaps_t,
            #                                     # Or_op1, Or_op2,
            #                                     self.output_features,
            #                                     self.inputs['evecs'],
            #                                     self.inputs['evecs_trans'],
            #                                     self.inputs['evals'],
            #                                     self.output_features_2,
            #                                     self.inputs_2['evecs'],
            #                                     self.inputs_2['evecs_trans'],
            #                                     self.inputs_2['evals'],
            #                                     self.n_batch)

            self.output_loss = global_fmap_loss_sup(self.fmaps,
                                                self.inputs['evecs_full'],
                                                self.inputs_2['evecs_full'],
                                                self.inputs['stack_lengths'],
                                                self.inputs_2['stack_lengths'],
                                                self.inputs['batch_inds'],
                                                self.inputs_2['batch_inds'],                                                
                                                self.n_batch)

            self.loss = self.output_loss  #+ self.regularization_losses()  # + self.output_loss
            self.reg_loss = self.regularization_losses()
        return

    def regularization_losses(self):

        #####################
        # Regularization loss
        #####################

        # Get L2 norm of all weights
        regularization_losses = [tf.nn.l2_loss(v) for v in tf.global_variables() if 'weights' in v.name]
        self.regularization_loss = self.config.weights_decay * tf.add_n(regularization_losses)

        ##############################
        # Gaussian regularization loss
        ##############################

        gaussian_losses = []
        for v in tf.global_variables():
            if 'kernel_extents' in v.name:

                # Layer index
                layer = int(v.name.split('/')[1].split('_')[-1])

                # Radius of convolution for this layer
                conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** (layer - 1))

                # Target extent
                target_extent = conv_radius / 1.5
                gaussian_losses += [tf.nn.l2_loss(v - target_extent)]

        if len(gaussian_losses) > 0:
            self.gaussian_loss = self.config.gaussian_decay * tf.add_n(gaussian_losses)
        else:
            self.gaussian_loss = tf.constant(0, dtype=tf.float32)

        #############################
        # Offsets regularization loss
        #############################

        offset_losses = []

        if self.config.offsets_loss == 'permissive':

            for op in tf.get_default_graph().get_operations():
                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_positions = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    conv_radius = self.config.first_subsampling_dl * self.config.density_parameter * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_positions/conv_radius

                    # Loss will be zeros inside radius and linear outside radius
                    # Mean => loss independent from the number of input points
                    radius_outside = tf.maximum(0.0, tf.norm(KP_locs, axis=2) - 1.0)
                    offset_losses += [tf.reduce_mean(radius_outside)]


        elif self.config.offsets_loss == 'fitting':

            for op in tf.get_default_graph().get_operations():

                if op.name.endswith('deformed_d2'):

                    # Get deformed distances
                    deformed_d2 = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Get the distance to closest input point
                    KP_min_d2 = tf.reduce_min(deformed_d2, axis=1)

                    # Normalize KP locations to be independant from layers
                    KP_min_d2 = KP_min_d2 / (KP_extent**2)

                    # Loss will be the square distance to closest input point.
                    # Mean => loss independent from the number of input points
                    offset_losses += [tf.reduce_mean(KP_min_d2)]

                if op.name.endswith('deformed_KP'):

                    # Get deformed positions
                    deformed_KP = op.outputs[0]

                    # Layer index
                    layer = int(op.name.split('/')[1].split('_')[-1])

                    # Radius of deformed convolution for this layer
                    KP_extent = self.config.first_subsampling_dl * self.config.KP_extent * (2 ** layer)

                    # Normalized KP locations
                    KP_locs = deformed_KP/KP_extent

                    # Point should not be close to each other
                    for i in range(self.config.num_kernel_points):
                        other_KP = tf.stop_gradient(tf.concat([KP_locs[:, :i, :], KP_locs[:, i + 1:, :]], axis=1))
                        distances = tf.sqrt(tf.reduce_sum(tf.square(other_KP - KP_locs[:, i:i+1, :]), axis=2))
                        repulsive_losses = tf.reduce_sum(tf.square(tf.maximum(0.0, 1.5 - distances)), axis=1)
                        offset_losses += [tf.reduce_mean(repulsive_losses)]

        elif self.config.offsets_loss != 'none':
            raise ValueError('Unknown offset loss')

        if len(offset_losses) > 0:
            self.offsets_loss = self.config.offsets_decay * tf.add_n(offset_losses)
        else:
            self.offsets_loss = tf.constant(0, dtype=tf.float32)

        return self.offsets_loss + self.gaussian_loss + self.regularization_loss

    def parameters_log(self):

        self.config.save(self.saving_path)


def features_to_spectral(features, evecs_trans, stack_lengths, neig):
    """
    Multiply stacked features with stacked spectral data and store them into 3D tensor now that they all have same size
    :param features: Tensor with size [None, 64] where None is the total number of points in the stacked batch
    :param evecs_trans: Tensor with size [neig, None] where None is the same number of points as above
    :param stack_lengths: Tensor with size [None] where None is the number of batch
    :param neig: number of eigen vectors used for the representation of the fmaps
    """

    evecs_ = tf.transpose(evecs_trans)  # shape = [None, neig], same as features

    # building the indexes bracket
    slcs = tf.cumsum(stack_lengths)
    slcs0 = slcs[:-1]
    slcs0 = tf.concat([[0], slcs0], axis=0)
    slf = tf.stack([slcs0, slcs], axis=1)

    slf = tf.cast(slf, tf.float32)  # cast to float in order to be able to render floats

    def fun(x):  # input and output have to be of same dtype, and output of consistent size
        x = tf.cast(x, tf.int32)  # cast back to int to be able to index
        piece_evecs = evecs_[x[0]: x[1]]
        piece_feat = features[x[0]: x[1]]
        return tf.transpose(piece_evecs) @ piece_feat

    desc = tf.map_fn(fun, slf)  # here the output tensor is of size neig, d_desc

    return desc


# def global_fmap_loss(fmap, fmap_t, Or_op1, Or_op2, n_batch):
# def global_fmap_loss(fmap, fmap_t, va1, va2, n_batch):
def global_fmap_loss(fmap, fmap_t, of1, ev1, evt1, va1, of2, ev2, evt2, va2, n_batch):
    alpha = 1e3
    beta = 1e3
    gamma = 1
    delta = 1e5
    n = tf.cast(n_batch, tf.float32)
    E1 = penalty_bijectivity(fmap, fmap_t)
    E2 = (penalty_ortho(fmap)+penalty_ortho(fmap_t))/2
    #E3 = penalty_desc_orientation(fmap, Or_op1, Or_op2)
    E3 = penalty_laplacian_commutativity(fmap, va1, va2)
    #E4 = penalty_desc_commutativity(fmap, of1, of2, ev1, evt1, ev2, evt2)
    E4 = 0
    #return (alpha * E1 + beta * E2 + gamma * E3)/n
    #return (alpha * E1 + beta * E2 + gamma * 0.)/n
    return (alpha * E1 + beta * E2 + gamma * E3 + delta * E4) / n


def global_fmap_loss_sup(fmap, evecs_full_1, evecs_full_2, l1, l2, b1, b2, n_batch):
    n = tf.cast(n_batch, tf.float32)
    E1 = supervised_penalty(fmap, evecs_full_1, evecs_full_2, l1, l2, b1, b2, n_batch)
    return E1/n



















