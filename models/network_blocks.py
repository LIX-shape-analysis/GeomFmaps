#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Definition of networks models with blocks
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
import numpy as np
import tensorflow as tf
import kernels.convolution_ops as conv_ops


# ----------------------------------------------------------------------------------------------------------------------
#
#           Utilities
#       \***************/
#

def weight_variable(shape):
    tf.set_random_seed(42)
    print(shape[-1])
    initial = tf.truncated_normal(shape, stddev=np.sqrt(2 / shape[-1]))
    initial = tf.round(initial * tf.constant(1000, dtype=tf.float32)) / tf.constant(1000, dtype=tf.float32)
    return tf.Variable(initial, name='weights')


def bias_variable(shape):
    initial = tf.constant(0., shape=shape)
    return tf.Variable(initial, name='bias')


def ind_max_pool(x, inds):
    """
    This tensorflow operation compute a maxpooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] each row of this tensor is a list of indices of features to be pooled together
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.reduce_min(x, axis=0, keep_dims=True)], axis=0)

    # Get features for each pooling cell [n2, max_num, d]
    pool_features = tf.gather(x, inds, axis=0)

    # Pool the maximum
    return tf.reduce_max(pool_features, axis=1)


def closest_pool(x, inds):
    """
    This tensorflow operation compute a pooling according to the list of indices 'inds'.
    > x = [n1, d] features matrix
    > inds = [n2, max_num] We only use the first column of this which should be the closest points too pooled positions
    >> output = [n2, d] pooled features matrix
    """

    # Add a last row with minimum features for shadow pools
    x = tf.concat([x, tf.zeros((1, int(x.shape[1])), x.dtype)], axis=0)

    # Get features for each pooling cell [n2, d]
    pool_features = tf.gather(x, inds[:, 0], axis=0)

    return pool_features


def KPConv(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv(query_points,
                           support_points,
                           neighbors_indices,
                           features,
                           K_values,
                           fixed=config.fixed_kernel_points,
                           KP_extent=extent,
                           KP_influence=config.KP_influence,
                           aggregation_mode=config.convolution_mode,)


def KPConv_deformable(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Returns the output features of a deformable KPConv
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv_deformable(query_points,
                                      support_points,
                                      neighbors_indices,
                                      features,
                                      K_values,
                                      fixed=config.fixed_kernel_points,
                                      KP_extent=extent,
                                      KP_influence=config.KP_influence,
                                      aggregation_mode=config.convolution_mode,
                                      modulated=config.modulated)


def KPConv_deformable_v2(query_points, support_points, neighbors_indices, features, K_values, radius, config):
    """
    Perform a simple convolution followed by a batch normalization (or a simple bias) and ReLu
    """

    # Get KP extent from current radius and config density
    extent = config.KP_extent * radius / config.density_parameter

    # Convolution
    return conv_ops.KPConv_deformable_v2(query_points,
                                         support_points,
                                         neighbors_indices,
                                         features,
                                         K_values,
                                         config.num_kernel_points,
                                         fixed=config.fixed_kernel_points,
                                         KP_extent=extent,
                                         KP_influence=config.KP_influence,
                                         mode=config.convolution_mode,
                                         modulated=config.modulated)


def batch_norm(x, use_batch_norm=True, momentum=0.99, training=True):
    """
    This tensorflow operation compute a batch normalization.
    > x = [n1, d] features matrix
    >> output = [n1, d] normalized, scaled, offset features matrix
    """

    if use_batch_norm:
        return tf.layers.batch_normalization(x,
                                             momentum=momentum,
                                             epsilon=1e-6,
                                             training=training,
                                             name='bn')  # maybe include name of layer ?

    else:
        # Just add biases
        beta = tf.Variable(tf.zeros([x.shape[-1]]), name='offset')
        return x + beta


def leaky_relu(features, alpha=0.2):
    return tf.nn.leaky_relu(features, alpha=alpha, name=None)

# ----------------------------------------------------------------------------------------------------------------------
#
#           Convolution blocks
#       \************************/
#


def unary_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple 1x1 convolution
    """

    w = weight_variable([int(features.shape[1]), fdim])
    x = conv_ops.unary_convolution(features, w)
    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def simple_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind],
               inputs['points'][layer_ind],
               inputs['neighbors'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def simple_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple strided convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind + 1],
               inputs['points'][layer_ind],
               inputs['pools'][layer_ind],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnet_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet double convolution (two convolution vgglike and a shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   features,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_light_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            x = conv_ops.unary_convolution(features, w)
            x = batch_norm(x,
                           config.use_batch_norm,
                           config.batch_norm_momentum,
                           training)
        else:
            x = features

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a resnet bottleneck convolution (1conv > KPconv > 1conv + shortcut)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind],
                              inputs['points'][layer_ind],
                              inputs['neighbors'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def inception_deformable_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an inception style convolution combining rigid and deformable KPConv
             (1conv > rigid) \
                               > CONCAT > 1conv + shortcut
    (1conv > rigid > deform) /
    """

    with tf.variable_scope('path1'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x1 = conv_ops.unary_convolution(features, w)
            x1 = leaky_relu(batch_norm(x1,
                                      config.use_batch_norm,
                                      config.batch_norm_momentum,
                                      training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x1.shape[1]), fdim // 2])
            x1 = KPConv(inputs['points'][layer_ind],
                        inputs['points'][layer_ind],
                        inputs['neighbors'][layer_ind],
                        x1,
                        w,
                        radius,
                        config)

    with tf.variable_scope('path2'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x2 = conv_ops.unary_convolution(features, w)
            x2 = leaky_relu(batch_norm(x2,
                                       config.use_batch_norm,
                                       config.batch_norm_momentum,
                                       training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv(inputs['points'][layer_ind],
                        inputs['points'][layer_ind],
                        inputs['neighbors'][layer_ind],
                        x2,
                        w,
                        radius,
                        config)

        with tf.variable_scope('conv2_deform'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv_deformable_v2(inputs['points'][layer_ind],
                                      inputs['points'][layer_ind],
                                      inputs['neighbors'][layer_ind],
                                      x2,
                                      w,
                                      radius,
                                      config)

    with tf.variable_scope('concat'):
        x = tf.concat([x1, x2], axis=1)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('unary'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):
        if int(features.shape[1]) != 2 * fdim:
            w = weight_variable([int(features.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(features, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)
        else:
            shortcut = features

    return leaky_relu(x + shortcut)


def resnetb_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def resnetb_light_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        if int(features.shape[1]) != fdim:
            w = weight_variable([int(features.shape[1]), fdim])
            x = conv_ops.unary_convolution(features, w)
            x = batch_norm(x,
                           config.use_batch_norm,
                           config.batch_norm_momentum,
                           training)
        else:
            x = features

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind + 1],
                   inputs['points'][layer_ind],
                   inputs['pools'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def resnetb_deformable_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a strided resnet bottleneck convolution (shortcut is a maxpooling)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv_deformable(inputs['points'][layer_ind + 1],
                              inputs['points'][layer_ind],
                              inputs['pools'][layer_ind],
                              x,
                              w,
                              radius,
                              config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def inception_deformable_strided_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an inception style convolution combining rigid and deformable KPConv
             (1conv > rigid) \
                               > CONCAT > 1conv + shortcut
    (1conv > rigid > deform) /
    """

    with tf.variable_scope('path1'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x1 = conv_ops.unary_convolution(features, w)
            x1 = leaky_relu(batch_norm(x1,
                                      config.use_batch_norm,
                                      config.batch_norm_momentum,
                                      training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x1.shape[1]), fdim // 2])
            x1 = KPConv(inputs['points'][layer_ind+1],
                        inputs['points'][layer_ind],
                        inputs['pools'][layer_ind],
                        x1,
                        w,
                        radius,
                        config)

    with tf.variable_scope('path2'):

        with tf.variable_scope('unary'):
            w = weight_variable([int(features.shape[1]), fdim // 2])
            x2 = conv_ops.unary_convolution(features, w)
            x2 = leaky_relu(batch_norm(x2,
                                       config.use_batch_norm,
                                       config.batch_norm_momentum,
                                       training))

        with tf.variable_scope('conv'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv(inputs['points'][layer_ind+1],
                        inputs['points'][layer_ind],
                        inputs['pools'][layer_ind],
                        x2,
                        w,
                        radius,
                        config)

        with tf.variable_scope('conv2_deform'):
            w = weight_variable([config.num_kernel_points, int(x2.shape[1]), fdim // 2])
            x2 = KPConv_deformable_v2(inputs['points'][layer_ind+1],
                                      inputs['points'][layer_ind],
                                      inputs['pools'][layer_ind],
                                      x2,
                                      w,
                                      radius,
                                      config)

    with tf.variable_scope('concat'):
        x = tf.concat([x1, x2], axis=1)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('unary'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points TODO: max_pool or closest_pool ?
        shortcut = ind_max_pool(features, inputs['pools'][layer_ind])
        # shortcut = closest_pool(features, neighbors_indices)

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def vgg_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing two simple convolutions vgg style
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   features,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim])
        x = KPConv(inputs['points'][layer_ind],
                   inputs['points'][layer_ind],
                   inputs['neighbors'][layer_ind],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    return x


def max_pool_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a max pooling
    """

    with tf.variable_scope('max_pool'):
        pooled_features = ind_max_pool(features, inputs['pools'][layer_ind])

    return pooled_features


def global_average_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a global average over batch pooling
    """

    # Average pooling to aggregate feature in the end
    with tf.variable_scope('average_pooling'):

        # Get the number of features
        N = tf.shape(features)[0]

        # Add a last zero features for shadow batch inds
        features = tf.concat([features, tf.zeros((1, int(features.shape[1])), features.dtype)], axis=0)

        # Collect each batch features
        batch_features = tf.gather(features, inputs['out_batches'], axis=0)

        # Average features in each batch
        batch_features = tf.reduce_sum(batch_features, axis=1)
        #batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] >= 0, tf.float32), axis=1, keep_dims=True)
        batch_num = tf.reduce_sum(tf.cast(inputs['out_batches'] < N, tf.float32), axis=1, keep_dims=True)

        features = batch_features / batch_num

    return features


def simple_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing a simple upsampling convolution
    """

    # Weights
    w = weight_variable([config.num_kernel_points, int(features.shape[1]), fdim])

    # Convolution
    x = KPConv(inputs['points'][layer_ind - 1],
               inputs['points'][layer_ind],
               inputs['upsamples'][layer_ind - 1],
               features,
               w,
               radius,
               config)

    x = leaky_relu(batch_norm(x,
                              config.use_batch_norm,
                              config.batch_norm_momentum,
                              training))

    return x


def resnetb_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling resnet bottleneck convolution (shortcut is nearest interpolation)
    """

    with tf.variable_scope('conv1'):
        w = weight_variable([int(features.shape[1]), fdim // 2])
        x = conv_ops.unary_convolution(features, w)
        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv2'):
        w = weight_variable([config.num_kernel_points, int(x.shape[1]), fdim // 2])
        x = KPConv(inputs['points'][layer_ind - 1],
                   inputs['points'][layer_ind],
                   inputs['upsamples'][layer_ind - 1],
                   x,
                   w,
                   radius,
                   config)

        x = leaky_relu(batch_norm(x,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training))

    with tf.variable_scope('conv3'):
        w = weight_variable([int(x.shape[1]), 2 * fdim])
        x = conv_ops.unary_convolution(x, w)
        x = batch_norm(x,
                       config.use_batch_norm,
                       config.batch_norm_momentum,
                       training)

    with tf.variable_scope('shortcut'):

        # Pool shortcuts to strided points (nearest interpolation)
        shortcut = closest_pool(features, inputs['upsamples'][layer_ind - 1])

        # Regular upsample of the features if not the same dimension
        if int(shortcut.shape[1]) != 2 * fdim:
            w = weight_variable([int(shortcut.shape[1]), 2 * fdim])
            shortcut = conv_ops.unary_convolution(shortcut, w)
            shortcut = batch_norm(shortcut,
                                  config.use_batch_norm,
                                  config.batch_norm_momentum,
                                  training)

    return leaky_relu(x + shortcut)


def nearest_upsample_block(layer_ind, inputs, features, radius, fdim, config, training):
    """
    Block performing an upsampling by nearest interpolation
    """

    with tf.variable_scope('nearest_upsample'):
        upsampled_features = closest_pool(features, inputs['upsamples'][layer_ind - 1])

    return upsampled_features


def get_block_ops(block_name):

    if block_name == 'unary':
        return unary_block

    if block_name == 'simple':
        return simple_block

    if block_name == 'simple_strided':
        return simple_strided_block

    elif block_name == 'resnet':
        return resnet_block

    elif block_name == 'resnetb':
        return resnetb_block

    elif block_name == 'resnetb_light':
        return resnetb_light_block

    elif block_name == 'resnetb_deformable':
        return resnetb_deformable_block

    elif block_name == 'inception_deformable':
        return inception_deformable_block()

    elif block_name == 'resnetb_strided':
        return resnetb_strided_block

    elif block_name == 'resnetb_light_strided':
        return resnetb_light_strided_block

    elif block_name == 'resnetb_deformable_strided':
        return resnetb_deformable_strided_block

    elif block_name == 'inception_deformable_strided':
        return inception_deformable_strided_block()

    elif block_name == 'vgg':
        return vgg_block

    elif block_name == 'max_pool' or block_name == 'max_pool_wide':
        return max_pool_block

    elif block_name == 'global_average':
        return global_average_block

    elif block_name == 'nearest_upsample':
        return nearest_upsample_block

    elif block_name == 'simple_upsample':
        return simple_upsample_block

    elif block_name == 'resnetb_upsample':
        return resnetb_upsample_block

    else:
        raise ValueError('Unknown block name in the architecture definition : ' + block_name)


# ----------------------------------------------------------------------------------------------------------------------
#
#           Architectures
#       \*******************/
#


def assemble_CNN_FM_blocks(inputs, inputs_2, config, dropout_prob):
    """
    Definition of all the layers according to config architecture
    :param inputs: dictionary of inputs with keys [points, neighbors, pools, upsamples, features, batches]
                                             and  [points2, neighbors2, pools2, upsamples2, features2, batches2]
    :param config:
    :param dropout_prob:
    :return:
    """

    # Current radius of convolution and feature dimension
    r = config.first_subsampling_dl * config.density_parameter
    layer = 0
    fdim = config.first_features_dim

    # Input features
    features = inputs['features']
    features_2 = inputs_2['features']
    F = []
    F_2 = []

    # Boolean of training
    training = dropout_prob < 0.99

    # Loop over consecutive blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture):

        # Detect change to next layer
        if np.any([tmp in block for tmp in ['pool', 'strided', 'upsample', 'global']]):
            # Save this layer features
            F += [features]
            F_2 += [features_2]

        # Detect upsampling block to stop
        if 'upsample' in block:
            break

        ##### downsampling blocks #####
        with tf.variable_scope('layer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer), reuse=tf.AUTO_REUSE) as scope:

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

            scope.reuse_variables()

            features_2 = block_ops(layer,
                                   inputs_2,
                                   features_2,
                                   r,
                                   fdim,
                                   config,
                                   training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'pool' in block or 'strided' in block:
            # Update radius and feature dimension for next layer
            layer += 1
            r *= 2
            fdim *= 2
            block_in_layer = 0

        # Save feature vector after global pooling
        if 'global' in block:
            # Save this layer features
            F += [features]
            F_2 += [features_2]

    features = F[-1]
    features_2 = F_2[-1]

    # Current radius of convolution and feature dimension
    layer = config.num_layers - 1
    r = config.first_subsampling_dl * config.density_parameter * 2 ** layer
    fdim = config.first_features_dim * 2 ** layer # if you use resnet, fdim is actually 2 times that

    # Boolean of training
    training = dropout_prob < 0.99

    # Find first upsampling block
    start_i = 0
    for block_i, block in enumerate(config.architecture):
        if 'upsample' in block:
            start_i = block_i
            break

    ##### upsampling blocks #####
    # Loop over upsampling blocks
    block_in_layer = 0
    for block_i, block in enumerate(config.architecture[start_i:]):

        with tf.variable_scope('uplayer_{:d}/{:s}_{:d}'.format(layer, block, block_in_layer)) as scope:

            # Get the function for this layer
            block_ops = get_block_ops(block)

            # Apply the layer function defining tf ops
            features = block_ops(layer,
                                 inputs,
                                 features,
                                 r,
                                 fdim,
                                 config,
                                 training)

            scope.reuse_variables()

            features_2 = block_ops(layer,
                                   inputs_2,
                                   features_2,
                                   r,
                                   fdim,
                                   config,
                                   training)

        # Index of block in this layer
        block_in_layer += 1

        # Detect change to a subsampled layer
        if 'upsample' in block:

            # Update radius and feature dimension for next layer
            layer -= 1
            r *= 0.5
            fdim = fdim // 2
            block_in_layer = 0

            # Concatenate with CNN feature map
            features = tf.concat((features, F[layer]), axis=1)
            features_2 = tf.concat((features_2, F_2[layer]), axis=1)

    return features, features_2


def shape_matching_head(A, B, config):
    """functional maps layer.

        Args:
            A: source descriptors projected onto source shape eigenvectors.
            B: target descriptors projected onto target shape eigenvectors.

        Returns:
            Ct_est: estimated C (transposed), such that CA ~= B
            safeguard_inverse:
        """

    # Transpose input matrices
    At = tf.transpose(A, [0, 2, 1])
    Bt = tf.transpose(B, [0, 2, 1])

    # Solve C via least-squares
    Ct_est = tf.matrix_solve_ls(At, Bt)
    fmap = tf.transpose(Ct_est, [0, 2, 1], name='C_est')

    return fmap


def shape_matching_head_2(A, B, val1, val2, config):
    # TODO : optimize this reg : can be very useful when neig goes up to 50, 80 and especially over 100
    """functional maps layer with reg.

    Args:
        A: source descriptors projected onto source shape eigenvectors.
        B: target descriptors projected onto target shape eigenvectors.
        L: batch tensor, each containing kn diagonal matrices storing squared differences of eigenvalues

    Returns:
        Ct_est: estimated C (transposed), such that CA ~= B up to a regularization factor
        Actually the formula becomes for each line Ci of C, Ci = (AA.T + lambda Li)^(-1) ABi with Bi line i of B
        here we just take line i of BA.T instead of computing A @ Bi... Bad ?
    """
    lam = config.lam_reg
    n_evecs = config.neig

    # get matrices for pseudo inversion
    At = tf.transpose(A, [0, 2, 1])
    Ap = tf.matmul(A, At)
    Bp = tf.matmul(B, At)

    # get the eigenvalues diagonal set of matrices
    val1 = tf.transpose(val1)
    val2 = tf.transpose(val2)
    vals = tf.square(val2[:, None, :] - val1[:, :, None])
    D = tf.matrix_diag(vals)
    L = tf.transpose(D, [1, 0, 2, 3])

    # Solve C via least-squares with reg
    Ct_stack = []
    for i in range(n_evecs):
        L_i = tf.gather(L, i)
        B_i = tf.gather(tf.transpose(Bp, [1, 0, 2]), i)
        B_i = tf.reshape(B_i, [tf.shape(B_i)[0], n_evecs, 1])

        Ct_est_i = tf.matmul(tf.matrix_inverse(Ap + lam * L_i), B_i)
        # Ct_est_i = tf.matrix_solve_ls(Ap + lam * L_i, B_i)

        Ct_stack += [Ct_est_i]

    Ct_est = tf.concat(Ct_stack, axis=2)
    fmap = tf.transpose(Ct_est, [0, 2, 1], name='C_est')

    return fmap


