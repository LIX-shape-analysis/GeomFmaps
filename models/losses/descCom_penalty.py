import tensorflow as tf


flags = tf.app.flags
FLAGS = flags.FLAGS

# def penalty_desc_commutativity(
#         C_est, F, G, source_evecs, source_evecs_trans,
#         target_evecs, target_evecs_trans):
#     """Descriptors preservation constraint using commutativity
#     from Dorian Nogneng's paper : Informative Descriptor Preservation via
#     Commutativity for Shape Matching, 2017 EUROGRAPHICS.
#
#     Args:
#         C_est: estimated functional map from source to target or vice-versa.
#         F : Descriptors on source shape, in full basis.
#         G : Descriptors on target shape, in full basis.
#         source_evecs : eigen vectors of target shape.
#         source_evecs_trans : source shape eigen vectors, transposed with area
#                             preservation factor.
#         target_evecs : eigen vectors of target shape.
#         target_evecs_trans : target shape eigen vectors, transposed with area
#                             preservation factor.
#     """
#
#     F_trans = tf.transpose(F, perm=[0, 2, 1])  # Columns become rows
#     G_trans = tf.transpose(G, perm=[0, 2, 1])
#     # Size : [batch, shot_dim, num_vertices]
#
#     percent = 50  # Chosing percent of the total number of descriptors
#     #dim_out = np.array(FLAGS.lay_dims).astype(int)[-1]
#     dim_out = FLAGS.dim_out
#
#     num_desc = int(dim_out*percent/100)
#     batch_range = tf.tile(
#                         tf.reshape(
#                                     tf.range(FLAGS.batch_size, dtype=tf.int32),
#                                     shape=[FLAGS.batch_size, 1, 1]),
#                         [1, num_desc, 1])
#     random_idx = tf.random_uniform(
#                             [FLAGS.batch_size, num_desc, 1],
#                             minval=0,
#                             maxval=tf.shape(F)[1] - 1,
#                             dtype=tf.int32)
#
#     indices = tf.concat([batch_range, random_idx], axis=2)
#
#     F_ = tf.gather_nd(F_trans, indices)  # percent% of descriptors chosen
#     G_ = tf.gather_nd(G_trans, indices)
#
#     F_expand = tf.expand_dims(F_, 2)
#     G_expand = tf.expand_dims(G_, 2)
#     # Size : # [batch, num_desc, 1, num_vertices]
#
#     # This is quicker than taking a diagonal matrix for the descriptor
#     F_diag_reduce1 = tf.einsum('abcd,ade->abcde', F_expand, source_evecs)
#     G_diag_reduce1 = tf.einsum('abcd,ade->abcde', G_expand, target_evecs)
#     # Size : [batch, num_desc, 1, num_vertices, num_evecs]
#
#     F_diag_reduce2 = tf.einsum(
#                             'afd,abcde->abcfe',
#                             source_evecs_trans,
#                             F_diag_reduce1)
#     G_diag_reduce2 = tf.einsum(
#                             'afd,abcde->abcfe',
#                             target_evecs_trans,
#                             G_diag_reduce1)
#     # Size : #[batch, num_desc, 1, num_evecs, num_evecs]
#
#     C_est_expand = tf.expand_dims(tf.expand_dims(C_est, 1), 1)
#
#     C_est_tile = tf.tile(C_est_expand, [1, num_desc, 1, 1, 1])
#
#     term_source = tf.einsum('abcde,abcef->abcdf', C_est_tile, F_diag_reduce2)
#     term_target = tf.einsum('abcef,abcfd->abced', G_diag_reduce2, C_est_tile)
#
#     subtract = tf.subtract(term_source, term_target)
#
#     return tf.nn.l2_loss(subtract)

def from_features_to_spectral_operators(ofs, ev, evt,  stack_lengths, neig):
    """
    Multiply stacked features with stacked spectral data and store them into 3D tensor now that they all have same size
    :param features: Tensor with size [None, 64] where None is the total number of points in the stacked batch
    :param evecs_trans: Tensor with size [neig, None] where None is the same number of points as above
    :param stack_lengths: Tensor with size [None] where None is the number of batch
    :param neig: number of eigen vectors used for the representation of the fmaps
    """

    evecs_t = tf.transpose(evt)  # shape = [None, neig], same as features

    # building the indexes bracket
    slcs = tf.cumsum(stack_lengths)
    slcs0 = slcs[:-1]
    slcs0 = tf.concat([[0], slcs0], axis=0)
    slf = tf.stack([slcs0, slcs], axis=1)

    slf = tf.cast(slf, tf.float32)  # cast to float in order to be able to render floats
    desc_ops = []

    for i in range(tf.shape(ofs)[1]):
        of = ofs[i]
        def fun(x):  # input and output have to be of same dtype, and output of consistent size
            x = tf.cast(x, tf.int32)  # cast back to int to be able to index
            piece_evecs = ev[x[0]: x[1]]
            piece_evecs_t = evecs_t[x[0]: x[1]]
            piece_feat = of[x[0]: x[1]]
            return tf.transpose(piece_evecs_t) @ (piece_feat * piece_evecs)

        desc_op = tf.map_fn(fun, slf)  # here the output tensor is of size neig, d_desc
        desc_ops += [desc_op]

    return tf.concat(desc_ops, axis = 1)

def penalty_desc_commutativity(fmap, of1, of2, ev1, evt1, ev2, evt2):
    return
