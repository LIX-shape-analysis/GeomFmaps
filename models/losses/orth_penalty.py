import tensorflow as tf


def penalty_ortho(C_est):
    """Orthogonal constraint on the functional map
    implying that the underlying map T is area-preserving.

    Args:
        C_est : estimated functional from source to target or vice-versa.
    """

    return tf.nn.l2_loss(
                         tf.subtract(
                             tf.matmul(
                                       tf.transpose(C_est, perm=[0, 2, 1]),
                                       C_est),
                             tf.eye(tf.shape(C_est)[1]))
                        )


def penalty_sub_ortho(C, neig):
    """
    Introduced in Zoomout

    Orthogonal constraint on the functional map
    implying that all the submatrices of C need to be orthogonal as well.
    This energy should be low on the maps coming from point to point maps, and then enforces the map
    to be an excellent mapping between 2 shapes

    Args:
        C : estimated functional from source to target or vice-versa.
    """
    sub_Ck = [tf.transpose(C[:, 0:k, 0:k], [0, 2, 1]) @ C[:, 0:k, 0:k] - tf.eye(k) for k in range(1, neig+1)]

    loss = 0
    i = 0
    for ck in sub_Ck:
        i += 1
        loss += tf.nn.l2_loss(ck) / i

    return loss
