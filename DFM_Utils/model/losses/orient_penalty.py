import tensorflow as tf


def orient_penalty(C_est, Or_op1, Or_op2):
    """Orientation constraint on the functional map
    implying that the underlying map T is orientation-preserving.

    Args:
        C_est : estimated functional from source to target or vice-versa.
        Or_op : Orientation operator, supposed to capture the orientation of a given descriptor
    """

    return tf.nn.l2_loss(
                         tf.subtract(
                             tf.matmul(
                                 Or_op,
                                 C_est),
                             tf.matmul(
                                 C_est,
                                 Or_op),
                         )
                        )
