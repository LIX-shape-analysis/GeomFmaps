import tensorflow as tf


def penalty_bijectivity(C_est_AB, C_est_BA):
    """Having both functionals maps for two given shapes,
       composition should yield identity.

    Args:
        C_est_AB : estimated functional from source to target.
        C_est_BA : estimated functional from target to source.
    """

    return tf.nn.l2_loss(
                        tf.subtract(tf.matmul(C_est_AB, C_est_BA),
                                    tf.eye(tf.shape(C_est_AB)[1])
                                    )
                        )
