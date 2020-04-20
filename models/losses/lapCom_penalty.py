import tensorflow as tf


def penalty_laplacian_commutativity(C_est, source_evals, target_evals):
    """Loss function using the preservation
    under isometries of the Laplace-Beltrami operator.

    Args:
        C_est : estimated functional map. From source to target or vice-versa.
        source_evals : eigen values of the source shape.
        target_evals : eigen values of the target shape.
    """

    # reshape data
    C_shp = tf.shape(C_est)
    source_evals = tf.reshape(source_evals, [C_shp[0], C_shp[1]])
    target_evals = tf.reshape(target_evals, [C_shp[0], C_shp[1]])

    # Quicker and less memory than taking diagonal matrix
    eig1 = tf.einsum('abc,ac->abc', C_est, source_evals)
    eig2 = tf.einsum('ab,abc->abc', target_evals, C_est)

    return tf.nn.l2_loss(tf.subtract(eig2, eig1))

def penalty_laplacian_commutativity_101(C_est, source_evals, target_evals):
    """Loss function using the preservation
    under isometries of the Laplace-Beltrami operator.

    Args:
        C_est : estimated functional map. From source to target or vice-versa.
        source_evals : eigen values of the source shape.
        target_evals : eigen values of the target shape.
    """

    C_shp = tf.shape(C_est)
    D1 = tf.linalg.diag(tf.reshape(source_evals, [C_shp[0], C_shp[1]]))
    D2 = tf.linalg.diag(tf.reshape(target_evals, [C_shp[0], C_shp[1]]))

    return tf.nn.l2_loss(C_est @ D1 - D2 @ C_est)