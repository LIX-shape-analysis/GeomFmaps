import tensorflow as tf
import numpy as np
import os

def supervised_penalty(C_est, full_source_evecs, full_target_evecs, l1, l2, b1, b2, n_batch, gtmap= None):
    """Orthogonal constraint on the functional map
    implying that the underlying map T is area-preserving.

    Args:
        C_est : estimated functional from source to target or vice-versa.
    """
    vts = []
    vts_folder = '../../../../media/donati/Data1/Datasets/FAUST_r/corres/'
    if os.path.exists(vts_folder):
        print('loading vts...')
        for i in range(80):
            filename = 'tr_reg_{:03d}.vts'
            phi = np.loadtxt(os.path.join(vts_folder, filename.format(i)), dtype = np.int32) - 1
            vts += [phi]
    else:
        print('could not find vts files')
    
     
    n_points = tf.cast(tf.shape(full_source_evecs)[0]/n_batch, tf.int32)
    n_eig = tf.shape(full_source_evecs)[1]
    full_source_evecs = tf.reshape(full_source_evecs, [n_batch, n_points, n_eig])
    full_target_evecs = tf.reshape(full_target_evecs, [n_batch, n_points, n_eig])

    #in surreal_dfaust case ground_truth maps are simply identity
    gtmap = tf.range(tf.shape(full_source_evecs)[0])
    #print(tf.shape(gtmap))
    C_gt = fmap_from_p2p(gtmap, full_source_evecs, full_target_evecs)

    # in faust_r we need vts files and all the evecs are not the right size so it is trickier
    #C_gt = fmap_faust(full_source_evecs, full_target_evecs, l1, l2, b1, b2, vts, n_batch)

    #C_gt = fmap_from_p2p(gtmap, full_source_evecs, full_target_evecs)
    #M11 = tf.ones([10, 10]) + 10. * tf.eye(10)
    #M12 = tf.ones([10, 20])
    #M21 = tf.ones([20, 10])
    #M22 = 3. * tf.ones([20, 20])
    #Mask = tf.concat([tf.concat([M11,M12],1),tf.concat([M21,M22],1)],0)  # mask matrix for weight importance
    #Mask = tf.ones([n_batch, 1]) * Mask
    #return tf.nn.l2_loss(Mask * (C_est - C_gt))
    return tf.nn.l2_loss(C_est - C_gt)

def fmap_from_p2p(gtmap, full_source_evecs, full_target_evecs):
    fmap_gt = tf.matrix_solve_ls(full_target_evecs, full_source_evecs) # in surreal _dfaust the gtmap can even be removed
    return fmap_gt

def fmap_faust(ev_s, ev_t, l1, l2, b1, b2, vts, n_batch):
    #print(vts.shape)
    vts = np.stack(vts)
    print(np.max(vts, axis = 1))
    vts = tf.constant(vts, dtype=tf.int32)
    print(vts)
    slcs = tf.cumsum(l1)
    slcs0 = slcs[:-1]
    slcs0 = tf.concat([[0], slcs0], axis=0)
    slf1 = tf.stack([slcs0, slcs], axis=1)

    slcs = tf.cumsum(l2)
    slcs0 = slcs[:-1]
    slcs0 = tf.concat([[0], slcs0], axis=0)
    slf2 = tf.stack([slcs0, slcs], axis=1)

    #n_batch = tf.shape(b1)[0]
    n_eig = tf.shape(ev_s)[1]
    
    fmaps = []
    for i in range(4):
        bi1 = b1[i]; bi2 = b2[i]
        #inds1, _ = tf.unique(vts[bi1]); inds2, _ = tf.unique(vts[bi2])
        inds1 = vts[bi1] ; inds2 = vts[bi2]
        slc1 = slf1[i] ; slc2 = slf2[i]
        ev_s_i = ev_s[slc1[0]:slc1[1]]
        ev_s_i = tf.gather(ev_s_i,inds1)
        ev_t_i = ev_t[slc2[0]:slc2[1]]
        ev_t_i = tf.gather(ev_t_i,inds2)
        
        print(ev_t_i)
        fmap_gt = tf.matrix_solve_ls(ev_t_i, ev_s_i)
        fmaps += [fmap_gt]

    return tf.stack(fmaps, axis =0)
        
