#
#
#      0=================================0
#      |    Kernel Point Convolutions    |
#      0=================================0
#
#
# ----------------------------------------------------------------------------------------------------------------------
#
#      Class handling the test of any model
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
from os import makedirs, listdir
from os.path import exists, join
import time
from sklearn.neighbors import KDTree

# PLY reader
from utils.ply import read_ply, write_ply

from tensorflow.python.client import timeline
import json


# ----------------------------------------------------------------------------------------------------------------------
#
#           Tester Class
#       \******************/
#

class TimeLiner:

    def __init__(self):
        self._timeline_dict = None

    def update_timeline(self, chrome_trace):

        # convert crome trace to python dict
        chrome_trace_dict = json.loads(chrome_trace)

        # for first run store full trace
        if self._timeline_dict is None:
            self._timeline_dict = chrome_trace_dict

        # for other - update only time consumption, not definitions
        else:
            for event in chrome_trace_dict['traceEvents']:
                # events time consumption started with 'ts' prefix
                if 'ts' in event:
                    self._timeline_dict['traceEvents'].append(event)

    def save(self, f_name):
        with open(f_name, 'w') as f:
            json.dump(self._timeline_dict, f)


class ModelTester:

    # Initiation methods
    # ------------------------------------------------------------------------------------------------------------------

    def __init__(self, model, restore_snap=None):

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

        # Init variables -- not necessary ...
        self.sess.run(tf.global_variables_initializer())

        # Name of the snapshot to restore to (None if you want to start from beginning)
        # restore_snap = join(self.saving_path, 'snapshots/snap-40000')
        if restore_snap is not None:
            print('restoring...')
            self.saver.restore(self.sess, restore_snap)
            print("Model restored from " + restore_snap)
            self.restore_snap = str(int(float(restore_snap.split('-')[-1]) / model.config.epoch_steps)) + '_epochs'

        # self.config = model.config

        self.fmaps = model.fmaps
        #self.fmaps_t = model.fmaps_t
        # self.output_features = model.output_features
        # self.output_features_2 = model.output_features_2
        self.desc = model.desc
        self.desc_2 = model.desc_2
        self.bi = model.inputs['batch_inds']
        self.bi_2 = model.inputs_2['batch_inds']
        #self.pts = model.inputs['points']
        #self.pts_2 = model.inputs_2['points']

    # Test main methods
    # ------------------------------------------------------------------------------------------------------------------

    def test_shape_matching(self, model, dataset):
        ##################
        # Pre-computations
        ##################

        print('Preparing test structures')
        t1 = time.time()


        ##########
        # Initiate
        ##########

        # Test saving path
        if model.config.saving:
            savepath = join('test', model.saving_path.split('/')[-1])
            if not exists(savepath):
                makedirs(savepath)
        else:
            savepath = None

        print(savepath)
        # Initialise iterator with test data
        self.sess.run(dataset.test_init_op)
        # self.sess.run(dataset.train_init_op)

        # Initiate result containers
        # average_predictions = [np.zeros((1, 1), dtype=np.float32) for _ in test_names]

        #####################
        # Network predictions
        #####################

        # mean_dt = np.zeros(2)
        # last_display = time.time()
        # for v in range(num_votes):
        #
        #     # Run model on all test examples
        #     # ******************************
        #
        # Initiate result containers
        fmaps = []
        #fmaps_t = []
        desc1 = []
        desc2 = []
        #of1 = []
        #of2 = []
        bi1 = []
        bi2 = []
        #pts = []
        #pts2 = []

        while True:
            try:

                # Run one step of the model
                t = [time.time()]
                ops = (self.fmaps,
                       #self.fmaps_t,
                       #self.output_features,
                       #self.output_features_2,
                       self.desc,
                       self.desc_2,
                       self.bi,
                       self.bi_2,
                       #self.pts,
                       #self.pts_2
                       )

                # C, C_t, F, G, A, B, b1, b2 = self.sess.run(ops, {model.dropout_prob: 0.5})
                # C, A, B, b1, b2, ps, ps2 = self.sess.run(ops, {model.dropout_prob: 0.5})
                C, A, B, b1, b2 = self.sess.run(ops, {model.dropout_prob: 0.5})
                # C, A, B, b1, b2 = self.sess.run(ops, {model.dropout_prob: 1.0})
                t += [time.time()]

                fmaps += [C]
                #fmaps_t += [C_t]
                desc1 += [A]
                desc2 += [B]
                #of1 += [F]
                #of2 += [G]
                bi1 += [b1]
                bi2 += [b2]
                #pts += [ps]
                #pts2 += [ps2]

            except tf.errors.OutOfRangeError:
                break

        fld_name = model.fld_name
        if self.restore_snap is not None:
            fld_name = fld_name + '/' + self.restore_snap
        if not exists(join(savepath, fld_name)):
            print('creating tests path for this new setting')
            makedirs(join(savepath, fld_name))
        np.save(join(savepath, fld_name, 'fmaps.npy'), fmaps)
        #np.save(join(savepath, 'fmapst.npy'), fmaps_t)
        np.save(join(savepath, fld_name, 'desc1.npy'), desc1)
        np.save(join(savepath, fld_name, 'desc2.npy'), desc2)
        #np.save(join(savepath, 'of1.npy'), of1)
        #np.save(join(savepath, 'of2.npy'), of2)
        np.save(join(savepath, fld_name, 'bi1.npy'), bi1)
        np.save(join(savepath, fld_name, 'bi2.npy'), bi2)
        #np.save(join(savepath, 'pts.npy'), pts)
        #np.save(join(savepath, 'pts2.npy'), pts2)


        t2 = time.time()
        print('Done in {:.1f} s\n'.format(t2 - t1))



        return






























