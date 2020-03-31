import numpy as np
import matplotlib.pyplot as plt
from os.path import exists, join
import pickle
from scipy import spatial
import scipy.io as sio
import meshplot as mp
from off import *
from fmap import *

#plt.rcParams['figure.figsize'] = 7, 7

#TODO: correct the mistakes and clean code

class viz:
    
    def load_data(self, Log, dataset):
        
        foldername = join(Log)

        if dataset == 'FAUST':
            foldername = join(Log)
            filename = '../../../../media/donati/Data1/Datasets/FAUST_r/validation_-1.000_record.pkl'

            # load full data
            if exists(filename):
                with open(filename, 'rb') as file:
                    self.input_points, \
                    self.input_normals, \
                    self.input_evecs, \
                    self.input_evecs_trans, \
                    self.input_evals, \
                    _ = pickle.load(file)

        if dataset == 'SCAPE':
            foldername = join(Log)
            filename = '../../../../media/donati/Data1/Datasets/SCAPE_r/validation_-1.000_record.pkl'

            # load full data
            if exists(filename):
                with open(filename, 'rb') as file:
                    self.input_points, \
                    self.input_normals, \
                    self.input_evecs, \
                    self.input_evecs_trans, \
                    self.input_evals, \
                    _ = pickle.load(file)
        
        if dataset == 'SHREC':
            foldername = join(Log)
            filename = '../../../../media/donati/Data1/Datasets/SHREC_r/pickle/validation/3/validation_-1.000_record.pkl'

            # load full data
            if exists(filename):
                with open(filename, 'rb') as file:
                    self.input_points, \
                    self.input_normals, \
                    self.input_evecs, \
                    self.input_evecs_trans, \
                    self.input_evals, \
                    _ = pickle.load(file)

        if dataset == 'surreal_dfaust':
            #filename = '../../../../media/donati/Data1/Datasets/surreal_dfaust/train_-1.000_record.pkl'
            filename = '../../../../media/donati/Data1/Datasets/surreal_dfaust/surreal_dfaust_val.txt'
            self.testfiles = np.loadtxt(filename,  dtype=np.str)

        # batches inds
        bi1 = np.load(join(foldername, 'bi1.npy'))
        bi2 = np.load(join(foldername, 'bi2.npy'))

        # spectral desc
        desc1 = np.load(join(foldername, 'desc1.npy'))
        desc2 = np.load(join(foldername, 'desc2.npy'))

        # raw desc
        #of1 = np.load(join(foldername, 'of1.npy'), allow_pickle=True)
        #of2 = np.load(join(foldername, 'of2.npy'), allow_pickle=True)

        # maps
        maps = np.load(join(foldername, 'fmaps.npy'))

        #### reshap
    
        print(desc1.shape)#, len(of1), of1[0].shape)
        n_val = desc1.shape[0] * desc1.shape[1]
        n_eig = desc1.shape[2]
        feat_d = desc1.shape[3]

        self.bi1 = np.reshape(bi1, [n_val,])
        self.bi2 = np.reshape(bi2, [n_val,])

        self.desc1 = np.reshape(desc1, [n_val, n_eig, feat_d])
        self.desc2 = np.reshape(desc2, [n_val, n_eig, feat_d])

        self.maps = np.reshape(maps, [n_val, n_eig, n_eig])
        print(self.desc1.shape)
        self.n_eig = n_eig
        self.dataset = dataset
        

    def viz_sp_desc_on_shapes(self, i_b, do_plot = 1):
        
        self.i_b = i_b
        i_s = self.bi1[i_b]
        i_t = self.bi2[i_b]
        n_eig = self.n_eig
        print(i_s, i_t)

        if self.dataset == 'FAUST':
            # file for fetching mesh data
            sourcefolder = '../../../../../../../media/donati/Data1/Datasets/FAUST_r'
            meshname = 'tr_reg_{:03d}.off'
            
            i_s_ = i_s + 80
            i_t_ = i_t + 80
            
            p_s, f_s = readOFF(join(sourcefolder, 'off_al',meshname.format(i_s_)))
            p_t, f_t = readOFF(join(sourcefolder, 'off_al', meshname.format(i_t_)))
            print('size1 :', p_s.shape[0], 'size2 :', p_t.shape[0])

            # evecs
            #self.ev_s = self.input_evecs[i_s]
            #self.ev_t = self.input_evecs[i_t]
            self.ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s_)[:-4]+'.mat'))['target_evecs']
            self.ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t_)[:-4]+'.mat'))['target_evecs']

            # spectral desc
            d_s = self.desc1[i_b]
            d_t = self.desc2[i_b]

        elif self.dataset == 'SCAPE':
            # file for fetching mesh data
            sourcefolder = '../../../../../../../media/donati/Data1/Datasets/SCAPE_r'
            meshname = 'mesh{:03d}.off'
            
            i_s_ = i_s + 52
            i_t_ = i_t + 52

            p_s, f_s = readOFF(join(sourcefolder, 'off_al2',meshname.format(i_s_)))
            p_t, f_t = readOFF(join(sourcefolder, 'off_al2', meshname.format(i_t_)))
            print('size1 :', p_s.shape[0], 'size2 :', p_t.shape[0])

            # evecs
            #self.ev_s = self.input_evecs[i_s]
            #self.ev_t = self.input_evecs[i_t]
            self.ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s_)[:-4]+'.mat'))['target_evecs']
            self.ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t_)[:-4]+'.mat'))['target_evecs']

            # spectral desc
            d_s = self.desc1[i_b]
            d_t = self.desc2[i_b]
        
        elif self.dataset == 'SHREC':
            # file for fetching mesh data
            sourcefolder = '../../../../../../../media/donati/Data1/Datasets/SHREC_r'
            meshname = '{:d}.off'
            
            i_s_ = i_s + 1
            i_t_ = i_t + 1
            
            p_s, f_s = readOFF(join(sourcefolder, 'off_al2',meshname.format(i_s_)))
            p_t, f_t = readOFF(join(sourcefolder, 'off_al2', meshname.format(i_t_)))
            print('size1 :', p_s.shape[0], 'size2 :', p_t.shape[0])

            # evecs
            #self.ev_s = self.input_evecs[i_s]
            #self.ev_t = self.input_evecs[i_t]
            self.ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s_)[:-4]+'.mat'))['target_evecs']
            self.ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t_)[:-4]+'.mat'))['target_evecs']
            
            # spectral desc
            d_s = self.desc1[i_b]
            d_t = self.desc2[i_b]

        else:  # for surreal dfaust
            sourcefolder = '../../../../../../../media/donati/Data1/Datasets/surreal_dfaust'

            # points
            #p_s = input_points[i_s]
            #p_t = input_points[i_t]
            file_s = join(sourcefolder, 'points', self.testfiles[i_s])
            file_t = join(sourcefolder, 'points', self.testfiles[i_t])
            p_s = np.loadtxt(file_s, delimiter=' ', dtype=np.float32)
            p_t = np.loadtxt(file_t, delimiter=' ', dtype=np.float32)

            #evecs
            spc_data_s = sio.loadmat(join(sourcefolder, 'spectral', self.testfiles[i_s][:-4]+'.mat'))
            self.ev_s = spc_data_s['target_evecs']
            spc_data_t = sio.loadmat(join(sourcefolder, 'spectral', self.testfiles[i_t][:-4]+'.mat'))
            self.ev_t = spc_data_t['target_evecs']

            # spectral desc
            d_s = desc1[i_b]
            d_t = desc2[i_b]

            # raw desc
            #of_s = of1[i_b*1000: (i_b+1)*1000]
            #of_t = of2[i_b*1000: (i_b+1)*1000]

            # triangulation
            f = sio.loadmat('../../../../../../../media/donati/Data1/Datasets/surreal_dfaust/TRIV2.mat')['TRIV'] -1
            f_s = f
            f_t = f

#         colmap_s = of_s
#         colmap_t = of_t
        colmap2_s = (self.ev_s[:, :n_eig] @ d_s) # spectral desc
        colmap2_t = (self.ev_t[:, :n_eig] @ d_t) # spectral desc

        #p1 = mp.plot(p_s, f_s, colmap_s[:, 0], return_plot=True) #, s = [1, 2, 0])
        #p2 = mp.plot(p_t, f_t, colmap_t[:, 0], return_plot=True) #, s = [1, 2, 1])
        #p3 = mp.plot(p_s, f, colmap2_s[:, 0], return_plot=True) #, s = [1, 2, 1])
        #p4 = mp.plot(p_t, f, colmap2_t[:, 0], return_plot=True) #, s = [1, 2, 1])

        # Add interactive visulization
        #@mp.interact(level=(0, of_s.shape[1]))
        #def mcf(level=0):
            #p1.update_object(colors = colmap_s[:, level])
            #p2.update_object(colors = colmap_t[:, level])
        #    p3.update_object(colors = colmap2_s[:, level])
        #    p4.update_object(colors = colmap2_t[:, level])
        
        if do_plot:
            p3 = mp.plot(p_s, f_s, colmap2_s[:, 0], return_plot=True) #, s = [1, 2, 1])
            p4 = mp.plot(p_t, f_t, colmap2_t[:, 0], return_plot=True) #, s = [1, 2, 1])
            @mp.interact(level=(0, colmap2_s.shape[1]))
            def mcf(level=0):
               p3.update_object(colors = colmap2_s[:, level])
               p4.update_object(colors = colmap2_t[:, level])
        
        ## make the useful data part of the object
        self.p_s = p_s
        self.p_t = p_t
        self.f_s = f_s
        self.f_t = f_t
        self.i_s = i_s
        self.i_t = i_t
        
        return p_s, f_s, colmap2_s, p_t, f_t, colmap2_t
        

    def viz_map(self):
        
        C = self.maps[self.i_b]
        #plt.imshow(C)
        #B1 = self.ev_s[:, :self.n_eig]
        #B2 = self.ev_t[:, :self.n_eig]
        i_s_ = self.bi1[self.i_b] + 52  #SCAPE
        i_t_ = self.bi2[self.i_b] + 52  #SCAPE
        sourcefolder = '../../../../../../../media/donati/Data1/Datasets/SCAPE_r'
        meshname = 'mesh{:03d}.off'
        
        ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s_)[:-4]+'.mat'))['target_evecs']
        ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t_)[:-4]+'.mat'))['target_evecs']
        
        B1 = ev_s[:, :self.n_eig]
        B2 = ev_t[:, :self.n_eig]
        B1_full = ev_s
        B2_full = ev_t
        #C = maps[i_b]
        #C_gt = np.linalg.lstsq(ev_t[:, :30], ev_s[:, :30], rcond=None)[0]

        T21 = convert_functional_map_to_pointwise_map(C, B1, B2)
        T21_ref, C_ref = refine_pMap_icp(T21, B2, B1)
        
        T21_zo, C_zo = refine_pMap_zo(T21, B2_full, B1_full, self.n_eig)
        #T21_gt = convert_functional_map_to_pointwise_map(np.eye(30), B1, B2)
        
        fig, axs = plt.subplots(2, 2, figsize=(7, 7), constrained_layout=True)
        
        ax1 = axs.flat[0]
        ax1.imshow(C)
        ax1.set_title('fmap before ref')
        ax2 = axs.flat[1]
        ax2.plot(T21)
        ax2.set_title('p2pmap before ref')
        
        ax3 = axs.flat[2]
        ax3.imshow(C_zo)
        ax3.set_title('fmap after ref')
        self.pmap = T21_zo
        ax4 = axs.flat[3]
        ax4.plot(self.pmap)
        ax4.set_title('p2pmap after ref')
        
        self.pmap_raw = T21
        return
    
    def viz_map_on_shapes(self, uv_bool = 0, axis = 0):

        if uv_bool:
            uv_s = viz_textmap(self.p_s)
            uv_t = uv_s[self.pmap]
            p6 = mp.plot(self.p_s, self.f_s, uv = uv_s)
            p5 = mp.plot(self.p_t, self.f_t, uv = uv_t)
        else:
            colmap_s = self.p_s[:, axis] #*p_s[:, 1]
            colmap_t = colmap_s[self.pmap]#[:-1]
            p6 = mp.plot(self.p_s, self.f_s, c = colmap_s)
            p5 = mp.plot(self.p_t, self.f_t, c = colmap_t)
        return
    
    def geodesic_err_on_shapes(self, ref = 1):
        n_s = self.p_s.shape[0]
        n_t = self.p_t.shape[0]
        #i_s, i_t

        # loading geodistance matrix for error assessment

        if self.dataset == 'SCAPE':
            dataset1 = 'SCAPE_r'
            dataset2 = 'SCAPE_remesh'
            meshname = 'mesh{:03d}.off'
        elif self.dataset == 'FAUST':
            dataset1 = 'FAUST_r'
            dataset2 = 'FAUST_remesh'
            meshname = 'tr_reg_{:03d}.off'
        elif self.dataset == 'SHREC':
            dataset1 = 'SHREC_r'
            dataset2 = 'SHREC_remesh'
            meshname = '{:d}.off'
        else :
            raise ValueError('Not possible for this dataset')

        geodist_folder = '../../../../media/donati/Data1/GeoDistanceMatrix/' + dataset2
        MAT_s = sio.loadmat(join(geodist_folder, meshname.format(self.i_s)[:-4]+'.mat'))
        G_s = MAT_s['Gamma']
        SQ_s = MAT_s['SQRarea'][0]
        #print(SQ_s[0])

        vts_folder = '../../../../media/donati/Data1/Datasets/'+dataset1+'/corres/'
        phi_s = np.loadtxt(join(vts_folder, meshname.format(self.i_s)[:-4]+'.vts'), dtype = np.int32) - 1
        phi_t = np.loadtxt(join(vts_folder, meshname.format(self.i_t)[:-4]+'.vts'), dtype = np.int32) - 1
        phi_sym_s = np.loadtxt(join(vts_folder, meshname.format(self.i_s)[:-4]+'.sym.vts'), dtype = np.int32) - 1
        phi_sym_t = np.loadtxt(join(vts_folder, meshname.format(self.i_t)[:-4]+'.sym.vts'), dtype = np.int32) - 1
        #phi_s = np.argsort(np.argsort(phi_s))
        #phi_t = np.argsort(np.argsort(phi_t))

        map_st = phi_t[np.argsort(phi_s)]
        map_ts = phi_s[np.argsort(phi_t)]

        #pmap = T21_ref #map_ts
        #pmap = map_ts
        #pmap = np.arange(len(phi_s))
        #ind21 = np.stack([phi_s, np.abs(phi_s-1)], axis=-1)
        if ref:
            ind21 = np.stack([phi_s, self.pmap[phi_t]], axis=-1)
        else:
            ind21 = np.stack([phi_s, self.pmap_raw[phi_t]], axis=-1)
        
        ind21 = np.ravel_multi_index(ind21.T, dims = [n_s, n_s])
        
        
        errs = np.take(G_s, ind21)/SQ_s
        #errs.shape
        plt.plot(np.sort(errs), np.linspace(0, 0.01, len(errs)))
        print(np.mean(errs))
        
        return
    
    def geodesic_error_on_all(self):
        
        if self.dataset == 'SCAPE':
            dataset1 = 'SCAPE_r'
            dataset2 = 'SCAPE_remesh'
            meshname = 'mesh{:03d}.off'
        elif self.dataset == 'FAUST':
            dataset1 = 'FAUST_r'
            dataset2 = 'FAUST_remesh'
            meshname = 'tr_reg_{:03d}.off'
        else :
            raise('Not possible for this dataset')
        
        errs_tot = []

        for i_b in range(len(self.bi1)):
            #print(i_b)
            i_s = self.bi1[i_b]
            i_t = self.bi2[i_b]
            #print(i_s, i_t)
            i_s_ = i_s
            i_t_ = i_t    
            if self.dataset=='SCAPE':
                #i_s_ += i_s>50
                #i_t_ += i_t>50
                i_s_ += 52
                i_t_ += 52
            
            sourcefolder = '../../../../../../../media/donati/Data1/Datasets/'+ dataset1
            #meshname = 'tr_reg_{:03d}.off'
            #meshname = 'mesh{:03d}.off'

            p_s, f_s = readOFF(join(sourcefolder, 'off',meshname.format(i_s_)))
            p_t, f_t = readOFF(join(sourcefolder, 'off', meshname.format(i_t_)))
            n_s = p_s.shape[0]
            n_t = p_t.shape[0]
            #print('size1 :', n_s, 'size2 :', n_t)

            # evecs
            #ev_s = self.input_evecs[i_s]
            #ev_t = self.input_evecs[i_t]
            ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s_)[:-4]+'.mat'))['target_evecs']
            ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t_)[:-4]+'.mat'))['target_evecs']
            #print(ev_s.shape, ev_t.shape)

            # spectral desc
            d_s = self.desc1[i_b]
            d_t = self.desc2[i_b]

            # loading geodistance matrix for error assessment
            geodist_folder = '../../../../media/donati/Data1/GeoDistanceMatrix/'+dataset2
            MAT_s = sio.loadmat(join(geodist_folder, meshname.format(i_s_)[:-4]+'.mat'))
            G_s = MAT_s['Gamma']
            SQ_s = MAT_s['SQRarea'][0]
            #print(SQ_s[0])

            # vts
            vts_folder = '../../../../media/donati/Data1/Datasets/'+dataset1+'/corres/'
            phi_s = np.loadtxt(join(vts_folder, meshname.format(i_s_)[:-4]+'.vts'), dtype = np.int32) - 1
            phi_t = np.loadtxt(join(vts_folder, meshname.format(i_t_)[:-4]+'.vts'), dtype = np.int32) - 1
            phi_sym_s = np.loadtxt(join(vts_folder, meshname.format(i_s_)[:-4]+'.sym.vts'), dtype = np.int32) - 1
            phi_sym_t = np.loadtxt(join(vts_folder, meshname.format(i_t_)[:-4]+'.sym.vts'), dtype = np.int32) - 1

            # maps
            C = self.maps[i_b]
            n_eig = C.shape[0]
            B1 = ev_s[:, :n_eig]
            B2 = ev_t[:, :n_eig]
            #C = maps[i_b]

            T21 = convert_functional_map_to_pointwise_map(C, B1, B2)
            #T21_ref, C_ref = refine_pMap_icp(T21, B2, B1)
            T21_ref, C_ref = refine_pMap_zo(T21, ev_t, ev_s, n_eig)

            #pmap = T21_ref
            pmap = T21
            ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
            ind21 = np.ravel_multi_index(ind21.T, dims = [n_s, n_s])

            # sym
            ind21_sym = np.stack([phi_sym_s, pmap[phi_t]], axis=-1)
            ind21_sym = np.ravel_multi_index(ind21_sym.T, dims = [n_s, n_s])

            errs = np.take(G_s, ind21)/SQ_s
            errs_sym = np.take(G_s, ind21_sym)/SQ_s

            if np.mean(errs) < np.mean(errs_sym):
                errs_tot += [errs]
                print(i_s_, '->', i_t_, ':', np.mean(errs))
            else:
                errs_tot += [errs_sym]
                print(i_s_, '->', i_t_, '(SYM) :', np.mean(errs_sym))
            
        mean_err = np.mean(np.stack(errs_tot, axis=-1), axis = 1)
        plt.subplot(2, 1, 1)
        plt.plot(np.sort(mean_err), np.linspace(0, 1, len(mean_err)))
        print('\nThe average geodesic error is : ', np.mean(mean_err))
        
        mean_err_bis = np.mean(np.stack(errs_tot, axis=-1), axis = 0)
        plt.subplot(2, 1, 2)
        plt.plot(np.sort(mean_err_bis))
        plt.axhline(y=0.04, color='r', linestyle='-')
        
        j_worst = np.argsort(mean_err_bis)[-2] #find bad maps
        print(j_worst, 'is the worst map')

        per = 0.04
        num_per = np.sum(mean_err_bis<per)/mean_err_bis.shape[0]*100
        print('there are', num_per, '% maps bellow that percentage')
        
        return
    
