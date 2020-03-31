
from viz import *
import os

sourcefolder = '../../../../../../../media/donati/Data1/Datasets/FAUST_r'
meshname = 'tr_reg_{:03d}.off'

n_tr = 52

#log = 'Log_2020-03-16_16-01-48' #'Log_2020-03-12_16-23-10'  # 'Log_2020-02-13_16-09-59' s100
#log = 'Log_2020-03-04_17-26-11'  # surreal5k
#log = 'Log_2020-03-05_08-34-57'  # s5k NoReg
#log = 'Log_2020-03-04_21-01-24'  # s2k
# log = 'Log_2020-03-10_16-51-30'  # 'Log_2020-03-06_09-54-02'  #s500
#log = 'Log_2020-03-10_16-51-51'  # 'Log_2020-03-07_13-34-24'  #s100
#log = 'Log_2020-03-08_07-58-35'  #faust
log = 'Log_2020-03-09_08-35-52'  #SCAPE
#log = 'surreal'+str(n_tr)

epoch = 50  #large dataset (>1000)
epoch = 1000  #small dataset (few hundred)
epoch = 1500  #very small dataset (50)

foldername = join('../test',log, 'FAUST_r/'+str(epoch)+'_epochs')
bi1 = np.load(join(foldername, 'bi1.npy'))
bi2 = np.load(join(foldername, 'bi2.npy'))

# spectral desc
desc1 = np.load(join(foldername, 'desc1.npy'))
desc2 = np.load(join(foldername, 'desc2.npy'))

# maps
maps = np.load(join(foldername, 'fmaps.npy'))

# reshaping
print(desc1.shape)#, len(of1), of1[0].shape)
n_val = desc1.shape[0] * desc1.shape[1]
n_eig = desc1.shape[2]
feat_d = desc1.shape[3]

bi1 = np.reshape(bi1, [n_val, ])
bi2 = np.reshape(bi2, [n_val, ])

desc1 = np.reshape(desc1, [n_val, n_eig, feat_d])
desc2 = np.reshape(desc2, [n_val, n_eig, feat_d])

maps = np.reshape(maps, [n_val, n_eig, n_eig])
print(desc1.shape)

#v.geodesic_error_on_all()

errs_tot = []
errs_tot_ref = []

for i_b in range(len(bi1)):
    #if i_b == 2: break;
    #print(i_b)
    i_s = bi1[i_b] + 80 #number of training shapes
    i_t = bi2[i_b] + 80
    
    p_s, f_s = readOFF(join(sourcefolder, 'off',meshname.format(i_s)))
    p_t, f_t = readOFF(join(sourcefolder, 'off', meshname.format(i_t)))
    n_s = p_s.shape[0]
    n_t = p_t.shape[0]
    #print('size1 :', n_s, 'size2 :', n_t)

    # evecs
    ev_s = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_s)[:-4]+'.mat'))['target_evecs']
    ev_t = sio.loadmat(join(sourcefolder, 'spectral',meshname.format(i_t)[:-4]+'.mat'))['target_evecs']
    #print(ev_s.shape, ev_t.shape)

    # spectral desc
    d_s = desc1[i_b]
    d_t = desc2[i_b]

    # loading geodistance matrix for error assessment
    geodist_folder = '../../../../media/donati/Data1/GeoDistanceMatrix/FAUST_remesh'
    MAT_s = sio.loadmat(join(geodist_folder, meshname.format(i_s)[:-4]+'.mat'))
    G_s = MAT_s['Gamma']
    SQ_s = MAT_s['SQRarea'][0]
    #print(SQ_s[0])

    # vts
    vts_folder = '../../../../media/donati/Data1/Datasets/FAUST_r/corres/'
    phi_s = np.loadtxt(join(vts_folder, meshname.format(i_s)[:-4]+'.vts'), dtype = np.int32) - 1
    phi_t = np.loadtxt(join(vts_folder, meshname.format(i_t)[:-4]+'.vts'), dtype = np.int32) - 1
    phi_sym_s = np.loadtxt(join(vts_folder, meshname.format(i_s)[:-4]+'.sym.vts'), dtype = np.int32) - 1
    phi_sym_t = np.loadtxt(join(vts_folder, meshname.format(i_t)[:-4]+'.sym.vts'), dtype = np.int32) - 1

    # maps
    C = maps[i_b]
    n_eig = C.shape[0]
    B1 = ev_s[:, :n_eig]
    B2 = ev_t[:, :n_eig]

    T21 = convert_functional_map_to_pointwise_map(C, B1, B2)
    #T21_ref, C_ref = refine_pMap_icp(T21, B2, B1)
    T21_ref, C_ref = refine_pMap_zo(T21, ev_t, ev_s, n_eig)

    #pmap = T21_ref
    pmap = T21
    ind21 = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims = [n_s, n_s])

    # sym
    #ind21_sym = np.stack([phi_sym_s, pmap[phi_t]], axis=-1)
    #ind21_sym = np.ravel_multi_index(ind21_sym.T, dims = [n_s, n_s])

    # ref
    pmap = T21_ref
    ind21_ref = np.stack([phi_s, pmap[phi_t]], axis=-1)
    ind21_ref = np.ravel_multi_index(ind21_ref.T, dims = [n_s, n_s])

    errs = np.take(G_s, ind21)/SQ_s
    #errs_sym = np.take(G_s, ind21_sym)/SQ_s
    errs_ref = np.take(G_s, ind21_ref)/SQ_s

    #if np.mean(errs) < np.mean(errs_sym):
    errs_tot += [errs]
    print(i_t, '-->', i_s, ' : ', np.mean(errs), ' ref : ', np.mean(errs_ref))
    #else:
    #errs_tot += [errs_sym]
    errs_tot_ref += [errs_ref]
    #print(i_t, '-->', i_s, ' ref : ', np.mean(errs_ref))

    ### save shapes to matlab for visualization
    params_to_save = {}
    params_to_save['matches'] = T21.astype(np.int32)
    params_to_save['matches_ref'] = T21_ref.astype(np.int32)
    savefolder = join(foldername, 'matches/')
    if not os.path.isdir(savefolder):
        os.mkdir(savefolder)
    sio.savemat(join(savefolder, str(i_t) + '--' + str(i_s) + '.mat'), params_to_save)

mean_err = np.mean(np.stack(errs_tot, axis=-1), axis = 1)
mean_err_ref = np.mean(np.stack(errs_tot_ref, axis=-1), axis = 1)
#iplt.subplot(2, 1, 1)
#plt.plot(np.sort(mean_err), np.linspace(0, 1, len(mean_err)))
print('\nThe average geodesic error is : ', np.mean(mean_err))
print('\nThe average geodesic error (ref) is : ', np.mean(mean_err_ref))

mean_err_bis = np.mean(np.stack(errs_tot, axis=-1), axis = 0)
#plt.subplot(2, 1, 2)
#plt.plot(np.sort(mean_err_bis))
#plt.axhline(y=0.04, color='r', linestyle='-')

j_worst = np.argsort(mean_err_bis)[-2] #find bad maps
print(j_worst, 'is the worst map')

per = 0.04
num_per = np.sum(mean_err_bis<per)/mean_err_bis.shape[0]*100
print('there are', num_per, '% maps bellow ', per*100, '% error')

to_save = np.concatenate(errs_tot, axis = None)
to_save_ref = np.concatenate(errs_tot_ref, axis = None)
print(to_save.shape)
np.save('../../CVPR_Results/FAUST/FAUST_matches_2ours'+str(n_tr), to_save)
np.save('../../CVPR_Results/FAUST/FAUST_matches_2ours_ref'+str(n_tr), to_save_ref)
