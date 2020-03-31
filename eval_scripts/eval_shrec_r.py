import os
from viz import *


#log = 'Log_2020-03-04_17-26-11'  # surreal5k
#log = 'Log_2020-03-05_08-34-57'  # s5k NoReg
#log = 'Log_2020-03-04_21-01-24'  # s2k
#log = 'Log_2020-03-10_16-51-30'  # 'Log_2020-03-06_09-54-02'  #s500
log = 'Log_2020-03-10_16-51-51'  # 'Log_2020-03-07_13-34-24'  #s100
#log = 'Log_2020-03-08_07-58-35'  #faust
#log = 'Log_2020-03-09_08-35-52'  #SCAPE
#log = 'surreal'+str(n_tr)

log = 'Log_2020-03-29_20-41-26'

#epoch = 50  #large dataset (>1000)
epoch = 1000  #small dataset (few hundred)
#epoch = 1500  #very small dataset (50)

n_tr = 500

foldername = join('../test',log, 'SHREC_r/'+str(epoch)+'_epochs')
bi1 = np.load(join(foldername, 'bi1.npy'))
bi2 = np.load(join(foldername, 'bi2.npy'))

sourcefolder = '../../../../../../../media/donati/Data1/Datasets/SHREC_r'
meshname = '{:d}.off'

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

bi1 = np.reshape(bi1, [n_val,])
bi2 = np.reshape(bi2, [n_val,])

desc1 = np.reshape(desc1, [n_val, n_eig, feat_d])
desc2 = np.reshape(desc2, [n_val, n_eig, feat_d])

maps = np.reshape(maps, [n_val, n_eig, n_eig])
print(desc1.shape)

#v.geodesic_error_on_all()

errs_tot = []
errs_tot_ref = []

for i_b in range(len(bi1)):
    #print(i_b)
    #if i_b == 2: break
    i_s = bi1[i_b] + 1 #adapt to notation
    i_t = bi2[i_b] + 1

    p_s, f_s = readOFF(join(sourcefolder, 'off_al2',meshname.format(i_s)))
    p_t, f_t = readOFF(join(sourcefolder, 'off_al2', meshname.format(i_t)))
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
    geodist_folder = '../../../../media/donati/Data1/GeoDistanceMatrix/SHREC_remesh'
    MAT_s = sio.loadmat(join(geodist_folder, meshname.format(i_s)[:-4]+'.mat'))
    G_s = MAT_s['Gamma']
    SQ_s = MAT_s['sqrt_area'][0].todense()
    #print(SQ_s[0])

    # groundtruths
    gt_folder = '../../../../media/donati/Data1/Datasets/SHREC_r/groundtruth/'
    gt_name = (gt_folder+str(i_t)+'_'+str(i_s)+'.map')#.format(i,j)
    gt_map = np.loadtxt(gt_name, dtype = np.int32) - 1
   
    gt_folder2 = '../../../../media/donati/Data1/Datasets/SHREC_r/FARMgt_remeshed_txt/'
    gt_name2 = (gt_folder2+str(i_t)+'_'+str(i_s)+'.txt')#.format(i,j)
    gt_map2 = np.loadtxt(gt_name2, dtype = np.int32) - 1
 
    # maps
    C = maps[i_b]
    n_eig = C.shape[0]
    B1 = ev_s[:, :n_eig]
    B2 = ev_t[:, :n_eig]

    T21 = convert_functional_map_to_pointwise_map(C, B1, B2)
    #T21_ref, C_ref = refine_pMap_icp(T21, B2, B1)
    #T21_ref, C_ref = refine_pMap_zo(T21, ev_t, ev_s, n_eig)
    
    #pmap = gt_map2
    #pmap = T21_ref
    pmap = T21
    ind21 = np.stack([gt_map, pmap], axis=-1)
    ind21 = np.ravel_multi_index(ind21.T, dims = [n_s, n_s])

    # sym
    #ind21_sym = np.stack([phi_sym_s, pmap[phi_t]], axis=-1)
    #ind21_sym = np.ravel_multi_index(ind21_sym.T, dims = [n_s, n_s])

    #pmap = T21_ref
    #ind21_ref = np.stack([gt_map, pmap], axis=-1)
    #ind21_ref = np.stack([gt_map2, pmap], axis=-1)
    #ind21_ref = np.ravel_multi_index(ind21_ref.T, dims = [n_s, n_s])

    errs = np.take(G_s, ind21)/SQ_s
    #errs_ref = np.take(G_s, ind21_ref)/SQ_s
    #errs_sym = np.take(G_s, ind21_sym)/SQ_s

    #if np.mean(errs) < np.mean(errs_sym):
    
    errs = np.reshape(np.array(errs), errs.shape[1])
    #errs_ref = np.reshape(np.array(errs_ref), errs_ref.shape[1])

    #print(errs)
   
    errs_tot += [errs]
    #errs_tot_ref += [errs_ref]
    #print(len(errs_tot))
    print(i_t, '-->', i_s, ' : ', np.mean(errs))#, ' ref : ', np.mean(errs_ref))
    #else:
    #errs_tot += [errs_sym]
    #print(i_t, '-->', i_s, '(SYM) : ', np.mean(errs_sym))
    
    ### save shapes to matlab for visualization
    #params_to_save = {}
    #params_to_save['matches'] = T21.astype(np.int32)
    #params_to_save['matches_ref'] = T21_ref.astype(np.int32)
    #savefolder = join(foldername, 'matches/')
    #if not os.path.isdir(savefolder):
    #    os.mkdir(savefolder)
    #sio.savemat(join(savefolder, str(i_t) + '--' + str(i_s) + '.mat'), params_to_save)

tot_matches = np.concatenate(errs_tot, axis = 0)
#tot_matches_ref = np.concatenate(errs_tot_ref, axis = 0)
#np.savetxt('5k_shrec.txt', tot_matches)

mean_err = np.mean(tot_matches)
#mean_err_ref = np.mean(tot_matches_ref)
#plt.subplot(2, 1, 1)
#plt.plot(np.sort(mean_err), np.linspace(0, 1, len(mean_err)))
print('\nThe average geodesic error is : ', mean_err)
#print('\nThe average geodesic error (ref) is : ', mean_err_ref)
#print(errs_tot)
#print(to_save)

#to_save = np.concatenate(errs_tot, axis = None)
#to_save_ref = np.concatenate(errs_tot_ref, axis = None)
#print(to_save.shape)
#np.save('shrec_matches_ours'+str(n_tr), to_save)
#np.save('shrec_matches_ours_ref'+str(n_tr), to_save_ref)


to_save = np.concatenate(errs_tot, axis = None)
to_save_ref = np.concatenate(errs_tot_ref, axis = None)
#print(to_save.shape)
#np.save('../../CVPR_Results/SHREC/SHREC_matches_2ours'+str(n_tr), to_save)
#np.save('../../CVPR_Results/SHREC/SHREC_matches_2ours_ref'+str(n_tr), to_save_ref)

#print('\nThe average geodesic error is : ', np.mean(to_save))
#print('\nThe average geodesic error (ref) is : ', np.mean(to_save_ref))


#mean_err_bis = np.mean(np.stack(errs_tot, axis=-1), axis = 0)
#plt.subplot(2, 1, 2)
#plt.plot(np.sort(mean_err_bis))
#plt.axhline(y=0.04, color='r', linestyle='-')

#j_worst = np.argsort(mean_err_bis)[-2] #find bad maps
#print(j_worst, 'is the worst map')

#per = 0.04
#num_per = np.sum(mean_err_bis<per)/mean_err_bis.shape[0]*100
#print('there are', num_per, '% maps bellow ', per*100, '% error')
