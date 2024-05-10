import numpy as np
import get_data_fun as gd
import h5py
from tqdm import tqdm



def generate_levi_civita():
    levi_civita = np.zeros((3,3,3))
    for ii in range(3):
        for jj in range(3):
            for kk in range(3):
                if (ii+jj+kk != 3) or (ii==1 and jj==1 and kk==1):
                     continue
                elif (jj-ii == 1 or ii-jj ==2) and (kk-jj == 1 or ii-kk == 1):
                    levi_civita[ii,jj,kk] = 1
                else:
                    levi_civita[ii,jj,kk] = -1
                 
    return levi_civita


def conditional_avg(field, condition, threshold=0):
    cond_sum = np.sum(field*condition, axis=(1,2))
    cond_count = np.sum(np.heaviside(field*condition,0), axis=(1,2))
    cond_count[cond_count==0] = 1 # shouldn't change result but avoids /0
    cond_avg = cond_sum/cond_count
    return cond_avg


def calculate_enstrophy(start,
                        end,
                        step,
                        file_read='./P125_21pi_vu/P125_21pi_vu',
                        file_grad='./P125_21pi_vu/grad/P125_21pi_vu'):
    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad)
    normdata.geom_param(start,1,1,1)
    levi_civita = generate_levi_civita()

    for ii in range(start, end, step):
        G = normdata.read_gradients(ii)
        vorticity = np.einsum('ijk,kjyzx->iyzx', levi_civita, G)
        enstrophy = 0.5*np.einsum('iyzx,iyzx->yzx', vorticity, vorticity)
        file = h5py.File(file_grad+f'.{ii}.grad', 'r+')
        file.create_dataset('enstrophy', data=enstrophy)
        
        
def calculate_vorticity(start,
                        end,
                        step,
                        file_read='./P125_21pi_vu/P125_21pi_vu',
                        file_grad='./P125_21pi_vu/grad/P125_21pi_vu',
                        file_vort='./P125_21pi_vu/vort/P125_21pi_vu'):
    
    normdata = gd.get_data_norm(file_read=file_read,
                                file_grad=file_grad)
    normdata.geom_param(start,1,1,1)
    levi_civita = generate_levi_civita()

    for ii in tqdm(range(start, end, step)):
        G = normdata.read_gradients(ii)
        vorticity = np.einsum('ijk,kjyzx->iyzx', levi_civita, G)
        file = h5py.File(file_grad+f'.{ii}.vort', 'r+')
        file.create_dataset('omega', data=vorticity)
        
        
        
def calc_enstrophy_shares(start,
                          end,
                          step,
                          dataset='P125_21pi_vu',
                          root='./',
                          root_grad='./',
                          root_structures='./'):
    
    normdata = gd.get_data_norm(file_read=root+dataset+'/'+dataset,
                                file_grad=root+dataset+'/grad/'+dataset)
    normdata.geom_param(start,1,1,1)
    volume_total = normdata.voltot
    
    path_hunt = root_structures+dataset+'_hunt/'+dataset
    path_chong = root_structures+dataset+'_chong/'+dataset
    path_grad = root_grad+dataset+'/grad/'+dataset
    path_enstrophy = root_structures+dataset+'_enstrophy/'
    
    enstrophy_mean = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_hunt = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_chong = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_hunt = np.zeros((1, int(np.floor((end-start)/step)+1)))
    # volume_ratio_chong = np.zeros((1, int(np.floor((end-start)/step)+1)))
    volume_ratio_hunt = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_chong = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    
    
    for ii in range(start, end, step):
        file_hunt = h5py.File(path_hunt+f'.{ii}.h5.hunt', 'r+')
        file_chong = h5py.File(path_chong+f'.{ii}.h5.chong', 'r+')
        file_grad = h5py.File(path_grad+f'.{ii}.grad', 'r+')
        
        enstrophy = np.array(file_grad['enstrophy'])
        binary_hunt = np.array(file_hunt['Qs'])
        vol_hunt = np.array(file_hunt['vol'])
        binary_chong = np.array(file_chong['Qs'])
        vol_chong = np.array(file_chong['vol'])
        
        enstrophy_mean[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy, axis=(1,2))
        enstrophy_hunt[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_hunt, axis=(1,2))
        enstrophy_chong[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_chong, axis=(1,2))
        # enstrophy_hunt[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_hunt)
        # enstrophy_chong[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_chong)
        # volume_ratio_hunt[0, int(np.floor((ii-start)/step))] = np.sum(vol_hunt)/volume_total
        # volume_ratio_chong[0, int(np.floor((ii-start)/step))] = np.sum(vol_chong)/volume_total 
        volume_ratio_hunt[:, int(np.floor((ii-start)/step))] = np.mean(binary_hunt, axis=(1,2))
        volume_ratio_chong[:, int(np.floor((ii-start)/step))] = np.mean(binary_chong, axis=(1,2))
        
    enstrophy_mean_all_fields = np.mean(enstrophy_mean, axis=1)
    enstrophy_hunt_all_fields = np.mean(enstrophy_hunt, axis=1)
    enstrophy_chong_all_fields = np.mean(enstrophy_chong, axis=1)
    # mean_vol_ratio_hunt = np.mean(volume_ratio_hunt)
    # mean_vol_ratio_chong = np.mean(volume_ratio_chong)
    mean_vol_ratio_hunt = np.mean(volume_ratio_hunt, axis=1)
    mean_vol_ratio_chong = np.mean(volume_ratio_chong, axis=1)
    
    
    file_enstrophy = h5py.File(path_enstrophy+dataset+f'.{start},{end},{step}.h5.ens', 'w')
    file_enstrophy.create_dataset('mean', data=enstrophy_mean_all_fields)
    file_enstrophy.create_dataset('hunt', data=enstrophy_hunt_all_fields)
    file_enstrophy.create_dataset('chong', data=enstrophy_chong_all_fields)
    file_enstrophy.create_dataset('vol_hunt', data=mean_vol_ratio_hunt)
    file_enstrophy.create_dataset('vol_chong', data=mean_vol_ratio_chong)
    
        
        
        
            
            
