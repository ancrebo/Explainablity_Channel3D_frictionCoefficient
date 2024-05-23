import numpy as np
import get_data_fun as gd
import h5py
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl



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
        file = h5py.File(file_vort+f'.{ii}.vort', 'w')
        file.create_dataset('omega', data=vorticity)
        
        
        
def wall_attached_detached(segmentation_Q, event, y_min_plus):
    binary_Q2_wa = np.zeros(segmentation_Q.shape)
    binary_Q2_wd = np.zeros(segmentation_Q.shape)
    binary_Q4_wa = np.zeros(segmentation_Q.shape)
    binary_Q4_wd = np.zeros(segmentation_Q.shape)
    
    for ii, event in enumerate(event):
        if event == 2:
            if y_min_plus[ii] < 20:
                binary_Q2_wa[segmentation_Q==ii+1] = 1
            else:
                binary_Q2_wd[segmentation_Q==ii+1] = 1
        elif event == 4:
            if y_min_plus[ii] < 20:
                binary_Q4_wa[segmentation_Q==ii+1] = 1
            else:
                binary_Q4_wd[segmentation_Q==ii+1] = 1
                
    return binary_Q2_wa, binary_Q2_wd, binary_Q4_wa, binary_Q4_wd
        
        
        
def calc_enstrophy_shares(start,
                          end,
                          step,
                          dataset='P125_21pi_vu',
                          root='./',
                          root_grad='./',
                          root_structures='./',
                          root_Q='./'):
    
    normdata = gd.get_data_norm(file_read=root+dataset+'/'+dataset,
                                file_grad=root_grad+dataset+'/grad/'+dataset)
    normdata.geom_param(start,1,1,1)
    # volume_total = normdata.voltot
    
    path_Q = root_Q+dataset+'_Q_divide/'+dataset
    # path_streak = root_structures+dataset+'_streak/'+dataset
    # path_streak_hv = root_structures+dataset+'_streak_high_vel/'+dataset
    # path_hunt = root_structures+dataset+'_hunt/'+dataset
    # path_chong = root_structures+dataset+'_chong/'+dataset
    path_grad = root_grad+dataset+'/grad/'+dataset
    path_enstrophy = root_structures+dataset+'_enstrophy/'
    
    # enstrophy_mean = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # enstrophy_Q = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q1 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q2 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q3 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q4 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q2_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q2_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q4_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    enstrophy_Q4_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # enstrophy_streak = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # enstrophy_streak_hv = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # enstrophy_hunt = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # enstrophy_chong = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_hunt = np.zeros((1, int(np.floor((end-start)/step)+1)))
    # volume_ratio_chong = np.zeros((1, int(np.floor((end-start)/step)+1)))
    # volume_ratio_Q = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q1 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q2 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q3 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q4 = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q2_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q2_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q4_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q4_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_streak = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_streak_hv = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_hunt = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_chong = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    
    
    for ii in tqdm(range(start, end, step)):
        file_Q = h5py.File(path_Q+f'.{ii}.h5.Q', 'r')
        # file_streak = h5py.File(path_streak+f'.{ii}.h5.streak', 'r+')
        # file_streak_hv = h5py.File(path_streak_hv+f'.{ii}.h5.streak_high_vel', 'r+')
        # file_hunt = h5py.File(path_hunt+f'.{ii}.h5.hunt', 'r+')
        # file_chong = h5py.File(path_chong+f'.{ii}.h5.chong', 'r+')
        file_grad = h5py.File(path_grad+f'.{ii}.grad', 'r+')
        
        enstrophy = np.array(file_grad['enstrophy'])
        binary_Q = np.array(file_Q['Qs'])
        events_Q = np.array(file_Q['Qs_event'])
        binary_Q1 = np.zeros(binary_Q.shape)
        binary_Q2 = np.zeros(binary_Q.shape)
        binary_Q3 = np.zeros(binary_Q.shape)
        binary_Q4 = np.zeros(binary_Q.shape)
        binary_Q1[events_Q==1] = 1
        binary_Q2[events_Q==2] = 1
        binary_Q3[events_Q==3] = 1
        binary_Q4[events_Q==4] = 1
        
        segmentation_Q = np.array(file_Q['Qs_segment'])
        event = np.array(file_Q['event'])
        y_min = np.array(file_Q['y_min'])
        y_min_plus = (1-np.abs(y_min))*normdata.vtau/normdata.ny
        
        binary_Q2_wa, binary_Q2_wd, binary_Q4_wa, binary_Q4_wd = wall_attached_detached(segmentation_Q, 
                                                                                        event, 
                                                                                        y_min_plus)
        
        # vol_Q = np.array(file_Q['vol'])
        # binary_streak = np.array(file_streak['Qs'])
        # # vol_streak = np.array(file_streak['vol'])
        # binary_streak_hv = np.array(file_streak_hv['Qs'])
        # # vol_streak_hv = np.array(file_streak_hv['vol'])
        # binary_hunt = np.array(file_hunt['Qs'])
        # # vol_hunt = np.array(file_hunt['vol'])
        # binary_chong = np.array(file_chong['Qs'])
        # vol_chong = np.array(file_chong['vol'])
        
        # enstrophy_mean[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy, axis=(1,2))
        # enstrophy_Q[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q, axis=(1,2))
        enstrophy_Q1[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q1, axis=(1,2))
        enstrophy_Q2[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q2, axis=(1,2))
        enstrophy_Q3[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q3, axis=(1,2))
        enstrophy_Q4[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q4, axis=(1,2))
        enstrophy_Q2_wa[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q2_wa, axis=(1,2))
        enstrophy_Q2_wd[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q2_wd, axis=(1,2))
        enstrophy_Q4_wa[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q4_wa, axis=(1,2))
        enstrophy_Q4_wd[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_Q4_wd, axis=(1,2))
        # enstrophy_streak[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_streak, axis=(1,2))
        # enstrophy_streak_hv[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_streak_hv, axis=(1,2))
        # enstrophy_hunt[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_hunt, axis=(1,2))
        # enstrophy_chong[:, int(np.floor((ii-start)/step))] = np.mean(enstrophy*binary_chong, axis=(1,2))
        # enstrophy_hunt[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_hunt)
        # enstrophy_chong[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_chong)
        # volume_ratio_hunt[0, int(np.floor((ii-start)/step))] = np.sum(vol_hunt)/volume_total
        # volume_ratio_chong[0, int(np.floor((ii-start)/step))] = np.sum(vol_chong)/volume_total 
        # volume_ratio_Q[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q, axis=(1,2))
        volume_ratio_Q1[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q1, axis=(1,2))
        volume_ratio_Q2[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2, axis=(1,2))
        volume_ratio_Q3[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q3, axis=(1,2))
        volume_ratio_Q4[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4, axis=(1,2))
        volume_ratio_Q2_wa[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2_wa, axis=(1,2))
        volume_ratio_Q2_wd[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2_wd, axis=(1,2))
        volume_ratio_Q4_wa[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4_wa, axis=(1,2))
        volume_ratio_Q4_wd[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4_wd, axis=(1,2))
        # volume_ratio_streak[:, int(np.floor((ii-start)/step))] = np.mean(binary_streak, axis=(1,2))
        # volume_ratio_streak_hv[:, int(np.floor((ii-start)/step))] = np.mean(binary_streak_hv, axis=(1,2))
        # volume_ratio_hunt[:, int(np.floor((ii-start)/step))] = np.mean(binary_hunt, axis=(1,2))
        # volume_ratio_chong[:, int(np.floor((ii-start)/step))] = np.mean(binary_chong, axis=(1,2))
        
    # enstrophy_mean_all_fields = np.mean(enstrophy_mean, axis=1)
    # enstrophy_Q_all_fields = np.mean(enstrophy_Q, axis=1)
    enstrophy_Q1_all_fields = np.mean(enstrophy_Q1, axis=1)
    enstrophy_Q2_all_fields = np.mean(enstrophy_Q2, axis=1)
    enstrophy_Q3_all_fields = np.mean(enstrophy_Q3, axis=1)
    enstrophy_Q4_all_fields = np.mean(enstrophy_Q4, axis=1)
    enstrophy_Q2_wa_all_fields = np.mean(enstrophy_Q2_wa, axis=1)
    enstrophy_Q2_wd_all_fields = np.mean(enstrophy_Q2_wd, axis=1)
    enstrophy_Q4_wa_all_fields = np.mean(enstrophy_Q4_wa, axis=1)
    enstrophy_Q4_wd_all_fields = np.mean(enstrophy_Q4_wd, axis=1)
    # enstrophy_streak_all_fields = np.mean(enstrophy_streak, axis=1)
    # enstrophy_streak_hv_all_fields = np.mean(enstrophy_streak_hv, axis=1)
    # enstrophy_hunt_all_fields = np.mean(enstrophy_hunt, axis=1)
    # enstrophy_chong_all_fields = np.mean(enstrophy_chong, axis=1)
    # mean_vol_ratio_hunt = np.mean(volume_ratio_hunt)
    # mean_vol_ratio_chong = np.mean(volume_ratio_chong)
    # mean_vol_ratio_Q = np.mean(volume_ratio_Q, axis=1)
    mean_vol_ratio_Q1 = np.mean(volume_ratio_Q1, axis=1)
    mean_vol_ratio_Q2 = np.mean(volume_ratio_Q2, axis=1)
    mean_vol_ratio_Q3 = np.mean(volume_ratio_Q3, axis=1)
    mean_vol_ratio_Q4 = np.mean(volume_ratio_Q4, axis=1)
    mean_vol_ratio_Q2_wa = np.mean(volume_ratio_Q2_wa, axis=1)
    mean_vol_ratio_Q2_wd = np.mean(volume_ratio_Q2_wd, axis=1)
    mean_vol_ratio_Q4_wa = np.mean(volume_ratio_Q4_wa, axis=1)
    mean_vol_ratio_Q4_wd = np.mean(volume_ratio_Q4_wd, axis=1)
    # mean_vol_ratio_streak = np.mean(volume_ratio_streak, axis=1)
    # mean_vol_ratio_streak_hv = np.mean(volume_ratio_streak_hv, axis=1)
    # mean_vol_ratio_hunt = np.mean(volume_ratio_hunt, axis=1)
    # mean_vol_ratio_chong = np.mean(volume_ratio_chong, axis=1)
    
    
    file_enstrophy = h5py.File(path_enstrophy+dataset+f'.{start},{end},{step}.h5.ens', 'r+')
    # file_enstrophy.create_dataset('mean', data=enstrophy_mean_all_fields)
    # file_enstrophy.create_dataset('Q-structure', data=enstrophy_Q_all_fields)
    file_enstrophy.create_dataset('Q1', data=enstrophy_Q1_all_fields)
    file_enstrophy.create_dataset('Q2', data=enstrophy_Q2_all_fields)
    file_enstrophy.create_dataset('Q3', data=enstrophy_Q3_all_fields)
    file_enstrophy.create_dataset('Q4', data=enstrophy_Q4_all_fields)
    file_enstrophy.create_dataset('Q2_wa', data=enstrophy_Q2_wa_all_fields)
    file_enstrophy.create_dataset('Q2_wd', data=enstrophy_Q2_wd_all_fields)
    file_enstrophy.create_dataset('Q4_wa', data=enstrophy_Q4_wa_all_fields)
    file_enstrophy.create_dataset('Q4_wd', data=enstrophy_Q4_wd_all_fields)
    # file_enstrophy.create_dataset('streak', data=enstrophy_streak_all_fields)
    # file_enstrophy.create_dataset('streak_high_vel', data=enstrophy_streak_hv_all_fields)
    # file_enstrophy.create_dataset('hunt', data=enstrophy_hunt_all_fields)
    # file_enstrophy.create_dataset('chong', data=enstrophy_chong_all_fields)
    # file_enstrophy.create_dataset('vol_Q-structure', data=mean_vol_ratio_Q)
    file_enstrophy.create_dataset('vol_Q1', data=mean_vol_ratio_Q1)
    file_enstrophy.create_dataset('vol_Q2', data=mean_vol_ratio_Q2)
    file_enstrophy.create_dataset('vol_Q3', data=mean_vol_ratio_Q3)
    file_enstrophy.create_dataset('vol_Q4', data=mean_vol_ratio_Q4)
    file_enstrophy.create_dataset('vol_Q2_wa', data=mean_vol_ratio_Q2_wa)
    file_enstrophy.create_dataset('vol_Q2_wd', data=mean_vol_ratio_Q2_wd)
    file_enstrophy.create_dataset('vol_Q4_wa', data=mean_vol_ratio_Q4_wa)
    file_enstrophy.create_dataset('vol_Q4_wd', data=mean_vol_ratio_Q4_wd)
    # file_enstrophy.create_dataset('vol_streak', data=mean_vol_ratio_streak)
    # file_enstrophy.create_dataset('vol_streak_high_vel', data=mean_vol_ratio_streak_hv)
    # file_enstrophy.create_dataset('vol_hunt', data=mean_vol_ratio_hunt)
    # file_enstrophy.create_dataset('vol_chong', data=mean_vol_ratio_chong)
    
    
def calc_intensity_shares(start,
                          end,
                          step,
                          dataset='P125_21pi_vu',
                          root='./',
                          root_grad='./',
                          root_structures='./',
                          root_Q='./'):
    
    normdata = gd.get_data_norm(file_read=root+dataset+'/'+dataset,
                                file_grad=root_grad+dataset+'/grad/'+dataset)
    normdata.geom_param(start,1,1,1)
    try:
        normdata.read_Umean()
    except:
        normdata.calc_Umean(start,end)
    # volume_total = normdata.voltot
    
    path_Q = root_Q+dataset+'_Q_divide/'+dataset
    # path_streak = root_structures+dataset+'_streak/'+dataset
    # path_streak_hv = root_structures+dataset+'_streak_high_vel/'+dataset
    # path_hunt = root_structures+dataset+'_hunt/'+dataset
    # path_chong = root_structures+dataset+'_chong/'+dataset
    # path_grad = root_grad+dataset+'/grad/'+dataset
    path_intensity = root_structures+dataset+'_intensity/'
    
    # intensity_square_mean = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_Q = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_Q1 = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_Q2 = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_Q3 = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_Q4 = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    intensity_square_Q2_wa = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    intensity_square_Q2_wd = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    intensity_square_Q4_wa = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    intensity_square_Q4_wd = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_streak = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_streak_hv = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_hunt = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # intensity_square_chong = np.zeros((4, normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_hunt = np.zeros((1, int(np.floor((end-start)/step)+1)))
    # volume_ratio_chong = np.zeros((1, int(np.floor((end-start)/step)+1)))
    # volume_ratio_Q = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q2_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q2_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q4_wa = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    volume_ratio_Q4_wd = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_streak = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_streak_hv = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_hunt = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    # volume_ratio_chong = np.zeros((normdata.my, int(np.floor((end-start)/step)+1)))
    
    
    for ii in tqdm(range(start, end, step)):
        file_Q = h5py.File(path_Q+f'.{ii}.h5.Q', 'r')
        # file_streak = h5py.File(path_streak+f'.{ii}.h5.streak', 'r+')
        # file_streak_hv = h5py.File(path_streak_hv+f'.{ii}.h5.streak_high_vel', 'r+')
        # file_hunt = h5py.File(path_hunt+f'.{ii}.h5.hunt', 'r+')
        # file_chong = h5py.File(path_chong+f'.{ii}.h5.chong', 'r+')
        
        uu,vv,ww = normdata.read_velocity(ii)
        intensity_square = np.zeros((4, normdata.my, normdata.mz, normdata.mx))
        intensity_square[0,:,:,:] = uu*uu
        intensity_square[1,:,:,:] = vv*vv
        intensity_square[2,:,:,:] = ww*ww
        intensity_square[3,:,:,:] = uu*vv
        
        
        # binary_Q = np.array(file_Q['Qs'])
        # events_Q = np.array(file_Q['Qs_event'])
        # binary_Q1 = np.zeros(binary_Q.shape)
        # binary_Q2 = np.zeros(binary_Q.shape)
        # binary_Q3 = np.zeros(binary_Q.shape)
        # binary_Q4 = np.zeros(binary_Q.shape)
        # binary_Q1[events_Q==1] = 1
        # binary_Q2[events_Q==2] = 1
        # binary_Q3[events_Q==3] = 1
        # binary_Q4[events_Q==4] = 1
        
        
        segmentation_Q = np.array(file_Q['Qs_segment'])
        event = np.array(file_Q['event'])
        y_min = np.array(file_Q['y_min'])
        y_min_plus = (1-np.abs(y_min))*normdata.vtau/normdata.ny
        
        binary_Q2_wa, binary_Q2_wd, binary_Q4_wa, binary_Q4_wd = wall_attached_detached(segmentation_Q, 
                                                                                        event, 
                                                                                        y_min_plus)
        
        # # vol_Q = np.array(file_Q['vol'])
        # binary_streak = np.array(file_streak['Qs'])
        # # vol_streak = np.array(file_streak['vol'])
        # binary_streak_hv = np.array(file_streak_hv['Qs'])
        # # vol_streak_hv = np.array(file_streak_hv['vol'])
        # binary_hunt = np.array(file_hunt['Qs'])
        # # vol_hunt = np.array(file_hunt['vol'])
        # binary_chong = np.array(file_chong['Qs'])
        # # vol_chong = np.array(file_chong['vol'])
        
        # intensity_square_mean[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square, axis=(2,3))
        # intensity_square_Q[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q, axis=(2,3))
        # intensity_square_Q1[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q1, axis=(2,3))
        # intensity_square_Q2[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q2, axis=(2,3))
        # intensity_square_Q3[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q3, axis=(2,3))
        # intensity_square_Q4[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q4, axis=(2,3))
        intensity_square_Q2_wa[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q2_wa, axis=(2,3))
        intensity_square_Q2_wd[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q2_wd, axis=(2,3))
        intensity_square_Q4_wa[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q4_wa, axis=(2,3))
        intensity_square_Q4_wd[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_Q4_wd, axis=(2,3))
        # intensity_square_streak[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_streak, axis=(2,3))
        # intensity_square_streak_hv[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_streak_hv, axis=(2,3))
        # intensity_square_hunt[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_hunt, axis=(2,3))
        # intensity_square_chong[:, :, int(np.floor((ii-start)/step))] = np.mean(intensity_square*binary_chong, axis=(2,3))
        # enstrophy_hunt[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_hunt)
        # enstrophy_chong[:, int(np.floor((ii-start)/step))] = conditional_avg(enstrophy, binary_chong)
        # volume_ratio_hunt[0, int(np.floor((ii-start)/step))] = np.sum(vol_hunt)/volume_total
        # volume_ratio_chong[0, int(np.floor((ii-start)/step))] = np.sum(vol_chong)/volume_total 
        # volume_ratio_Q[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q, axis=(1,2))
        # volume_ratio_Q1[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q1, axis=(1,2))
        # volume_ratio_Q2[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2, axis=(1,2))
        # volume_ratio_Q3[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q3, axis=(1,2))
        # volume_ratio_Q4[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4, axis=(1,2))
        volume_ratio_Q2_wa[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2_wa, axis=(1,2))
        volume_ratio_Q2_wd[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q2_wd, axis=(1,2))
        volume_ratio_Q4_wa[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4_wa, axis=(1,2))
        volume_ratio_Q4_wd[:, int(np.floor((ii-start)/step))] = np.mean(binary_Q4_wd, axis=(1,2))
        # volume_ratio_streak[:, int(np.floor((ii-start)/step))] = np.mean(binary_streak, axis=(1,2))
        # volume_ratio_streak_hv[:, int(np.floor((ii-start)/step))] = np.mean(binary_streak_hv, axis=(1,2))
        # volume_ratio_hunt[:, int(np.floor((ii-start)/step))] = np.mean(binary_hunt, axis=(1,2))
        # volume_ratio_chong[:, int(np.floor((ii-start)/step))] = np.mean(binary_chong, axis=(1,2))
        
    # intensity_mean_all_fields = np.mean(intensity_square_mean, axis=2)
    # intensity_Q_all_fields = np.mean(intensity_square_Q, axis=2)
    # intensity_Q1_all_fields = np.mean(intensity_square_Q1, axis=2)
    # intensity_Q2_all_fields = np.mean(intensity_square_Q2, axis=2)
    # intensity_Q3_all_fields = np.mean(intensity_square_Q3, axis=2)
    # intensity_Q4_all_fields = np.mean(intensity_square_Q4, axis=2)
    intensity_Q2_wa_all_fields = np.mean(intensity_square_Q2_wa, axis=2)
    intensity_Q2_wd_all_fields = np.mean(intensity_square_Q2_wd, axis=2)
    intensity_Q4_wa_all_fields = np.mean(intensity_square_Q4_wa, axis=2)
    intensity_Q4_wd_all_fields = np.mean(intensity_square_Q4_wd, axis=2)
    # intensity_streak_all_fields = np.mean(intensity_square_streak, axis=2)
    # intensity_streak_hv_all_fields = np.mean(intensity_square_streak_hv, axis=2)
    # intensity_hunt_all_fields = np.mean(intensity_square_hunt, axis=2)
    # intensity_chong_all_fields = np.mean(intensity_square_chong, axis=2)
    
    # intensity_mean_all_fields[:3,:] = np.sqrt(intensity_mean_all_fields[:3,:])
    # intensity_Q_all_fields[:3,:] = np.sqrt(intensity_Q_all_fields[:3,:])
    # intensity_Q1_all_fields[:3,:] = np.sqrt(intensity_Q1_all_fields[:3,:])
    # intensity_Q2_all_fields[:3,:] = np.sqrt(intensity_Q2_all_fields[:3,:])
    # intensity_Q3_all_fields[:3,:] = np.sqrt(intensity_Q3_all_fields[:3,:])
    # intensity_Q4_all_fields[:3,:] = np.sqrt(intensity_Q4_all_fields[:3,:])
    intensity_Q2_wa_all_fields[:3,:] = np.sqrt(intensity_Q2_wa_all_fields[:3,:])
    intensity_Q2_wd_all_fields[:3,:] = np.sqrt(intensity_Q2_wd_all_fields[:3,:])
    intensity_Q4_wa_all_fields[:3,:] = np.sqrt(intensity_Q4_wa_all_fields[:3,:])
    intensity_Q4_wd_all_fields[:3,:] = np.sqrt(intensity_Q4_wd_all_fields[:3,:])
    # intensity_streak_all_fields[:3,:] = np.sqrt(intensity_streak_all_fields[:3,:])
    # intensity_streak_hv_all_fields[:3,:] = np.sqrt(intensity_streak_hv_all_fields[:3,:])
    # intensity_hunt_all_fields[:3,:] = np.sqrt(intensity_hunt_all_fields[:3,:])
    # intensity_chong_all_fields[:3,:] = np.sqrt(intensity_chong_all_fields[:3,:])
    # mean_vol_ratio_hunt = np.mean(volume_ratio_hunt)
    # mean_vol_ratio_chong = np.mean(volume_ratio_chong)
    # mean_vol_ratio_Q = np.mean(volume_ratio_Q, axis=1)
    # mean_vol_ratio_Q1 = np.mean(volume_ratio_Q1, axis=1)
    # mean_vol_ratio_Q2 = np.mean(volume_ratio_Q2, axis=1)
    # mean_vol_ratio_Q3 = np.mean(volume_ratio_Q3, axis=1)
    # mean_vol_ratio_Q4 = np.mean(volume_ratio_Q4, axis=1)
    mean_vol_ratio_Q2_wa = np.mean(volume_ratio_Q2_wa, axis=1)
    mean_vol_ratio_Q2_wd = np.mean(volume_ratio_Q2_wd, axis=1)
    mean_vol_ratio_Q4_wa = np.mean(volume_ratio_Q4_wa, axis=1)
    mean_vol_ratio_Q4_wd = np.mean(volume_ratio_Q4_wd, axis=1)
    # mean_vol_ratio_streak = np.mean(volume_ratio_streak, axis=1)
    # mean_vol_ratio_streak_hv = np.mean(volume_ratio_streak_hv, axis=1)
    # mean_vol_ratio_hunt = np.mean(volume_ratio_hunt, axis=1)
    # mean_vol_ratio_chong = np.mean(volume_ratio_chong, axis=1)
    
    
    file_intensity = h5py.File(path_intensity+dataset+f'.{start},{end},{step}.h5.int', 'r+')
    # file_intensity.create_dataset('mean', data=intensity_mean_all_fields)
    # file_intensity.create_dataset('Q-structure', data=intensity_Q_all_fields)
    # file_intensity.create_dataset('Q1', data=intensity_Q1_all_fields)
    # file_intensity.create_dataset('Q2', data=intensity_Q2_all_fields)
    # file_intensity.create_dataset('Q3', data=intensity_Q3_all_fields)
    # file_intensity.create_dataset('Q4', data=intensity_Q4_all_fields)
    file_intensity.create_dataset('Q2_wa', data=intensity_Q2_wa_all_fields)
    file_intensity.create_dataset('Q2_wd', data=intensity_Q2_wd_all_fields)
    file_intensity.create_dataset('Q4_wa', data=intensity_Q4_wa_all_fields)
    file_intensity.create_dataset('Q4_wd', data=intensity_Q4_wd_all_fields)
    # file_intensity.create_dataset('streak', data=intensity_streak_all_fields)
    # file_intensity.create_dataset('streak_high_vel', data=intensity_streak_hv_all_fields)
    # file_intensity.create_dataset('hunt', data=intensity_hunt_all_fields)
    # file_intensity.create_dataset('chong', data=intensity_chong_all_fields)
    # file_intensity.create_dataset('vol_Q-structure', data=mean_vol_ratio_Q)
    # file_intensity.create_dataset('vol_Q1', data=mean_vol_ratio_Q1)
    # file_intensity.create_dataset('vol_Q2', data=mean_vol_ratio_Q2)
    # file_intensity.create_dataset('vol_Q3', data=mean_vol_ratio_Q3)
    # file_intensity.create_dataset('vol_Q4', data=mean_vol_ratio_Q4)
    file_intensity.create_dataset('vol_Q2_wa', data=mean_vol_ratio_Q2_wa)
    file_intensity.create_dataset('vol_Q2_wd', data=mean_vol_ratio_Q2_wd)
    file_intensity.create_dataset('vol_Q4_wa', data=mean_vol_ratio_Q4_wa)
    file_intensity.create_dataset('vol_Q4_wd', data=mean_vol_ratio_Q4_wd)
    # file_intensity.create_dataset('vol_streak', data=mean_vol_ratio_streak)
    # file_intensity.create_dataset('vol_streak_high_vel', data=mean_vol_ratio_streak_hv)
    # file_intensity.create_dataset('vol_hunt', data=mean_vol_ratio_hunt)
    # file_intensity.create_dataset('vol_chong', data=mean_vol_ratio_chong)
    
    
def plot_enstrophy_shares(file,
                          filenorm='./P125_21pi_vu/P125_21pi_vu',
                          structure=['Q-structure',
                                     'streak',
                                     'streak_high_vel',
                                     'hunt',
                                     'chong'],
                          Q_events=['all'],
                          plot_mean=False,
                          window=(None,101)):
    normdata = gd.get_data_norm(filenorm)
    normdata.geom_param(4544, 1,1,1)
    y = normdata.y_h
    ny = normdata.ny
    utau = normdata.vtau
    y_plus = (1-np.abs(y))*utau/ny
    hf = h5py.File(file, 'r+')
    enstrophy_mean = np.array(hf['mean'])
    enstrophy_mean_plus = enstrophy_mean*(ny**2)/(utau**4)
    
    
    legend = []
    cmap = mpl.colormaps['twilight']
    colors = cmap(np.linspace(0, 1, len(structure)+len(Q_events)+3))
    linewidth = 3
    fontsize = 14
    fontweight = 500
    # colors = ['r', 'b', 'k', 'm', 'c']
    
    jj = 0
    for ii, struc in enumerate(structure):
        if struc == 'Q-structure':
            for event in Q_events:
                if event == 'all':
                    continue
                enstrophy_struc = np.array(hf[event+'-structure'])
                enstrophy_struc_plus = enstrophy_struc*(ny**2)/(utau**4)
                vol_struc = np.array(hf[f'vol_{event}'+'-structure'])
                plt.plot(y_plus[window[0]:window[1]], 
                         enstrophy_struc_plus[window[0]:window[1]], 
                         color=colors[ii+jj+1],
                         linewidth=linewidth)
                legend.append(f'$\phi⁺ ({event})$')
                plt.plot(y_plus[window[0]:window[1]], 
                         (vol_struc*enstrophy_mean_plus)[window[0]:window[1]], 
                         '--',
                         color=colors[ii+jj+1],
                         linewidth=linewidth) 
                legend.append(r'$\phi⁺_{mean}$*'+f'$V({event})'+r'/V_{tot}$')
                jj += 1
        
        jj = len(Q_events)-1
    
        if struc == 'Q-structure' and not 'all' in Q_events:
            continue
        
        enstrophy_struc = np.array(hf[struc])
        enstrophy_struc_plus = enstrophy_struc*(ny**2)/(utau**4)
        vol_struc = np.array(hf[f'vol_{struc}'])
        plt.plot(y_plus[window[0]:window[1]], 
                 enstrophy_struc_plus[window[0]:window[1]], 
                 color=colors[ii+jj+1],
                 linewidth=linewidth)
        legend.append(f'$\phi⁺ ({struc})$')
        plt.plot(y_plus[window[0]:window[1]], 
                 (vol_struc*enstrophy_mean_plus)[window[0]:window[1]], 
                 '--',
                 color=colors[ii+jj+1],
                 linewidth=linewidth) 
        legend.append(r'$\phi⁺_{mean}$*'+f'$V({struc})'+r'/V_{tot}$')
        
        
        
        
    if plot_mean:
        plt.plot(y_plus[window[0]:window[1]], 
                 enstrophy_mean_plus[window[0]:window[1]], 
                 color=colors[-2],
                 linewidth=linewidth)
        legend.append(r'$\phi⁺_{mean}$')
        
    # plt.title('Enstrophy Contribution of Different Structures')
    plt.xlabel(r'$y^+$', 
               fontsize=fontsize,
               fontweight=fontweight)
    plt.ylabel(r'$\phi^+$', 
               fontsize=fontsize,
               fontweight=fontweight,
               rotation='horizontal',
               horizontalalignment='right')
    plt.tick_params(axis='both',
                    labelsize=fontsize)
    plt.legend(legend,
               prop={'size': fontsize-2,
                     'weight': fontweight})
    
    
def plot_intensity_shares(file,
                          filenorm='./P125_21pi_vu/P125_21pi_vu',
                          structure=['Q-structure',
                                     'streak',
                                     'streak_high_vel',
                                     'hunt',
                                     'chong'],
                          Q_events=['all'],
                          intensity='u',
                          plot_mean=False,
                          window=(None,101),
                          ratio=False):
    normdata = gd.get_data_norm(filenorm)
    normdata.geom_param(4544, 1,1,1)
    y = normdata.y_h
    ny = normdata.ny
    utau = normdata.vtau
    y_plus = (1-np.abs(y))*utau/ny
    hf = h5py.File(file, 'r+')
    intensity_mean = np.array(hf['mean'])
    intensity_mean_plus = intensity_mean
    intensity_mean_plus[:3,:] = intensity_mean_plus[:3,:]/utau
    intensity_mean_plus[3,:] = intensity_mean[3,:]/(utau**2)

    
    legend = []
    cmap = mpl.colormaps['twilight']
    colors = cmap(np.linspace(0, 1, len(structure)+len(Q_events)+3))
    # colors = ['r', 'b', 'k', 'm', 'c']
    intensities = ["u'", "v'", "w'", "uv"]
    linewidth = 3
    fontsize = 14
    fontweight = 500
    
    jj = 0
    
    for ii, struc in enumerate(structure):
        if struc == 'Q-structure':
            for event in Q_events:
                if event == 'all':
                    continue
                intensity_struc = np.array(hf[event])
                if intensity != 'uv':
                    intensity_struc_plus = intensity_struc/utau
                else:
                    intensity_struc_plus = intensity_struc/(utau**2)
                vol_struc = np.array(hf[f'vol_{event}'])
                plt.plot(y_plus[window[0]:window[1]], 
                         intensity_struc_plus[intensities.index(intensity),
                                              window[0]:window[1]], 
                         color=colors[ii+jj+1],
                         linewidth=linewidth)
                legend.append(f'$\phi⁺ ({event})$')
                plt.plot(y_plus[window[0]:window[1]], 
                         (vol_struc*intensity_mean_plus)[intensities.index(intensity),
                                                         window[0]:window[1]], 
                         '--',
                         color=colors[ii+jj+1],
                         linewidth=linewidth) 
                legend.append(r'$\phi⁺_{mean}$*'+f'$V({event})'+r'/V_{tot}$')
                jj += 1
                
        jj = len(Q_events)-1
    
        if struc == 'Q-structure' and not 'all' in Q_events:
            continue
        
        
        intensity_struc = np.array(hf[struc])
        if intensity != 'uv':
            intensity_struc_plus = intensity_struc/utau
        else:
            intensity_struc_plus = intensity_struc/(utau**2)
        vol_struc = np.array(hf[f'vol_{struc}'])
        if ratio:
            plt.plot(y_plus[window[0]:window[1]], 
                     np.divide(intensity_struc_plus[intensities.index(intensity), 
                                     window[0]:window[1]], 
                               (vol_struc*intensity_mean_plus)[intensities.index(intensity),
                                                          window[0]:window[1]]), 
                     color=colors[ii+jj+1],
                     linewidth=linewidth)
            if intensity != 'uv':
                legend.append(f'{intensity}({struc})/vol({struc})')
            else:
                legend.append(f'{intensity}_mean({struc})/vol({struc})')
        
        else:
            plt.plot(y_plus[window[0]:window[1]], 
                     intensity_struc_plus[intensities.index(intensity), 
                                     window[0]:window[1]], 
                     color=colors[ii+jj+1],
                     linewidth=linewidth)
            plt.plot(y_plus[window[0]:window[1]], 
                     (vol_struc*intensity_mean_plus)[intensities.index(intensity),
                                                window[0]:window[1]], 
                     '--',
                     color = colors[ii+jj+1],
                     linewidth=linewidth) 
            if intensity != 'uv':
                legend.append(f'{intensity}⁺({struc})')
                legend.append(f'{intensity}⁺*V({struc})'+r'/V_{tot}$')
            else:
                legend.append(f'{intensity}⁺({struc})')
                legend.append(f'{intensity}⁺'+r'$_{mean}$*'+f'$V({struc})'+r'/V_{tot}$')
            
        
        
        
    if plot_mean and not ratio:
        plt.plot(y_plus[window[0]:window[1]], 
                 intensity_mean_plus[intensities.index(intensity), 
                                window[0]:window[1]], 
                 color=colors[-2],
                 linewidth=linewidth)
        if intensity != 'uv':
            legend.append(f'{intensity}⁺')
        else:
            legend.append(f'{intensity}⁺'+r'$_{mean}$')
        
    # plt.title('Intensity Contribution of Different Structures')
    plt.xlabel(r'$y^+$', 
               fontsize=fontsize,
               fontweight=fontweight)
    if not ratio:
        if intensity != 'uv':
            plt.ylabel(f'${intensity}⁺$', 
                       fontsize=fontsize,
                       fontweight=fontweight,
                       rotation='horizontal',
                       horizontalalignment='right')
        else:
            plt.ylabel(f'${intensity}⁺$', 
                       fontsize=fontsize,
                       fontweight=fontweight,
                       rotation='horizontal',
                       horizontalalignment='right')
    plt.tick_params(axis='both',
                    labelsize=fontsize)
    plt.legend(legend,
               prop={'size': fontsize-2,
                     'weight': fontweight})
    