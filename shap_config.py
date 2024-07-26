# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 12:58:26 2023

@author: andres cremades botella

File containing the functions of the SHAP
"""
import numpy as np
from time import time
from tqdm import tqdm

class shap_conf():
    
    def __init__(self,filecnn='trained_model.h5', mode='mse'):
        """
        Initialization of the SHAP class
        """
        if mode == 'mse':
            import ann_config_3D as ann
            self.background = None
            CNN = ann.convolutional_residual()
            # CNN.load_ANN(filename=filecnn)
            # CNN.load_frozen_model()
            CNN.load_optimized_model()
            # CNN.load_openvino_optimized_model()
            # self.model = CNN.model
            # self.frozen_func = CNN.frozen_func
            self.model_opt = CNN.model_opt 
            # self.model_opt_openvino = CNN.compiled_model
            # self.output_key = CNN.output_key
        elif mode == 'cf':
            import ann_config_1D as ann
            self.background = None
            CNN = ann.convolutional_residual()
            CNN.load_ANN(filename=filecnn)
            # CNN.load_frozen_model()
            # CNN.load_optimized_model()
            # CNN.load_openvino_optimized_model()
            self.model = CNN.model
            # self.frozen_func = CNN.frozen_func
            # self.model_opt = CNN.model_opt 
            # self.model_opt_openvino = CNN.compiled_model
            # self.output_key = CNN.output_key
            
        
    def calc_shap_kernel(self,
                         start,
                         end,
                         step=1,
                         structure='Q-structure',
                         error='mse',
                         dataset='P125_21pi_vu',
                         dir_shap='./',
                         dir_uvw='./',
                         dir_structures='./',
                         dir_cf='./',
                         fileUmean="Umean.txt",
                         filenorm="norm.txt",
                         filerms="Urms.txt",
                         padpix=15,
                         backgroundrms=False,
                         perc_H=0.95,
                         Href=1.75):
        
        import get_data_fun as gd
        import shap
        import os
        
        if error == 'mse':
            file=dir_shap+f'{dataset}_{structure}_SHAP_mse/{dataset}'
        elif error == 'cf':
            file=dir_shap+f'{dataset}_{structure}_SHAP_cf/{dataset}'
            
        fileuvw=dir_uvw+f'{dataset}/{dataset}'
        filecf=dir_cf+f'{dataset}_cf/{dataset}'
        normdata = gd.get_data_norm(file_read=fileuvw,
                                    file_cf=filecf)
        normdata.geom_param(start,1,1,1)
        try:
            normdata.read_Umean(file=fileUmean)
        except:
            normdata.calc_Umean(start,end)
        try:
            normdata.read_norm(file=filenorm)
        except:
            normdata.calc_norm(start,end)
        try:
            normdata.read_Urms(file=filerms)
        except:
            normdata.calc_rms(start,end)
            
        normdata.read_cfd_matrices()
        normdata.read_fd_coeffs()
            
        self.UUmean = normdata.UUmean
        self.uumin = normdata.uumin
        self.uumax = normdata.uumax
        # CFD matrices
        self.A = normdata.A
        self.B = normdata.B
        # FD coefficients
        self.fd_coeffs = normdata.fd_coeffs
        
        self.y_h = normdata.y_h
        self.ny = normdata.ny
        self.shap_values = []
        if not backgroundrms:
            self.create_background(normdata)
        for ii in range(start,end,step):
            if os.path.exists(file+'.'+str(ii)+'.h5.shap'):
                print('Existing...')
                continue
            
            struc = normdata.read_uvstruc(ii,
                                          cwd=dir_structures,
                                          padpix=padpix,
                                          structure=structure)
            
            # Get array of the ground truth velocity fields
            uvmax = np.max(struc.mat_segment)
            self.segmentation = struc.mat_segment-1
            self.segmentation[self.segmentation==-1] = uvmax
            uu_i,vv_i,ww_i = normdata.read_velocity(ii,padpix=padpix)
            self.input = normdata.norm_velocity(uu_i,vv_i,ww_i,padpix=padpix)[0,:,:,:,:]
            if backgroundrms:
                self.create_background_rms(normdata,struc,padpix=padpix,perc_H=perc_H,Href=Href)
                
            if error == 'mse':
                uu_o,vv_o,ww_o = normdata.read_velocity(ii+1)
                self.output = normdata.norm_velocity(uu_o,vv_o,ww_o)[0,:,:,:,:]
            elif error == 'cf':
                self.output = normdata.read_c_f(ii+1, dim_2D=False)
            # Calculate SHAP values 
            nmax2 = len(struc.vol)+1
            zshap = np.ones((1,nmax2))
            self.get_structure_indices(nmax2)
            # If clause MSE or CF 
            if error == 'mse':
                explainer = shap.KernelExplainer(self.model_function_mse,\
                                             np.zeros((1,nmax2)))
            elif error == 'cf':
                explainer = shap.KernelExplainer(self.model_function_cf,\
                                             np.zeros((1,nmax2)))
            shap_values = explainer.shap_values(zshap,nsamples="auto")[0][0]
            self.write_output(shap_values,ii,file=file)
            
            
    def predict_frozen(self, input_field):
        import tensorflow as tf
        predictions = self.frozen_func(x=tf.constant(input_field, dtype=tf.float32))
        return predictions[0].numpy()
            
            
    def write_output(self,shap,ii,file='../P125_21pi_vu_SHAP/P125_21pi_vu'):
        """
        Write the structures shap and geometric characteristics
        """ 
        import h5py
        hf = h5py.File(file+'.'+str(ii)+'.h5.shap', 'w')
        hf.create_dataset('SHAP', data=shap)
        
    def read_shap(self,ii,file='../P125_21pi_vu_SHAP/P125_21pi_vu'):
        """
        Function for read the SHAP values
        """
        import h5py
        hf = h5py.File(file+'.'+str(ii)+'.h5.shap', 'r')
        shap_values = np.array(hf['SHAP'])
        return shap_values
    
    def eval_shap(self,start=1000,end=9999,step=1,\
                  fileuvw='../P125_21pi_vu/P125_21pi_vu',\
                  fileQ='../P125_21pi_vu_Q/P125_21pi_vu',\
                  fileshap='../P125_21pi_vu_SHAP/P125_21pi_vu',\
                  fileUmean="Umean.txt",filenorm="norm.txt",padpix=15):
        """
        Function to evaluate the value of the mse calculated by SHAP and by the 
        model
        """
        import get_data_fun as gd
        normdata = gd.get_data_norm(file_read=fileuvw)
        normdata.geom_param(start,1,1,1)
        try:
            normdata.read_Umean(file=fileUmean)
        except:
            normdata.calc_Umean(start,end)
        try:
            normdata.read_norm(file=filenorm)
        except:
            normdata.calc_norm(start,end)
        self.create_background(normdata)
        self.error_mse2 = []
        for ii in range(start,end,step):
            uv_struc = normdata.read_uvstruc(ii,fileQ=fileQ,padpix=padpix)
            uvmax = np.max(uv_struc.mat_segment)
            self.segmentation = uv_struc.mat_segment-1
            self.segmentation[self.segmentation==-1] = uvmax
            nmax2 = len(uv_struc.event)+1
            zs = np.zeros((1,nmax2))
            uu_i,vv_i,ww_i = normdata.read_velocity(ii,padpix=padpix)
            self.input = normdata.norm_velocity(uu_i,vv_i,ww_i,padpix=padpix)[0,:,:,:,:]
            input_field = self.input.copy()
            uu_o,vv_o,ww_o = normdata.read_velocity(ii+1)
            self.output = normdata.norm_velocity(uu_o,vv_o,ww_o)[0,:,:,:,:] 
            # MSE or CF if clause
            mse_f = self.shap_model_kernel(input_field)
            shap_val = self.read_shap(ii,file=fileshap)
            shap0 = self.model_function(zs)[0][0]
            mse_g = np.sum(shap_val)+shap0
            error_mse2 = (mse_f-mse_g)**2
            print(error_mse2)
            self.error_mse2.append(error_mse2)        
        import h5py
        hf = h5py.File('mse_fg2.h5', 'w')
        hf.create_dataset('mse_fg2', data=self.error_mse2)
        
        
    def get_structure_indices(self, num_struc):
        struc_indx = []
        for ii in range(num_struc):
            indx = np.array(np.where(self.segmentation == ii)).transpose()
            struc_indx.append(indx.astype(int))
        self.struc_indx = np.array(struc_indx, dtype=object)
        
        
    def mask_dom(self,zs):
        """
        Function for making the domain
        """
        # If no background is defined the mean value of the field is take
        if self.background is None:
            self.background = self.input.mean((0,1))*np.ones((3,))
        
        mask_out = self.input.copy()
        if 0 not in zs:
            return mask_out
        
        # Replace the values of the field in which the feature is deleted
        
        struc_selected = np.where(zs==0)[0].astype(int)
        indx = np.vstack(self.struc_indx[struc_selected]).astype(int)
        if len(self.background.shape) == 1:
            mask_out[indx[:,0], indx[:,1], indx[:,2], :] = self.background
        else:
            mask_out[indx[:,0], indx[:,1], indx[:,2], :] = self.background[self.segmentation == indx[:,0], indx[:,1], indx[:,2], :]
        
        return mask_out
    
    def shap_model_kernel(self,model_input,error='mse'):
        """
        Model to calculate the shap value
        """
        input_pred = model_input.reshape(1,model_input.shape[0],\
                                         model_input.shape[1],\
                                             model_input.shape[2],\
                                                 model_input.shape[3])
        
        if error == 'mse':
            pred = self.model_opt.predict(input_pred)
            len_y = self.output.shape[0]
            len_z = self.output.shape[1]
            len_x = self.output.shape[2]
            mse  = np.mean(np.sqrt((self.output.reshape(-1,len_y,len_z,len_x,3)\
                                    -pred)**2))
            return mse
        
        elif error == 'cf':
            pred = self.model.predict(input_pred)
            mse_cf = np.mean(np.sqrt((self.output-pred)**2))
            return mse_cf
    
    
    def friction_coefficient_high_precision(self, input_field):
        '''
        Calculate the friction coefficient for a certain snapshot with 
        compact finite differences.

        '''
        avg_u_norm = np.mean(
                     input_field[0,:,:,:,0],
                    axis=(1,2)          
                     )
        avg_U = self.UUmean + self.uumin + \
                     avg_u_norm*(self.uumax-self.uumin)
        grad_U_y = np.linalg.solve(self.A, np.dot(self.B, avg_U))           
        grad_U_wall = 0.5*(grad_U_y[0]-grad_U_y[-1])
        U_bulk = 1/(self.y_h[-1]-self.y_h[0])*np.trapz(self.UUmean, 
                                                               self.y_h)
        c_f = 2*self.ny*grad_U_wall/(U_bulk**2) 
        
        return c_f
    
    
    def friction_coefficient(self, input_field):
        '''
        Calculate the friction coefficient for a certain snapshot.

        '''
        # average normalized fluctuating u at both walls
        avg_u_norm = np.zeros(3)
        avg_u_norm[0] = np.mean(
                     np.vstack(
                               [input_field[0,1,:,:,0], 
                                input_field[0,-2,:,:,0]]
                               )
                            )
        avg_u_norm[1] = np.mean(
                     np.vstack(
                               [input_field[0,2,:,:,0], 
                                input_field[0,-3,:,:,0]]
                               )
                            )
        avg_u_norm[2] = np.mean(
                     np.vstack(
                               [input_field[0,3,:,:,0], 
                                input_field[0,-4,:,:,0]]
                               )
                            )
        avg_U = 0.5*self.UUmean[1:4]+0.5*np.flip(self.UUmean[-4:-1])\
            + self.uumin + avg_u_norm*(self.uumax-self.uumin)
        avg_U = avg_U.reshape([1,3])
        grad_U_wall = np.dot(avg_U, self.fd_coeffs)
        U_bulk = 1/(self.y_h[-1]-self.y_h[0])*np.trapz(self.UUmean, 
                                                               self.y_h)
        c_f = 2*self.ny*grad_U_wall/(U_bulk**2)
        
        return c_f
    
    
    def calc_cf(self, 
                fileuvw,
                fileQ,
                file_output,
                start, 
                end, 
                step=1, 
                fileUmean="Umean.txt",
                filenorm="norm.txt",
                filerms="Urms.txt",
                padpix=15,
                high_precision=False):
        '''
        Calculate the friction coefficient for a portion of the dataset.

        '''
        import get_data_fun as gd
        import numpy as np
        import h5py
        
        normdata = gd.get_data_norm(file_read=fileuvw)
        normdata.geom_param(start,1,1,1)
        try:
            normdata.read_Umean(file=fileUmean)
        except:
            normdata.calc_Umean(start,end)
        try:
            normdata.read_norm(file=filenorm)
        except:
            normdata.calc_norm(start,end)
        try:
            normdata.read_Urms(file=filerms)
        except:
            normdata.calc_rms(start,end)
        
        normdata.read_cfd_matrices()
        normdata.read_fd_coeffs()
            
        c_f = np.zeros(int(np.floor((end-start)/step)))
        for ii in range(start,end,step):
    
            uv_struc = normdata.read_uvstruc(ii,fileQ=fileQ,padpix=padpix)
            
            # Get array of the velocity fields
            uvmax = np.max(uv_struc.mat_segment)
            self.segmentation = uv_struc.mat_segment-1
            self.segmentation[self.segmentation==-1] = uvmax
            uu_i,vv_i,ww_i = normdata.read_velocity(ii,padpix=padpix)
            self.input = normdata.norm_velocity(uu_i,vv_i,ww_i,padpix=padpix)[0,:,:,:,:]
            self.input_reshaped = self.input.reshape(-1,
                                                     self.input.shape[0],
                                                     self.input.shape[1],
                                                     self.input.shape[2],
                                                     3)
            if high_precision:
                c_f[ii-start] = self.friction_coefficient_high_precision(
                                            self.input_reshaped, normdata)
            else:
                c_f[ii-start] = self.friction_coefficient(
                                            self.input_reshaped, normdata)
            
        
        # Write to file    
        
        if high_precision:
            hf = h5py.File(file_output+'_'+str(start)+'_'+str(end)+'_hp'+'.h5.cf', 'w')
        else:
            hf = h5py.File(file_output+'_'+str(start)+'_'+str(end)+'.h5.cf', 'w')
        hf.create_dataset('c_f', data=c_f)
        
        
    def calc_fd_coeffs(self, 
                       fileuvw,
                       fileQ,
                       file_output,
                       start, 
                       end, 
                       step=1, 
                       fileUmean="Umean.txt",
                       filenorm="norm.txt",
                       filerms="Urms.txt",
                       padpix=15,
                       high_precision=False):
        '''
        Calculate the FD coefficients by least squares optimization

        '''
        import get_data_fun as gd
        import numpy as np
        import h5py
        
        normdata = gd.get_data_norm(file_read=fileuvw)
        normdata.geom_param(start,1,1,1)
        try:
            normdata.read_Umean(file=fileUmean)
        except:
            normdata.calc_Umean(start,end)
        try:
            normdata.read_norm(file=filenorm)
        except:
            normdata.calc_norm(start,end)
        try:
            normdata.read_Urms(file=filerms)
        except:
            normdata.calc_rms(start,end)
        
        normdata.read_cfd_matrices()
        normdata.read_fd_coeffs()
            
        grad_U_wall = np.zeros(int(np.floor((end-start)/step)))
        U = np.zeros([int(np.floor((end-start)/step)), 3])
        
        for ii in range(start,end,step):
    
            uv_struc = normdata.read_uvstruc(ii,fileQ=fileQ,padpix=padpix)
            
            # Get array of the velocity fields
            uvmax = np.max(uv_struc.mat_segment)
            self.segmentation = uv_struc.mat_segment-1
            self.segmentation[self.segmentation==-1] = uvmax
            uu_i,vv_i,ww_i = normdata.read_velocity(ii,padpix=padpix)
            self.input = normdata.norm_velocity(uu_i,vv_i,ww_i,padpix=padpix)[0,:,:,:,:]
            self.input_reshaped = self.input.reshape(-1,
                                                     self.input.shape[0],
                                                     self.input.shape[1],
                                                     self.input.shape[2],
                                                     3)
            
            avg_u_norm = np.mean(
                         self.input_reshaped[0,:,:,:,0],
                        axis=(1,2)          
                         )
            avg_U = normdata.UUmean + normdata.uumin + \
                         avg_u_norm*(normdata.uumax-normdata.uumin)
            grad_U_y = np.linalg.solve(normdata.A, np.dot(normdata.B, avg_U))           
            grad_U_wall[ii-start] = 0.5*(grad_U_y[0]-grad_U_y[-1])
        
            U[ii-start, 0] = 0.5*(avg_U[1] + avg_U[-2])
            U[ii-start, 1] = 0.5*(avg_U[2] + avg_U[-3])
            U[ii-start, 2] = 0.5*(avg_U[3] + avg_U[-4])
            
           
        print(U.shape, grad_U_wall.shape)
        c = np.linalg.lstsq(U, grad_U_wall, rcond=None)
            
        print(c[0])
        
        # Write to file    
        
        
        hf = h5py.File(file_output+'_'+str(start)+'_'+str(end)+'.h5.fd', 'w')
        hf.create_dataset('coeff', data=c[0])
        hf.create_dataset('U', data=U)
        hf.create_dataset('dUdy', data=grad_U_wall)
            
            
    def model_function_mse(self,zs):
        ii = 0
        lm = zs.shape[0]
        mse = np.zeros((lm,1))
        for zii in tqdm(zs):
            model_input = self.mask_dom(zii)
            mse[ii,0] = self.shap_model_kernel(model_input, error='mse')
            ii += 1
        return mse
    
    
    def model_function_cf(self,zs):
        ii = 0
        lm = zs.shape[0]
        cf = np.zeros((lm,1))
        for zii in tqdm(zs):
            model_input = self.mask_dom(zii)
            cf[ii,0] = self.shap_model_kernel(model_input, error='cf')
            ii += 1
        return cf
    

    def create_background(self,data,value=0):
        """
        Function for generating the bacground value
        """
        self.background = np.zeros((3,))
        self.background[0] = (value-data.uumin)/(data.uumax-data.uumin)
        self.background[1] = (value-data.vvmin)/(data.vvmax-data.vvmin)
        self.background[2] = (value-data.wwmin)/(data.wwmax-data.wwmin)

       
    def create_background_rms(self,data,uv_struc,padpix=15,perc_H=0.95,Href=1.75):
        """
        Function for generating the bacground value
        """
        self.background = self.input.copy()
        mat_struc = uv_struc.mat_struc
        struc_yes = np.where(mat_struc==1)
        ii_y = struc_yes[0]
        ii_z = struc_yes[1]
        ii_x = struc_yes[2]
        for ind in ii_y:
            alpha = np.sqrt(perc_H*Href*data.uurms[ii_y[ind]]*data.vvrms[ii_y[ind]]/abs(self.input[ii_y[ind],ii_z[ind],ii_x[ind],0]*self.input[ii_y[ind],ii_z[ind],ii_x[ind],1]))
            self.background[ii_y[ind],ii_z[ind],ii_x[ind],0] = (alpha*self.input[ii_y[ind],ii_z[ind],ii_x[ind],0]-data.uumin)/(data.uumax-data.uumin)
            self.background[ii_y[ind],ii_z[ind],ii_x[ind],1] = (alpha*self.input[ii_y[ind],ii_z[ind],ii_x[ind],1]-data.vvmin)/(data.vvmax-data.vvmin)
            # self.background[ii_y[ind],ii_z[ind],ii_x[ind],2] = (alpha*self.input[ii_y[ind],ii_z[ind],ii_x[ind],2]-data.wwmin)/(data.wwmax-data.wwmin)   
        self.background[:,:,:,2] = (-data.wwmin)/(data.wwmax-data.wwmin)

    # def create_background_rms(self,data,padpix=15,perc_H=0.95,Href=1.75):
    #     """
    #     Function for generating the bacground value
    #     """
    #     self.background = np.zeros((data.my,data.mz+2*padpix,data.mx+2*padpix,3))
    #     for ind_ii in np.arange(data.my):
    #         for ind_jj in np.arange(data.mz+2*padpix):
    #             for ind_kk in np.arange(data.mx+2*padpix):
    #                 if abs(self.input[ind_ii,ind_jj,ind_kk,0]*self.input[ind_ii,ind_jj,ind_kk,1]) > Href*data.uurms[ind_ii]*data.vvrms[ind_ii]:
    #                     alpha = np.sqrt(0.95*Href*data.uurms[ind_ii]*data.vvrms[ind_ii]/abs(self.input[ind_ii,ind_jj,ind_kk,0]*self.input[ind_ii,ind_jj,ind_kk,1]))
    #                     self.background[ind_ii,ind_jj,ind_kk,0] = ((alpha*data.uurms[ind_ii]-data.uumin)/(data.uumax-data.uumin))
    #                     self.background[ind_ii,ind_jj,ind_kk,1] = ((alpha*data.vvrms[ind_ii]-data.vvmin)/(data.vvmax-data.vvmin))
    #                     self.background[ind_ii,ind_jj,ind_kk,2] = ((alpha*data.wwrms[ind_ii]-data.wwmin)/(data.wwmax-data.wwmin))
        
    def read_data(self,start,end,step,\
                  file='../../../data2/cremades/P125_21pi_vu_SHAP_ann4_divide/P125_21pi_vu',\
                  fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu',\
                  fileuvw='../P125_21pi_vu/P125_21pi_vu',\
                  filegrad='/media/nils/Elements/P125_21pi_vu/grad/P125_21pi_vu',
                  fileUmean="Umean.txt",filenorm="norm.txt",shapmin=-8,\
                  shapmax=5,shapminvol=-8,shapmaxvol=8,nbars=1000,\
                  absolute=True,volmin=2.7e4,readdata=False,fileread='data_plots.h5.Q',editq3=False,find_files=False):
        """
        Function for reading the data
        """
        import h5py
        import os
        self.shapmin = shapmin
        self.shapmax = shapmax
        self.shapminvol = shapminvol
        self.shapmaxvol = shapmaxvol
        self.nbars = nbars
        self.volmin = volmin
        if absolute:
            self.ylabel_shap = '$|\phi_i| \cdot 10^{-3}$'
            self.ylabel_shap_vol = '$|\phi_i/V^+| \cdot 10^{-9}$'
            self.clabel_shap = '$|\phi_i|$'
            self.clabel_shap_vol = '$|\phi_i /V^+|$'
        else:
            self.ylabel_shap = '$\phi_i \cdot 10^{-3}$'
            self.ylabel_shap_vol = '$\phi_i/V^+ \cdot 10^{-9}$'
            self.clabel_shap = '$\phi_i$'
            self.clabel_shap_vol = '$\phi_i /V^+$'
        import get_data_fun as gd
        normdata = gd.get_data_norm(file_read=fileuvw,
                                    file_grad=filegrad)
        normdata.geom_param(start,1,1,1)
        utau = normdata.vtau
        ny = normdata.ny
        if readdata:
            hf = h5py.File(fileread, 'r')
            self.volume_wa = np.array(hf['volume_wa'])
            self.volume_wd = np.array(hf['volume_wd'])
            self.shap_wa = np.array(hf['shap_wa'])
            self.shap_wd = np.array(hf['shap_wd'])
            self.shap_wa_vol = np.array(hf['shap_wa_vol'])
            self.shap_wd_vol = np.array(hf['shap_wd_vol'])
            self.event_wa = np.array(hf['event_wa'])
            self.event_wd = np.array(hf['event_wd'])
            self.uv_uvtot_wa = np.array(hf['uv_uvtot_wa'])
            self.uv_uvtot_wd = np.array(hf['uv_uvtot_wd'])
            self.uv_uvtot_wa_vol = np.array(hf['uv_uvtot_wa_vol'])
            self.uv_uvtot_wd_vol = np.array(hf['uv_uvtot_wd_vol'])
            self.uv_vol_uvtot_vol_wa = np.array(hf['uv_vol_uvtot_vol_wa'])
            self.uv_vol_uvtot_vol_wd = np.array(hf['uv_vol_uvtot_vol_wd'])
            self.volume_1 = np.array(hf['volume_1'])
            self.volume_2 = np.array(hf['volume_2'])
            self.volume_3 = np.array(hf['volume_3'])
            self.volume_4 = np.array(hf['volume_4'])
            self.volume_1_wa = np.array(hf['volume_1_wa'])
            self.volume_2_wa = np.array(hf['volume_2_wa'])
            self.volume_3_wa = np.array(hf['volume_3_wa'])
            self.volume_4_wa = np.array(hf['volume_4_wa'])
            self.volume_1_wd = np.array(hf['volume_1_wd'])
            self.volume_2_wd = np.array(hf['volume_2_wd'])
            self.volume_3_wd = np.array(hf['volume_3_wd'])
            self.volume_4_wd = np.array(hf['volume_4_wd'])
            self.shap_1 = np.array(hf['shap_1'])
            self.shap_2 = np.array(hf['shap_2'])
            self.shap_3 = np.array(hf['shap_3'])
            self.shap_4 = np.array(hf['shap_4'])
            self.shap_1_wa = np.array(hf['shap_1_wa'])
            self.shap_2_wa = np.array(hf['shap_2_wa'])
            self.shap_3_wa = np.array(hf['shap_3_wa'])
            self.shap_4_wa = np.array(hf['shap_4_wa'])
            self.shap_1_wd = np.array(hf['shap_1_wd'])
            self.shap_2_wd = np.array(hf['shap_2_wd'])
            self.shap_3_wd = np.array(hf['shap_3_wd'])
            self.shap_4_wd = np.array(hf['shap_4_wd'])
            self.shap_1_vol = np.array(hf['shap_1_vol'])
            self.shap_2_vol = np.array(hf['shap_2_vol'])
            self.shap_3_vol = np.array(hf['shap_3_vol'])
            self.shap_4_vol = np.array(hf['shap_4_vol'])
            self.shap_1_vol_wa = np.array(hf['shap_1_vol_wa'])
            self.shap_2_vol_wa = np.array(hf['shap_2_vol_wa'])
            self.shap_3_vol_wa = np.array(hf['shap_3_vol_wa'])
            self.shap_4_vol_wa = np.array(hf['shap_4_vol_wa'])
            self.shap_1_vol_wd = np.array(hf['shap_1_vol_wd'])
            self.shap_2_vol_wd = np.array(hf['shap_2_vol_wd'])
            self.shap_3_vol_wd = np.array(hf['shap_3_vol_wd'])
            self.shap_4_vol_wd = np.array(hf['shap_4_vol_wd'])
            self.uv_uvtot_1 = np.array(hf['uv_uvtot_1'])
            self.uv_uvtot_2 = np.array(hf['uv_uvtot_2'])
            self.uv_uvtot_3 = np.array(hf['uv_uvtot_3'])
            self.uv_uvtot_4 = np.array(hf['uv_uvtot_4'])
            self.k_ktot_1 = np.array(hf['k_ktot_1'])
            self.k_ktot_2 = np.array(hf['k_ktot_2'])
            self.k_ktot_3 = np.array(hf['k_ktot_3'])
            self.k_ktot_4 = np.array(hf['k_ktot_4'])
            self.vv_vvtot_1 = np.array(hf['vv_vvtot_1'])
            self.vv_vvtot_2 = np.array(hf['vv_vvtot_2'])
            self.vv_vvtot_3 = np.array(hf['vv_vvtot_3'])
            self.vv_vvtot_4 = np.array(hf['vv_vvtot_4'])
            self.uu_uutot_1 = np.array(hf['uu_uutot_1'])
            self.uu_uutot_2 = np.array(hf['uu_uutot_2'])
            self.uu_uutot_3 = np.array(hf['uu_uutot_3'])
            self.uu_uutot_4 = np.array(hf['uu_uutot_4'])
            self.ens_enstot_1 = np.array(hf['ens_enstot_1'])
            self.ens_enstot_2 = np.array(hf['ens_enstot_2'])
            self.ens_enstot_3 = np.array(hf['ens_enstot_3'])
            self.ens_enstot_4 = np.array(hf['ens_enstot_4'])
            self.uv_uvtot_1_wa = np.array(hf['uv_uvtot_1_wa'])
            self.uv_uvtot_2_wa = np.array(hf['uv_uvtot_2_wa'])
            self.uv_uvtot_3_wa = np.array(hf['uv_uvtot_3_wa'])
            self.uv_uvtot_4_wa = np.array(hf['uv_uvtot_4_wa'])
            self.vv_vvtot_1_wa = np.array(hf['vv_vvtot_1_wa'])
            self.vv_vvtot_2_wa = np.array(hf['vv_vvtot_2_wa'])
            self.vv_vvtot_3_wa = np.array(hf['vv_vvtot_3_wa'])
            self.vv_vvtot_4_wa = np.array(hf['vv_vvtot_4_wa'])
            self.uu_uutot_1_wa = np.array(hf['uu_uutot_1_wa'])
            self.uu_uutot_2_wa = np.array(hf['uu_uutot_2_wa'])
            self.uu_uutot_3_wa = np.array(hf['uu_uutot_3_wa'])
            self.uu_uutot_4_wa = np.array(hf['uu_uutot_4_wa'])
            self.k_ktot_1_wa = np.array(hf['k_ktot_1_wa'])
            self.k_ktot_2_wa = np.array(hf['k_ktot_2_wa'])
            self.k_ktot_3_wa = np.array(hf['k_ktot_3_wa'])
            self.k_ktot_4_wa = np.array(hf['k_ktot_4_wa'])
            self.ens_enstot_1_wa = np.array(hf['ens_enstot_1_wa'])
            self.ens_enstot_2_wa = np.array(hf['ens_enstot_2_wa'])
            self.ens_enstot_3_wa = np.array(hf['ens_enstot_3_wa'])
            self.ens_enstot_4_wa = np.array(hf['ens_enstot_4_wa'])
            self.uv_uvtot_1_wd = np.array(hf['uv_uvtot_1_wd'])
            self.uv_uvtot_2_wd = np.array(hf['uv_uvtot_2_wd'])
            self.uv_uvtot_3_wd = np.array(hf['uv_uvtot_3_wd'])
            self.uv_uvtot_4_wd = np.array(hf['uv_uvtot_4_wd'])
            self.vv_vvtot_1_wd = np.array(hf['vv_vvtot_1_wd'])
            self.vv_vvtot_2_wd = np.array(hf['vv_vvtot_2_wd'])
            self.vv_vvtot_3_wd = np.array(hf['vv_vvtot_3_wd'])
            self.vv_vvtot_4_wd = np.array(hf['vv_vvtot_4_wd'])
            self.uu_uutot_1_wd = np.array(hf['uu_uutot_1_wd'])
            self.uu_uutot_2_wd = np.array(hf['uu_uutot_2_wd'])
            self.uu_uutot_3_wd = np.array(hf['uu_uutot_3_wd'])
            self.uu_uutot_4_wd = np.array(hf['uu_uutot_4_wd'])
            self.k_ktot_1_wd = np.array(hf['k_ktot_1_wd'])
            self.k_ktot_2_wd = np.array(hf['k_ktot_2_wd'])
            self.k_ktot_3_wd = np.array(hf['k_ktot_3_wd'])
            self.k_ktot_4_wd = np.array(hf['k_ktot_4_wd'])
            self.ens_enstot_1_wd = np.array(hf['ens_enstot_1_wd'])
            self.ens_enstot_2_wd = np.array(hf['ens_enstot_2_wd'])
            self.ens_enstot_3_wd = np.array(hf['ens_enstot_3_wd'])
            self.ens_enstot_4_wd = np.array(hf['ens_enstot_4_wd'])
            self.uv_uvtot_1_vol = np.array(hf['uv_uvtot_1_vol'])
            self.uv_uvtot_2_vol = np.array(hf['uv_uvtot_2_vol'])
            self.uv_uvtot_3_vol = np.array(hf['uv_uvtot_3_vol'])
            self.uv_uvtot_4_vol = np.array(hf['uv_uvtot_4_vol'])
            self.vv_vvtot_1_vol = np.array(hf['vv_vvtot_1_vol'])
            self.vv_vvtot_2_vol = np.array(hf['vv_vvtot_2_vol'])
            self.vv_vvtot_3_vol = np.array(hf['vv_vvtot_3_vol'])
            self.vv_vvtot_4_vol = np.array(hf['vv_vvtot_4_vol'])
            self.uu_uutot_1_vol = np.array(hf['uu_uutot_1_vol'])
            self.uu_uutot_2_vol = np.array(hf['uu_uutot_2_vol'])
            self.uu_uutot_3_vol = np.array(hf['uu_uutot_3_vol'])
            self.uu_uutot_4_vol = np.array(hf['uu_uutot_4_vol'])
            self.k_ktot_1_vol = np.array(hf['k_ktot_1_vol'])
            self.k_ktot_2_vol = np.array(hf['k_ktot_2_vol'])
            self.k_ktot_3_vol = np.array(hf['k_ktot_3_vol'])
            self.k_ktot_4_vol = np.array(hf['k_ktot_4_vol'])
            self.ens_enstot_1_vol = np.array(hf['ens_enstot_1_vol'])
            self.ens_enstot_2_vol = np.array(hf['ens_enstot_2_vol'])
            self.ens_enstot_3_vol = np.array(hf['ens_enstot_3_vol'])
            self.ens_enstot_4_vol = np.array(hf['ens_enstot_4_vol'])
            self.uv_uvtot_1_vol_wa = np.array(hf['uv_uvtot_1_vol_wa'])
            self.uv_uvtot_2_vol_wa = np.array(hf['uv_uvtot_2_vol_wa'])
            self.uv_uvtot_3_vol_wa = np.array(hf['uv_uvtot_3_vol_wa'])
            self.uv_uvtot_4_vol_wa = np.array(hf['uv_uvtot_4_vol_wa'])
            self.vv_vvtot_1_vol_wa = np.array(hf['vv_vvtot_1_vol_wa'])
            self.vv_vvtot_2_vol_wa = np.array(hf['vv_vvtot_2_vol_wa'])
            self.vv_vvtot_3_vol_wa = np.array(hf['vv_vvtot_3_vol_wa'])
            self.vv_vvtot_4_vol_wa = np.array(hf['vv_vvtot_4_vol_wa'])
            self.uu_uutot_1_vol_wa = np.array(hf['uu_uutot_1_vol_wa'])
            self.uu_uutot_2_vol_wa = np.array(hf['uu_uutot_2_vol_wa'])
            self.uu_uutot_3_vol_wa = np.array(hf['uu_uutot_3_vol_wa'])
            self.uu_uutot_4_vol_wa = np.array(hf['uu_uutot_4_vol_wa'])
            self.k_ktot_1_vol_wa = np.array(hf['k_ktot_1_vol_wa'])
            self.k_ktot_2_vol_wa = np.array(hf['k_ktot_2_vol_wa'])
            self.k_ktot_3_vol_wa = np.array(hf['k_ktot_3_vol_wa'])
            self.k_ktot_4_vol_wa = np.array(hf['k_ktot_4_vol_wa'])
            self.ens_enstot_1_vol_wa = np.array(hf['ens_enstot_1_vol_wa'])
            self.ens_enstot_2_vol_wa = np.array(hf['ens_enstot_2_vol_wa'])
            self.ens_enstot_3_vol_wa = np.array(hf['ens_enstot_3_vol_wa'])
            self.ens_enstot_4_vol_wa = np.array(hf['ens_enstot_4_vol_wa'])
            self.uv_uvtot_1_vol_wd = np.array(hf['uv_uvtot_1_vol_wd'])
            self.uv_uvtot_2_vol_wd = np.array(hf['uv_uvtot_2_vol_wd'])
            self.uv_uvtot_3_vol_wd = np.array(hf['uv_uvtot_3_vol_wd'])
            self.uv_uvtot_4_vol_wd = np.array(hf['uv_uvtot_4_vol_wd'])
            self.vv_vvtot_1_vol_wd = np.array(hf['vv_vvtot_1_vol_wd'])
            self.vv_vvtot_2_vol_wd = np.array(hf['vv_vvtot_2_vol_wd'])
            self.vv_vvtot_3_vol_wd = np.array(hf['vv_vvtot_3_vol_wd'])
            self.vv_vvtot_4_vol_wd = np.array(hf['vv_vvtot_4_vol_wd'])
            self.uu_uutot_1_vol_wd = np.array(hf['uu_uutot_1_vol_wd'])
            self.uu_uutot_2_vol_wd = np.array(hf['uu_uutot_2_vol_wd'])
            self.uu_uutot_3_vol_wd = np.array(hf['uu_uutot_3_vol_wd'])
            self.uu_uutot_4_vol_wd = np.array(hf['uu_uutot_4_vol_wd'])
            self.k_ktot_1_vol_wd = np.array(hf['k_ktot_1_vol_wd'])
            self.k_ktot_2_vol_wd = np.array(hf['k_ktot_2_vol_wd'])
            self.k_ktot_3_vol_wd = np.array(hf['k_ktot_3_vol_wd'])
            self.k_ktot_4_vol_wd = np.array(hf['k_ktot_4_vol_wd'])
            self.ens_enstot_1_vol_wd = np.array(hf['ens_enstot_1_vol_wd'])
            self.ens_enstot_2_vol_wd = np.array(hf['ens_enstot_2_vol_wd'])
            self.ens_enstot_3_vol_wd = np.array(hf['ens_enstot_3_vol_wd'])
            self.ens_enstot_4_vol_wd = np.array(hf['ens_enstot_4_vol_wd'])
            self.uv_vol_uvtot_vol_1 = np.array(hf['uv_vol_uvtot_vol_1'])
            self.uv_vol_uvtot_vol_2 = np.array(hf['uv_vol_uvtot_vol_2'])
            self.uv_vol_uvtot_vol_3 = np.array(hf['uv_vol_uvtot_vol_3'])
            self.uv_vol_uvtot_vol_4 = np.array(hf['uv_vol_uvtot_vol_4'])
            self.k_vol_ktot_vol_1 = np.array(hf['k_vol_ktot_vol_1'])
            self.k_vol_ktot_vol_2 = np.array(hf['k_vol_ktot_vol_2'])
            self.k_vol_ktot_vol_3 = np.array(hf['k_vol_ktot_vol_3'])
            self.k_vol_ktot_vol_4 = np.array(hf['k_vol_ktot_vol_4'])
            self.ens_vol_enstot_vol_1 = np.array(hf['ens_vol_enstot_vol_1'])
            self.ens_vol_enstot_vol_2 = np.array(hf['ens_vol_enstot_vol_2'])
            self.ens_vol_enstot_vol_3 = np.array(hf['ens_vol_enstot_vol_3'])
            self.ens_vol_enstot_vol_4 = np.array(hf['ens_vol_enstot_vol_4'])
            self.uv_vol_uvtot_vol_1_wa = np.array(hf['uv_vol_uvtot_vol_1_wa'])
            self.uv_vol_uvtot_vol_2_wa = np.array(hf['uv_vol_uvtot_vol_2_wa'])
            self.uv_vol_uvtot_vol_3_wa = np.array(hf['uv_vol_uvtot_vol_3_wa'])
            self.uv_vol_uvtot_vol_4_wa = np.array(hf['uv_vol_uvtot_vol_4_wa'])
            self.uv_vol_uvtot_vol_1_wd = np.array(hf['uv_vol_uvtot_vol_1_wd'])
            self.uv_vol_uvtot_vol_2_wd = np.array(hf['uv_vol_uvtot_vol_2_wd'])
            self.uv_vol_uvtot_vol_3_wd = np.array(hf['uv_vol_uvtot_vol_3_wd'])
            self.uv_vol_uvtot_vol_4_wd = np.array(hf['uv_vol_uvtot_vol_4_wd'])
            self.event_1 = np.array(hf['event_1'])
            self.event_2 = np.array(hf['event_2'])
            self.event_3 = np.array(hf['event_3'])
            self.event_4 = np.array(hf['event_4'])
            self.event_1_wa = np.array(hf['event_1_wa'])
            self.event_2_wa = np.array(hf['event_2_wa'])
            self.event_3_wa = np.array(hf['event_3_wa'])
            self.event_4_wa = np.array(hf['event_4_wa'])
            self.event_1_wd = np.array(hf['event_1_wd'])
            self.event_2_wd = np.array(hf['event_2_wd'])
            self.event_3_wd = np.array(hf['event_3_wd'])
            self.event_4_wd = np.array(hf['event_4_wd'])
            self.voltot = np.array(hf['voltot'])
            self.shapback_list = np.array(hf['shapback_list'])
            self.shapbackvol_list = np.array(hf['shapbackvol_list'])
            self.AR1_grid = np.array(hf['AR1_grid'])
            self.AR2_grid = np.array(hf['AR2_grid'])
            self.SHAP_grid1 = np.array(hf['SHAP_grid1'])
            self.SHAP_grid2 = np.array(hf['SHAP_grid2'])
            self.SHAP_grid3 = np.array(hf['SHAP_grid3'])
            self.SHAP_grid4 = np.array(hf['SHAP_grid4'])
            self.SHAP_grid1vol = np.array(hf['SHAP_grid1vol'])
            self.SHAP_grid2vol = np.array(hf['SHAP_grid2vol'])
            self.SHAP_grid3vol = np.array(hf['SHAP_grid3vol'])
            self.SHAP_grid4vol = np.array(hf['SHAP_grid4vol'])
            self.npoin1 = np.array(hf['npoin1'])
            self.npoin2 = np.array(hf['npoin2'])
            self.npoin3 = np.array(hf['npoin3'])
            self.npoin4 = np.array(hf['npoin4'])
            self.shap1cum = np.array(hf['shap1cum'])
            self.shap2cum = np.array(hf['shap2cum'])
            self.shap3cum = np.array(hf['shap3cum'])
            self.shap4cum = np.array(hf['shap4cum'])
            self.shapbcum = np.array(hf['shapbcum'])
            self.shap_vol1cum = np.array(hf['shap_vol1cum'])
            self.shap_vol2cum = np.array(hf['shap_vol2cum'])
            self.shap_vol3cum = np.array(hf['shap_vol3cum'])
            self.shap_vol4cum = np.array(hf['shap_vol4cum'])
            self.shap_volbcum = np.array(hf['shap_volbcum'])
            self.shapmax = np.array(hf['shapmax'])
            self.shapmin = np.array(hf['shapmin'])
            self.shapmaxvol = np.array(hf['shapmaxvol'])
            self.shapminvol = np.array(hf['shapminvol'])
            self.cdg_y_1 = (1-np.abs(np.array(hf['cdg_y_1'])))*utau/ny # convert to y plus
            self.cdg_y_2 = (1-np.abs(np.array(hf['cdg_y_2'])))*utau/ny
            self.cdg_y_3 = (1-np.abs(np.array(hf['cdg_y_3'])))*utau/ny
            self.cdg_y_4 = (1-np.abs(np.array(hf['cdg_y_4'])))*utau/ny
            self.cdg_y_wa = np.array(hf['cdg_y_wa'])
            self.cdg_y_wd = np.array(hf['cdg_y_wd'])
            self.y_plus_min_1 = np.array(hf['y_plus_min_1'])
            self.y_plus_min_2 = np.array(hf['y_plus_min_2'])
            self.y_plus_min_3 = np.array(hf['y_plus_min_3'])
            self.y_plus_min_4 = np.array(hf['y_plus_min_4'])
            hf.close()
        else:
            self.volume_wa = []
            self.volume_wd = []
            self.shap_wa = []
            self.shap_wd = []
            self.shap_wa_vol = []
            self.shap_wd_vol = []
            self.event_wa = []
            self.event_wd = []
            self.uv_uvtot_wa = []
            self.uv_uvtot_wd = []
            self.uv_uvtot_wa_vol = []
            self.uv_uvtot_wd_vol = []
            self.vv_vvtot_wa = []
            self.vv_vvtot_wd = []
            self.vv_vvtot_wa_vol = []
            self.vv_vvtot_wd_vol = []
            self.uv_vol_uvtot_vol_wa = []
            self.uv_vol_uvtot_vol_wd = []
            self.cdg_y_wa = []
            self.cdg_y_wd = []
            self.volume_1 = []
            self.uv_uvtot_1 = []
            self.uv_uvtot_1_vol = []
            self.uv_vol_uvtot_vol_1 = []
            self.vv_vvtot_1 = []
            self.vv_vvtot_1_vol = []
            self.uu_uutot_1 = []
            self.uu_uutot_1_vol = []
            self.k_ktot_1 = []
            self.k_ktot_1_vol = []
            self.k_vol_ktot_vol_1 = []
            self.ens_enstot_1 = []
            self.ens_enstot_1_vol = []
            self.ens_vol_enstot_vol_1 = []
            self.shap_1 = []
            self.shap_1_vol = []
            self.event_1 = []
            self.cdg_y_1 = []
            self.volume_1_wa = []
            self.uv_uvtot_1_wa = []
            self.uv_uvtot_1_vol_wa = []
            self.uv_vol_uvtot_vol_1_wa = []
            self.vv_vvtot_1_wa = []
            self.vv_vvtot_1_vol_wa = []
            self.uu_uutot_1_wa = []
            self.uu_uutot_1_vol_wa = []
            self.k_ktot_1_wa = []
            self.k_ktot_1_vol_wa = []
            self.ens_enstot_1_wa = []
            self.ens_enstot_1_vol_wa = []
            self.shap_1_wa = []
            self.shap_1_vol_wa = []
            self.event_1_wa = []
            self.volume_1_wd = []
            self.uv_uvtot_1_wd = []
            self.uv_uvtot_1_vol_wd = []
            self.uv_vol_uvtot_vol_1_wd = []
            self.vv_vvtot_1_wd = []
            self.vv_vvtot_1_vol_wd = []
            self.uu_uutot_1_wd = []
            self.uu_uutot_1_vol_wd = []
            self.k_ktot_1_wd = []
            self.k_ktot_1_vol_wd = []
            self.ens_enstot_1_wd = []
            self.ens_enstot_1_vol_wd = []
            self.shap_1_wd = []
            self.shap_1_vol_wd = []
            self.event_1_wd = []
            self.volume_2 = []
            self.uv_uvtot_2 = []
            self.uv_uvtot_2_vol = []
            self.uv_vol_uvtot_vol_2 = []
            self.vv_vvtot_2 = []
            self.vv_vvtot_2_vol = []
            self.uu_uutot_2 = []
            self.uu_uutot_2_vol = []
            self.k_ktot_2 = []
            self.k_ktot_2_vol = []
            self.k_vol_ktot_vol_2 = []
            self.ens_enstot_2 = []
            self.ens_enstot_2_vol = []
            self.ens_vol_enstot_vol_2 = []
            self.shap_2 = []
            self.shap_2_vol = []
            self.event_2 = []
            self.cdg_y_2 = []
            self.volume_2_wa = []
            self.uv_uvtot_2_wa = []
            self.uv_uvtot_2_vol_wa = []
            self.uv_vol_uvtot_vol_2_wa = []
            self.vv_vvtot_2_wa = []
            self.vv_vvtot_2_vol_wa = []
            self.uu_uutot_2_wa = []
            self.uu_uutot_2_vol_wa = []
            self.k_ktot_2_wa = []
            self.k_ktot_2_vol_wa = []
            self.ens_enstot_2_wa = []
            self.ens_enstot_2_vol_wa = []
            self.shap_2_wa = []
            self.shap_2_vol_wa = []
            self.event_2_wa = []
            self.volume_2_wd = []
            self.uv_uvtot_2_wd = []
            self.uv_uvtot_2_vol_wd = []
            self.uv_vol_uvtot_vol_2_wd = []
            self.vv_vvtot_2_wd = []
            self.vv_vvtot_2_vol_wd = []
            self.uu_uutot_2_wd = []
            self.uu_uutot_2_vol_wd = []
            self.k_ktot_2_wd = []
            self.k_ktot_2_vol_wd = []
            self.ens_enstot_2_wd = []
            self.ens_enstot_2_vol_wd = []
            self.shap_2_wd = []
            self.shap_2_vol_wd = []
            self.event_2_wd = []
            self.volume_3 = []
            self.uv_uvtot_3 = []
            self.uv_uvtot_3_vol = []
            self.uv_vol_uvtot_vol_3 = []
            self.vv_vvtot_3 = []
            self.vv_vvtot_3_vol = []
            self.uu_uutot_3 = []
            self.uu_uutot_3_vol = []
            self.k_ktot_3 = []
            self.k_ktot_3_vol = []
            self.k_vol_ktot_vol_3 = []
            self.ens_enstot_3 = []
            self.ens_enstot_3_vol = []
            self.ens_vol_enstot_vol_3 = []
            self.shap_3 = []
            self.shap_3_vol = []
            self.event_3 = []
            self.cdg_y_3 = []
            self.volume_3_wa = []
            self.uv_uvtot_3_wa = []
            self.uv_uvtot_3_vol_wa = []
            self.uv_vol_uvtot_vol_3_wa = []
            self.vv_vvtot_3_wa = []
            self.vv_vvtot_3_vol_wa = []
            self.uu_uutot_3_wa = []
            self.uu_uutot_3_vol_wa = []
            self.k_ktot_3_wa = []
            self.k_ktot_3_vol_wa = []
            self.ens_enstot_3_wa = []
            self.ens_enstot_3_vol_wa = []
            self.shap_3_wa = []
            self.shap_3_vol_wa = []
            self.event_3_wa = []
            self.volume_3_wd = []
            self.uv_uvtot_3_wd = []
            self.uv_uvtot_3_vol_wd = []
            self.uv_vol_uvtot_vol_3_wd = []
            self.vv_vvtot_3_wd = []
            self.vv_vvtot_3_vol_wd = []
            self.uu_uutot_3_wd = []
            self.uu_uutot_3_vol_wd = []
            self.k_ktot_3_wd = []
            self.k_ktot_3_vol_wd = []
            self.ens_enstot_3_wd = []
            self.ens_enstot_3_vol_wd = []
            self.shap_3_wd = []
            self.shap_3_vol_wd = []
            self.event_3_wd = []
            self.volume_4 = []
            self.uv_uvtot_4 = []
            self.uv_uvtot_4_vol = []
            self.uv_vol_uvtot_vol_4 = []
            self.vv_vvtot_4 = []
            self.vv_vvtot_4_vol = []
            self.uu_uutot_4 = []
            self.uu_uutot_4_vol = []
            self.k_ktot_4_wa = []
            self.k_ktot_4_vol_wa = []
            self.ens_enstot_4_wa = []
            self.ens_enstot_4_vol_wa = []
            self.k_ktot_4 = []
            self.k_ktot_4_vol = []
            self.k_vol_ktot_vol_4 = []
            self.ens_enstot_4 = []
            self.ens_enstot_4_vol = []
            self.ens_vol_enstot_vol_4 = []
            self.shap_4 = []
            self.shap_4_vol = []
            self.event_4 = []
            self.cdg_y_4 = []
            self.volume_4_wa = []
            self.uv_uvtot_4_wa = []
            self.uv_uvtot_4_vol_wa = []
            self.uv_vol_uvtot_vol_4_wa = []
            self.vv_vvtot_4_wa = []
            self.vv_vvtot_4_vol_wa = []
            self.uu_uutot_4_wa = []
            self.uu_uutot_4_vol_wa = []
            self.shap_4_wa = []
            self.shap_4_vol_wa = []
            self.event_4_wa = []
            self.volume_4_wd = []
            self.uv_uvtot_4_wd = []
            self.uv_uvtot_4_vol_wd = []
            self.uv_vol_uvtot_vol_4_wd = []
            self.vv_vvtot_4_wd = []
            self.vv_vvtot_4_vol_wd = []
            self.uu_uutot_4_wd = []
            self.uu_uutot_4_vol_wd = []
            self.k_ktot_4_wd = []
            self.k_ktot_4_vol_wd = []
            self.ens_enstot_4_wd = []
            self.ens_enstot_4_vol_wd = []
            self.shap_4_wd = []
            self.shap_4_vol_wd = []
            self.event_4_wd = []
            self.voltot = np.sum(normdata.vol)
            self.shapback_list = []
            self.shapbackvol_list = []
            self.y_plus_min_1 = []
            self.y_plus_min_2 = []
            self.y_plus_min_3 = []
            self.y_plus_min_4 = []
            expmax = 2
            expmin = -1
            ngrid = 50
            AR1_vec = np.linspace(expmin,expmax,ngrid+1)
            AR2_vec = np.linspace(expmin,expmax,ngrid+1)
            self.AR1_grid = np.zeros((ngrid,ngrid))
            self.AR2_grid = np.zeros((ngrid,ngrid))
            self.SHAP_grid1 = np.zeros((ngrid,ngrid))
            self.SHAP_grid2 = np.zeros((ngrid,ngrid))
            self.SHAP_grid3 = np.zeros((ngrid,ngrid))
            self.SHAP_grid4 = np.zeros((ngrid,ngrid))
            self.SHAP_grid1vol = np.zeros((ngrid,ngrid))
            self.SHAP_grid2vol = np.zeros((ngrid,ngrid))
            self.SHAP_grid3vol = np.zeros((ngrid,ngrid))
            self.SHAP_grid4vol = np.zeros((ngrid,ngrid))
            self.npoin1 = np.zeros((ngrid,ngrid))
            self.npoin2 = np.zeros((ngrid,ngrid))
            self.npoin3 = np.zeros((ngrid,ngrid))
            self.npoin4 = np.zeros((ngrid,ngrid))
            if find_files:
                indbar = file.rfind('/')
                filesexist = os.listdir(file[:indbar])
                text1 = file[indbar+1:]+'.'
                text2 = '.h5.shap'
                range_shap =[int(x_file.replace(text1,'').replace(text2,'')) for x_file in filesexist]
            else:
                range_shap = range(start,end,step)
            lenrange_shap = len(range_shap)
            uv_Qminus = 0
            vol_Qminus = 0
            uv_Qminus_wa = 0
            vol_Qminus_wa = 0
            uvtot_cum = 0
            voltot_cum = 0
            self.shap1cum = 0
            self.shap2cum = 0
            self.shap3cum = 0
            self.shap4cum = 0
            self.shap_vol1cum = 0
            self.shap_vol2cum = 0
            self.shap_vol3cum = 0
            self.shap_vol4cum = 0
            self.shapbcum = 0
            self.shap_volbcum = 0
            try:
                normdata.read_norm()
            except:
                normdata.calc_norm(start,end)
            try:
                normdata.UUmean 
            except:
                normdata.read_Umean()
            for ii_arlim1 in np.arange(ngrid):
                arlim1inf = 10**AR1_vec[ii_arlim1]
                arlim1sup = 10**AR1_vec[ii_arlim1+1]
                for ii_arlim2 in np.arange(ngrid):
                    arlim2inf = 10**AR2_vec[ii_arlim2]
                    arlim2sup = 10**AR2_vec[ii_arlim2+1]
                    self.AR1_grid[ii_arlim1,ii_arlim2] = \
                    10**((AR1_vec[ii_arlim1]+AR1_vec[ii_arlim1+1])/2)
                    self.AR2_grid[ii_arlim1,ii_arlim2] =\
                    10**((AR2_vec[ii_arlim2]+AR2_vec[ii_arlim2+1])/2)
            for ii in range_shap:
                try:
                    uu,vv,ww = normdata.read_velocity(ii)
                    phi = normdata.read_enstrophy(ii)
                    uvtot = np.sum(abs(np.multiply(uu,vv)))
                    ktot = 0.5*np.sum(uu**2+vv**2+ww**2)
                    vvtot = np.sum(abs(np.multiply(vv,vv)))
                    uutot = np.sum(abs(np.multiply(uu,uu)))
                    enstot = np.sum(phi)
                    uv_struc = normdata.read_uvstruc(ii,
                                                     cwd=fileQ.replace('P125_21pi_vu_Q_divide/P125_21pi_vu', 
                                                                       ''))
                    if editq3:
                        for jjind in np.arange(len(uv_struc.event)):
                            if uv_struc.cdg_y[jjind]>0.9 and uv_struc.event[jjind]==3:
                                uv_struc.event[jjind] = 2
                    voltot = np.sum(normdata.vol)
                    lenstruc = len(uv_struc.event)
                    uvtot_cum += uvtot
                    voltot_cum += voltot
                    if any(uv_struc.vol>5e6):
                        print(ii)
                    lenstruc = len(uv_struc.event)
                    if absolute:
                        shapvalues = abs(self.read_shap(ii,file=file))
                        shapback = abs(shapvalues[-1])
                        self.shapmax = np.max(abs(np.array([self.shapmin,self.shapmax])))
                        self.shapmin = 0
                        self.shapmaxvol = np.max(abs(np.array([self.shapminvol,self.shapmaxvol])))
                        self.shapminvol = 0
                    else:
                        shapvalues = self.read_shap(ii,file=file)
                        shapback = shapvalues[-1]
                        self.shapmax = np.max(np.array([self.shapmin,self.shapmax]))
                        self.shapmin = np.min(np.array([self.shapmin,self.shapmax]))
                        self.shapmaxvol = np.max(np.array([self.shapminvol,self.shapmaxvol]))
                        self.shapminvol = np.min(np.array([self.shapminvol,self.shapmaxvol]))
                    uv = np.zeros((lenstruc,))
                    v2 = np.zeros((lenstruc,))
                    u2 = np.zeros((lenstruc,))
                    k = np.zeros((lenstruc,))
                    ens = np.zeros((lenstruc,))
                    uv_vol = np.zeros((lenstruc,))
                    vv_vol = np.zeros((lenstruc,))
                    uu_vol = np.zeros((lenstruc,))
                    k_vol = np.zeros((lenstruc,))
                    ens_vol = np.zeros((lenstruc,))
                    for jj in np.arange(lenstruc):
                        indexuv = np.where(uv_struc.mat_segment==jj+1)
                        for kk in np.arange(len(indexuv[0])):
                            uv[jj] += abs(uu[indexuv[0][kk],indexuv[1][kk],\
                                          indexuv[2][kk]]*vv[indexuv[0][kk],\
                                                 indexuv[1][kk],indexuv[2][kk]])
                            v2[jj] += vv[indexuv[0][kk],
                                         indexuv[1][kk],
                                         indexuv[2][kk]]**2
                            u2[jj] += uu[indexuv[0][kk],
                                         indexuv[1][kk],
                                         indexuv[2][kk]]**2
                            k[jj] += 0.5*(uu[indexuv[0][kk],
                                             indexuv[1][kk],
                                             indexuv[2][kk]]**2\
                                          +vv[indexuv[0][kk],
                                              indexuv[1][kk],
                                              indexuv[2][kk]]**2\
                                          +ww[indexuv[0][kk],
                                              indexuv[1][kk],
                                              indexuv[2][kk]]**2)
                            ens[jj] += phi[indexuv[0][kk],
                                           indexuv[1][kk],
                                           indexuv[2][kk]]    
                        uv_vol[jj] = uv[jj]/uv_struc.vol[jj]
                        vv_vol[jj] = v2[jj]/uv_struc.vol[jj]
                        uu_vol[jj] = u2[jj]/uv_struc.vol[jj]
                        k_vol[jj] = k[jj]/uv_struc.vol[jj]
                        ens_vol[jj] = ens[jj]/uv_struc.vol[jj]
                        uv_back_vol = (uvtot-np.sum(uv))/\
                        (voltot-np.sum(uv_struc.vol))
                        uv_vol_sum = np.sum(uv_vol)+uv_back_vol
                        vv_back_vol = (vvtot-np.sum(v2))/\
                        (voltot-np.sum(uv_struc.vol))
                        vv_vol_sum = np.sum(vv_vol)+vv_back_vol
                        uu_back_vol = (uutot-np.sum(v2))/\
                        (voltot-np.sum(uv_struc.vol))
                        uu_vol_sum = np.sum(uu_vol)+uu_back_vol
                        k_back_vol = (ktot-np.sum(k))/\
                        (voltot-np.sum(uv_struc.vol))
                        k_vol_sum = np.sum(k_vol)+k_back_vol
                        ens_back_vol = (enstot-np.sum(ens))/\
                        (voltot-np.sum(uv_struc.vol))
                        ens_vol_sum = np.sum(ens_vol)+ens_back_vol
                        
                        if uv_struc.cdg_y[jj] <= 0:
                            yplus_min_ii = (1+uv_struc.ymin[jj])*normdata.rey
                        else:
                            yplus_min_ii = (1-uv_struc.ymax[jj])*normdata.rey
                        if uv_struc.event[jj] == 2 or uv_struc.event[jj] == 4:
                            uv_Qminus += uv[jj]
                            vol_Qminus += uv_struc.vol[jj]
                            if yplus_min_ii < 20:
                                uv_Qminus_wa += uv[jj]
                                vol_Qminus_wa += uv_struc.vol[jj]
                        uv[jj] /= uvtot
                        v2[jj] /= vvtot
                        u2[jj] /= uutot
                        k[jj] /= ktot
                        ens[jj] /= enstot
                        
                        if uv_struc.vol[jj] > self.volmin:                  
                            Dy = uv_struc.ymax[jj]-uv_struc.ymin[jj]
                            Dz = uv_struc.dz[jj]
                            Dx = uv_struc.dx[jj]
                            AR1 = Dx/Dy
                            AR2 = Dz/Dy
                            for ii_arlim1 in np.arange(ngrid):
                                arlim1inf = 10**AR1_vec[ii_arlim1]
                                arlim1sup = 10**AR1_vec[ii_arlim1+1]
                                if AR1 >= arlim1inf and AR1<arlim1sup:
                                    for ii_arlim2 in np.arange(ngrid):
                                        arlim2inf = 10**AR2_vec[ii_arlim2]
                                        arlim2sup = 10**AR2_vec[ii_arlim2+1]
                                        if AR2 >= arlim2inf and AR2<arlim2sup:
                                            if uv_struc.event[jj] == 1:
                                                self.SHAP_grid1[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]
                                                self.SHAP_grid1vol[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]/uv_struc.vol[jj]
                                                self.npoin1[ii_arlim1,ii_arlim2] += 1
                                            elif uv_struc.event[jj] == 2:
                                                self.SHAP_grid2[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]
                                                self.SHAP_grid2vol[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]/uv_struc.vol[jj]
                                                self.npoin2[ii_arlim1,ii_arlim2] += 1
                                            elif uv_struc.event[jj] == 3:
                                                self.SHAP_grid3[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]
                                                self.SHAP_grid3vol[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]/uv_struc.vol[jj]
                                                self.npoin3[ii_arlim1,ii_arlim2] += 1
                                            elif uv_struc.event[jj] == 4:
                                                self.SHAP_grid4[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]
                                                self.SHAP_grid4vol[ii_arlim1,ii_arlim2] +=\
                                                shapvalues[jj]/uv_struc.vol[jj]
                                                self.npoin4[ii_arlim1,ii_arlim2] += 1          
                            if yplus_min_ii < 20:
                                self.volume_wa.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_wa.append(uv[jj])
                                self.uv_uvtot_wa_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_wa.append(uv_vol[jj]/uv_vol_sum)
                                self.shap_wa.append(shapvalues[jj]*1e3)
                                self.shap_wa_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_wa.append(uv_struc.event[jj])
                                self.cdg_y_wa.append(uv_struc.cdg_y[jj])
                            else:
                                self.volume_wd.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_wd.append(uv[jj])
                                self.uv_uvtot_wd_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_wd.append(uv_vol[jj]/uv_vol_sum)
                                self.shap_wd.append(shapvalues[jj]*1e3)
                                self.shap_wd_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_wd.append(uv_struc.event[jj])
                                self.cdg_y_wd.append(uv_struc.cdg_y[jj])
                            if uv_struc.event[jj] == 1:
                                self.volume_1.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_1.append(uv[jj])
                                self.uv_uvtot_1_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_1.append(uv_vol[jj]/uv_vol_sum)
                                self.vv_vvtot_1.append(v2[jj])
                                self.vv_vvtot_1_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                self.uu_uutot_1.append(u2[jj])
                                self.uu_uutot_1_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                self.k_ktot_1.append(k[jj])
                                self.k_ktot_1_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                self.k_vol_ktot_vol_1.append(k_vol[jj]/k_vol_sum)
                                self.ens_enstot_1.append(ens[jj])
                                self.ens_enstot_1_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                self.ens_vol_enstot_vol_1.append(ens_vol[jj]/ens_vol_sum)
                                self.shap_1.append(shapvalues[jj]*1e3)
                                self.shap_1_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_1.append(uv_struc.event[jj])
                                self.shap1cum += shapvalues[jj]
                                self.shap_vol1cum += shapvalues[jj]/uv_struc.vol[jj]
                                self.cdg_y_1.append(uv_struc.cdg_y[jj])
                                self.y_plus_min_1.append(yplus_min_ii) 
                                if yplus_min_ii < 20:
                                    self.volume_1_wa.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_1_wa.append(uv[jj])
                                    self.uv_uvtot_1_vol_wa.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_1_wa.append(v2[jj])
                                    self.vv_vvtot_1_vol_wa.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_1_wa.append(u2[jj])
                                    self.uu_uutot_1_vol_wa.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_1_wa.append(k[jj])
                                    self.k_ktot_1_vol_wa.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_1_wa.append(ens[jj])
                                    self.ens_enstot_1_vol_wa.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_1_wa.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_1_wa.append(shapvalues[jj]*1e3)
                                    self.shap_1_vol_wa.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_1_wa.append(uv_struc.event[jj])
                                else:
                                    self.volume_1_wd.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_1_wd.append(uv[jj])
                                    self.uv_uvtot_1_vol_wd.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_1_wd.append(v2[jj])
                                    self.vv_vvtot_1_vol_wd.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_1_wd.append(u2[jj])
                                    self.uu_uutot_1_vol_wd.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_1_wd.append(k[jj])
                                    self.k_ktot_1_vol_wd.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_1_wd.append(ens[jj])
                                    self.ens_enstot_1_vol_wd.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_1_wd.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_1_wd.append(shapvalues[jj]*1e3)
                                    self.shap_1_vol_wd.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_1_wd.append(uv_struc.event[jj])
                            elif uv_struc.event[jj] == 2:
                                self.volume_2.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_2.append(uv[jj])
                                self.uv_uvtot_2_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_2.append(uv_vol[jj]/uv_vol_sum)
                                self.vv_vvtot_2.append(v2[jj])
                                self.vv_vvtot_2_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                self.uu_uutot_2.append(u2[jj])
                                self.uu_uutot_2_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                self.k_ktot_2.append(k[jj])
                                self.k_ktot_2_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                self.k_vol_ktot_vol_2.append(k_vol[jj]/k_vol_sum)
                                self.ens_enstot_2.append(ens[jj])
                                self.ens_enstot_2_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                self.ens_vol_enstot_vol_2.append(ens_vol[jj]/ens_vol_sum)
                                self.shap_2.append(shapvalues[jj]*1e3)
                                self.shap_2_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_2.append(uv_struc.event[jj])
                                self.shap2cum += shapvalues[jj]
                                self.shap_vol2cum += shapvalues[jj]/uv_struc.vol[jj]
                                self.cdg_y_2.append(uv_struc.cdg_y[jj])
                                self.y_plus_min_2.append(yplus_min_ii)
                                if yplus_min_ii < 20:
                                    self.volume_2_wa.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_2_wa.append(uv[jj])
                                    self.uv_uvtot_2_vol_wa.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_2_wa.append(v2[jj])
                                    self.vv_vvtot_2_vol_wa.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_2_wa.append(u2[jj])
                                    self.uu_uutot_2_vol_wa.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_2_wa.append(k[jj])
                                    self.k_ktot_2_vol_wa.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_2_wa.append(ens[jj])
                                    self.ens_enstot_2_vol_wa.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_2_wa.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_2_wa.append(shapvalues[jj]*1e3)
                                    self.shap_2_vol_wa.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_2_wa.append(uv_struc.event[jj])
                                else:
                                    self.volume_2_wd.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_2_wd.append(uv[jj])
                                    self.uv_uvtot_2_vol_wd.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_2_wd.append(v2[jj])
                                    self.vv_vvtot_2_vol_wd.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_2_wd.append(u2[jj])
                                    self.uu_uutot_2_vol_wd.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_2_wd.append(k[jj])
                                    self.k_ktot_2_vol_wd.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_2_wd.append(ens[jj])
                                    self.ens_enstot_2_vol_wd.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_2_wd.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_2_wd.append(shapvalues[jj]*1e3)
                                    self.shap_2_vol_wd.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_2_wd.append(uv_struc.event[jj])
                            elif uv_struc.event[jj] == 3:
                                self.volume_3.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_3.append(uv[jj])
                                self.uv_uvtot_3_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_3.append(uv_vol[jj]/uv_vol_sum)
                                self.vv_vvtot_3.append(v2[jj])
                                self.vv_vvtot_3_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                self.uu_uutot_3.append(u2[jj])
                                self.uu_uutot_3_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                self.k_ktot_3.append(k[jj])
                                self.k_ktot_3_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                self.k_vol_ktot_vol_3.append(k_vol[jj]/k_vol_sum)
                                self.ens_enstot_3.append(ens[jj])
                                self.ens_enstot_3_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                self.ens_vol_enstot_vol_3.append(ens_vol[jj]/ens_vol_sum)
                                self.shap_3.append(shapvalues[jj]*1e3)
                                self.shap_3_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_3.append(uv_struc.event[jj])
                                self.shap3cum += shapvalues[jj]
                                self.shap_vol3cum += shapvalues[jj]/uv_struc.vol[jj]
                                self.cdg_y_3.append(uv_struc.cdg_y[jj])
                                self.y_plus_min_3.append(yplus_min_ii)
                                if yplus_min_ii < 20:
                                    self.volume_3_wa.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_3_wa.append(uv[jj])
                                    self.uv_uvtot_3_vol_wa.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_3_wa.append(v2[jj])
                                    self.vv_vvtot_3_vol_wa.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_3_wa.append(u2[jj])
                                    self.uu_uutot_3_vol_wa.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_3_wa.append(k[jj])
                                    self.k_ktot_3_vol_wa.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_3_wa.append(ens[jj])
                                    self.ens_enstot_3_vol_wa.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_3_wa.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_3_wa.append(shapvalues[jj]*1e3)
                                    self.shap_3_vol_wa.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_3_wa.append(uv_struc.event[jj])
                                else:
                                    self.volume_3_wd.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_3_wd.append(uv[jj])
                                    self.uv_uvtot_3_vol_wd.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_3_wd.append(v2[jj])
                                    self.vv_vvtot_3_vol_wd.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_3_wd.append(u2[jj])
                                    self.uu_uutot_3_vol_wd.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_3_wd.append(k[jj])
                                    self.k_ktot_3_vol_wd.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_3_wd.append(ens[jj])
                                    self.ens_enstot_3_vol_wd.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_3_wd.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_3_wd.append(shapvalues[jj]*1e3)
                                    self.shap_3_vol_wd.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_3_wd.append(uv_struc.event[jj])
                            elif uv_struc.event[jj] == 4:
                                self.volume_4.append(uv_struc.vol[jj]/1e6)
                                self.uv_uvtot_4.append(uv[jj])
                                self.uv_uvtot_4_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                self.uv_vol_uvtot_vol_4.append(uv_vol[jj]/uv_vol_sum)
                                self.vv_vvtot_4.append(v2[jj])
                                self.vv_vvtot_4_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                self.uu_uutot_4.append(u2[jj])
                                self.uu_uutot_4_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                self.k_ktot_4.append(k[jj])
                                self.k_ktot_4_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                self.k_vol_ktot_vol_4.append(k_vol[jj]/k_vol_sum)
                                self.ens_enstot_4.append(ens[jj])
                                self.ens_enstot_4_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                self.ens_vol_enstot_vol_4.append(ens_vol[jj]/ens_vol_sum)
                                self.shap_4.append(shapvalues[jj]*1e3)
                                self.shap_4_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                self.event_4.append(uv_struc.event[jj])
                                self.shap4cum += shapvalues[jj]
                                self.shap_vol4cum += shapvalues[jj]/uv_struc.vol[jj]
                                self.cdg_y_4.append(uv_struc.cdg_y[jj])
                                self.y_plus_min_4.append(yplus_min_ii)
                                if yplus_min_ii < 20:
                                    self.volume_4_wa.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_4_wa.append(uv[jj])
                                    self.uv_uvtot_4_vol_wa.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_4_wa.append(v2[jj])
                                    self.vv_vvtot_4_vol_wa.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_4_wa.append(u2[jj])
                                    self.uu_uutot_4_vol_wa.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_4_wa.append(k[jj])
                                    self.k_ktot_4_vol_wa.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_4_wa.append(ens[jj])
                                    self.ens_enstot_4_vol_wa.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_4_wa.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_4_wa.append(shapvalues[jj]*1e3)
                                    self.shap_4_vol_wa.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_4_wa.append(uv_struc.event[jj])
                                else:
                                    self.volume_4_wd.append(uv_struc.vol[jj]/1e6)
                                    self.uv_uvtot_4_wd.append(uv[jj])
                                    self.uv_uvtot_4_vol_wd.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                    self.vv_vvtot_4_wd.append(v2[jj])
                                    self.vv_vvtot_4_vol_wd.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                    self.uu_uutot_4_wd.append(u2[jj])
                                    self.uu_uutot_4_vol_wd.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                    self.k_ktot_4_wd.append(k[jj])
                                    self.k_ktot_4_vol_wd.append(k[jj]/uv_struc.vol[jj]*1e7)
                                    self.ens_enstot_4_wd.append(ens[jj])
                                    self.ens_enstot_4_vol_wd.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                    self.uv_vol_uvtot_vol_4_wd.append(uv_vol[jj]/uv_vol_sum)
                                    self.shap_4_wd.append(shapvalues[jj]*1e3)
                                    self.shap_4_vol_wd.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                    self.event_4_wd.append(uv_struc.event[jj])
                    vol_b = np.sum(normdata.vol)-np.sum(uv_struc.vol)
                    self.shapbcum += shapvalues[-1]
                    self.shap_volbcum += shapvalues[-1]/vol_b
                    self.shapback_list.append(shapback)
                    volback = np.sum(normdata.vol)-np.sum(uv_struc.vol)
                    self.shapbackvol_list.append(shapback/volback)
                except:
                    print('Missing: '+file+'.'+str(ii)+'.h5.shap')
            for ii_arlim1 in np.arange(ngrid):
                for ii_arlim2 in np.arange(ngrid):
                    if self.npoin1[ii_arlim1,ii_arlim2] == 0:
                        self.SHAP_grid1[ii_arlim1,ii_arlim2] = np.nan
                        self.SHAP_grid1vol[ii_arlim1,ii_arlim2] = np.nan
                    else:
                        self.SHAP_grid1[ii_arlim1,ii_arlim2] /= self.npoin1[ii_arlim1,ii_arlim2]
                        self.SHAP_grid1vol[ii_arlim1,ii_arlim2] /= self.npoin1[ii_arlim1,ii_arlim2]
                    if self.npoin2[ii_arlim1,ii_arlim2] == 0:
                        self.SHAP_grid2[ii_arlim1,ii_arlim2] = np.nan
                        self.SHAP_grid2vol[ii_arlim1,ii_arlim2] = np.nan
                    else:
                        self.SHAP_grid2[ii_arlim1,ii_arlim2] /= self.npoin2[ii_arlim1,ii_arlim2]
                        self.SHAP_grid2vol[ii_arlim1,ii_arlim2] /= self.npoin2[ii_arlim1,ii_arlim2]
                    if self.npoin3[ii_arlim1,ii_arlim2] == 0:
                        self.SHAP_grid3[ii_arlim1,ii_arlim2] = np.nan
                        self.SHAP_grid3vol[ii_arlim1,ii_arlim2] = np.nan
                    else:
                        self.SHAP_grid3[ii_arlim1,ii_arlim2] /= self.npoin3[ii_arlim1,ii_arlim2]
                        self.SHAP_grid3vol[ii_arlim1,ii_arlim2] /= self.npoin3[ii_arlim1,ii_arlim2]
                    if self.npoin4[ii_arlim1,ii_arlim2] == 0:
                        self.SHAP_grid4[ii_arlim1,ii_arlim2] = np.nan
                        self.SHAP_grid4vol[ii_arlim1,ii_arlim2] = np.nan
                    else:
                        self.SHAP_grid4[ii_arlim1,ii_arlim2] /= self.npoin4[ii_arlim1,ii_arlim2]
                        self.SHAP_grid4vol[ii_arlim1,ii_arlim2] /= self.npoin4[ii_arlim1,ii_arlim2]
            file_save = open('Qinfo.txt', "w+") 
            file_save.write('uv_Q- '+str(uv_Qminus/uvtot_cum)+'\n')
            file_save.write('vol_Q- '+str(vol_Qminus/voltot_cum)+'\n')
            file_save.write('uv_Q-_wa '+str(uv_Qminus_wa/uvtot_cum)+'\n')
            file_save.write('vol_Q-_wa '+str(vol_Qminus_wa/voltot_cum)+'\n')
            file_save.close()
            hf = h5py.File(fileread, 'w')
            hf.create_dataset('volume_wa', data=self.volume_wa)
            hf.create_dataset('volume_wd', data=self.volume_wd)
            hf.create_dataset('shap_wa', data=self.shap_wa)
            hf.create_dataset('shap_wd', data=self.shap_wd)
            hf.create_dataset('shap_wa_vol', data=self.shap_wa_vol)
            hf.create_dataset('shap_wd_vol', data=self.shap_wd_vol)
            hf.create_dataset('event_wa', data=self.event_wa)
            hf.create_dataset('event_wd', data=self.event_wd)
            hf.create_dataset('uv_uvtot_wa', data=self.uv_uvtot_wa)
            hf.create_dataset('uv_uvtot_wd', data=self.uv_uvtot_wd)
            hf.create_dataset('uv_uvtot_wa_vol', data=self.uv_uvtot_wa_vol)
            hf.create_dataset('uv_uvtot_wd_vol', data=self.uv_uvtot_wd_vol)
            hf.create_dataset('uv_vol_uvtot_vol_wa', data=self.uv_vol_uvtot_vol_wa)
            hf.create_dataset('uv_vol_uvtot_vol_wd', data=self.uv_vol_uvtot_vol_wd)
            hf.create_dataset('volume_1', data=self.volume_1)
            hf.create_dataset('volume_2', data=self.volume_2)
            hf.create_dataset('volume_3', data=self.volume_3)
            hf.create_dataset('volume_4', data=self.volume_4)
            hf.create_dataset('shap_1', data=self.shap_1)
            hf.create_dataset('shap_2', data=self.shap_2)
            hf.create_dataset('shap_3', data=self.shap_3)
            hf.create_dataset('shap_4', data=self.shap_4)
            hf.create_dataset('shap_1_vol', data=self.shap_1_vol)
            hf.create_dataset('shap_2_vol', data=self.shap_2_vol)
            hf.create_dataset('shap_3_vol', data=self.shap_3_vol)
            hf.create_dataset('shap_4_vol', data=self.shap_4_vol)
            hf.create_dataset('uv_uvtot_1', data=self.uv_uvtot_1)
            hf.create_dataset('uv_uvtot_2', data=self.uv_uvtot_2)
            hf.create_dataset('uv_uvtot_3', data=self.uv_uvtot_3)
            hf.create_dataset('uv_uvtot_4', data=self.uv_uvtot_4)
            hf.create_dataset('uv_uvtot_1_vol', data=self.uv_uvtot_1_vol)
            hf.create_dataset('uv_uvtot_2_vol', data=self.uv_uvtot_2_vol)
            hf.create_dataset('uv_uvtot_3_vol', data=self.uv_uvtot_3_vol)
            hf.create_dataset('uv_uvtot_4_vol', data=self.uv_uvtot_4_vol)
            hf.create_dataset('uv_vol_uvtot_vol_1', data=self.uv_vol_uvtot_vol_1)
            hf.create_dataset('uv_vol_uvtot_vol_2', data=self.uv_vol_uvtot_vol_2)
            hf.create_dataset('uv_vol_uvtot_vol_3', data=self.uv_vol_uvtot_vol_3)
            hf.create_dataset('uv_vol_uvtot_vol_4', data=self.uv_vol_uvtot_vol_4)
            hf.create_dataset('vv_vvtot_1', data=self.vv_vvtot_1)
            hf.create_dataset('vv_vvtot_2', data=self.vv_vvtot_2)
            hf.create_dataset('vv_vvtot_3', data=self.vv_vvtot_3)
            hf.create_dataset('vv_vvtot_4', data=self.vv_vvtot_4)
            hf.create_dataset('vv_vvtot_1_vol', data=self.vv_vvtot_1_vol)
            hf.create_dataset('vv_vvtot_2_vol', data=self.vv_vvtot_2_vol)
            hf.create_dataset('vv_vvtot_3_vol', data=self.vv_vvtot_3_vol)
            hf.create_dataset('vv_vvtot_4_vol', data=self.vv_vvtot_4_vol)
            hf.create_dataset('uu_uutot_1', data=self.uu_uutot_1)
            hf.create_dataset('uu_uutot_2', data=self.uu_uutot_2)
            hf.create_dataset('uu_uutot_3', data=self.uu_uutot_3)
            hf.create_dataset('uu_uutot_4', data=self.uu_uutot_4)
            hf.create_dataset('uu_uutot_1_vol', data=self.uu_uutot_1_vol)
            hf.create_dataset('uu_uutot_2_vol', data=self.uu_uutot_2_vol)
            hf.create_dataset('uu_uutot_3_vol', data=self.uu_uutot_3_vol)
            hf.create_dataset('uu_uutot_4_vol', data=self.uu_uutot_4_vol)
            hf.create_dataset('k_ktot_1', data=self.k_ktot_1)
            hf.create_dataset('k_ktot_2', data=self.k_ktot_2)
            hf.create_dataset('k_ktot_3', data=self.k_ktot_3)
            hf.create_dataset('k_ktot_4', data=self.k_ktot_4)
            hf.create_dataset('k_ktot_1_vol', data=self.k_ktot_1_vol)
            hf.create_dataset('k_ktot_2_vol', data=self.k_ktot_2_vol)
            hf.create_dataset('k_ktot_3_vol', data=self.k_ktot_3_vol)
            hf.create_dataset('k_ktot_4_vol', data=self.k_ktot_4_vol)
            hf.create_dataset('k_vol_ktot_vol_1', data=self.k_vol_ktot_vol_1)
            hf.create_dataset('k_vol_ktot_vol_2', data=self.k_vol_ktot_vol_2)
            hf.create_dataset('k_vol_ktot_vol_3', data=self.k_vol_ktot_vol_3)
            hf.create_dataset('k_vol_ktot_vol_4', data=self.k_vol_ktot_vol_4)
            hf.create_dataset('ens_enstot_1', data=self.ens_enstot_1)
            hf.create_dataset('ens_enstot_2', data=self.ens_enstot_2)
            hf.create_dataset('ens_enstot_3', data=self.ens_enstot_3)
            hf.create_dataset('ens_enstot_4', data=self.ens_enstot_4)
            hf.create_dataset('ens_enstot_1_vol', data=self.ens_enstot_1_vol)
            hf.create_dataset('ens_enstot_2_vol', data=self.ens_enstot_2_vol)
            hf.create_dataset('ens_enstot_3_vol', data=self.ens_enstot_3_vol)
            hf.create_dataset('ens_enstot_4_vol', data=self.ens_enstot_4_vol)
            hf.create_dataset('ens_vol_enstot_vol_1', data=self.ens_vol_enstot_vol_1)
            hf.create_dataset('ens_vol_enstot_vol_2', data=self.ens_vol_enstot_vol_2)
            hf.create_dataset('ens_vol_enstot_vol_3', data=self.ens_vol_enstot_vol_3)
            hf.create_dataset('ens_vol_enstot_vol_4', data=self.ens_vol_enstot_vol_4)
            hf.create_dataset('event_1', data=self.event_1)
            hf.create_dataset('event_2', data=self.event_2)
            hf.create_dataset('event_3', data=self.event_3)
            hf.create_dataset('event_4', data=self.event_4)
            hf.create_dataset('volume_1_wa', data=self.volume_1_wa)
            hf.create_dataset('volume_2_wa', data=self.volume_2_wa)
            hf.create_dataset('volume_3_wa', data=self.volume_3_wa)
            hf.create_dataset('volume_4_wa', data=self.volume_4_wa)
            hf.create_dataset('shap_1_wa', data=self.shap_1_wa)
            hf.create_dataset('shap_2_wa', data=self.shap_2_wa)
            hf.create_dataset('shap_3_wa', data=self.shap_3_wa)
            hf.create_dataset('shap_4_wa', data=self.shap_4_wa)
            hf.create_dataset('shap_1_vol_wa', data=self.shap_1_vol_wa)
            hf.create_dataset('shap_2_vol_wa', data=self.shap_2_vol_wa)
            hf.create_dataset('shap_3_vol_wa', data=self.shap_3_vol_wa)
            hf.create_dataset('shap_4_vol_wa', data=self.shap_4_vol_wa)
            hf.create_dataset('uv_uvtot_1_wa', data=self.uv_uvtot_1_wa)
            hf.create_dataset('uv_uvtot_2_wa', data=self.uv_uvtot_2_wa)
            hf.create_dataset('uv_uvtot_3_wa', data=self.uv_uvtot_3_wa)
            hf.create_dataset('uv_uvtot_4_wa', data=self.uv_uvtot_4_wa)
            hf.create_dataset('uv_uvtot_1_vol_wa', data=self.uv_uvtot_1_vol_wa)
            hf.create_dataset('uv_uvtot_2_vol_wa', data=self.uv_uvtot_2_vol_wa)
            hf.create_dataset('uv_uvtot_3_vol_wa', data=self.uv_uvtot_3_vol_wa)
            hf.create_dataset('uv_uvtot_4_vol_wa', data=self.uv_uvtot_4_vol_wa)
            hf.create_dataset('vv_vvtot_1_wa', data=self.vv_vvtot_1_wa)
            hf.create_dataset('vv_vvtot_2_wa', data=self.vv_vvtot_2_wa)
            hf.create_dataset('vv_vvtot_3_wa', data=self.vv_vvtot_3_wa)
            hf.create_dataset('vv_vvtot_4_wa', data=self.vv_vvtot_4_wa)
            hf.create_dataset('vv_vvtot_1_vol_wa', data=self.vv_vvtot_1_vol_wa)
            hf.create_dataset('vv_vvtot_2_vol_wa', data=self.vv_vvtot_2_vol_wa)
            hf.create_dataset('vv_vvtot_3_vol_wa', data=self.vv_vvtot_3_vol_wa)
            hf.create_dataset('vv_vvtot_4_vol_wa', data=self.vv_vvtot_4_vol_wa)
            hf.create_dataset('uu_uutot_1_wa', data=self.uu_uutot_1_wa)
            hf.create_dataset('uu_uutot_2_wa', data=self.uu_uutot_2_wa)
            hf.create_dataset('uu_uutot_3_wa', data=self.uu_uutot_3_wa)
            hf.create_dataset('uu_uutot_4_wa', data=self.uu_uutot_4_wa)
            hf.create_dataset('uu_uutot_1_vol_wa', data=self.uu_uutot_1_vol_wa)
            hf.create_dataset('uu_uutot_2_vol_wa', data=self.uu_uutot_2_vol_wa)
            hf.create_dataset('uu_uutot_3_vol_wa', data=self.uu_uutot_3_vol_wa)
            hf.create_dataset('uu_uutot_4_vol_wa', data=self.uu_uutot_4_vol_wa)
            hf.create_dataset('k_ktot_1_wa', data=self.k_ktot_1_wa)
            hf.create_dataset('k_ktot_2_wa', data=self.k_ktot_2_wa)
            hf.create_dataset('k_ktot_3_wa', data=self.k_ktot_3_wa)
            hf.create_dataset('k_ktot_4_wa', data=self.k_ktot_4_wa)
            hf.create_dataset('k_ktot_1_vol_wa', data=self.k_ktot_1_vol_wa)
            hf.create_dataset('k_ktot_2_vol_wa', data=self.k_ktot_2_vol_wa)
            hf.create_dataset('k_ktot_3_vol_wa', data=self.k_ktot_3_vol_wa)
            hf.create_dataset('k_ktot_4_vol_wa', data=self.k_ktot_4_vol_wa)
            hf.create_dataset('ens_enstot_1_wa', data=self.ens_enstot_1_wa)
            hf.create_dataset('ens_enstot_2_wa', data=self.ens_enstot_2_wa)
            hf.create_dataset('ens_enstot_3_wa', data=self.ens_enstot_3_wa)
            hf.create_dataset('ens_enstot_4_wa', data=self.ens_enstot_4_wa)
            hf.create_dataset('ens_enstot_1_vol_wa', data=self.ens_enstot_1_vol_wa)
            hf.create_dataset('ens_enstot_2_vol_wa', data=self.ens_enstot_2_vol_wa)
            hf.create_dataset('ens_enstot_3_vol_wa', data=self.ens_enstot_3_vol_wa)
            hf.create_dataset('ens_enstot_4_vol_wa', data=self.ens_enstot_4_vol_wa)
            hf.create_dataset('uv_vol_uvtot_vol_1_wa', data=self.uv_vol_uvtot_vol_1_wa)
            hf.create_dataset('uv_vol_uvtot_vol_2_wa', data=self.uv_vol_uvtot_vol_2_wa)
            hf.create_dataset('uv_vol_uvtot_vol_3_wa', data=self.uv_vol_uvtot_vol_3_wa)
            hf.create_dataset('uv_vol_uvtot_vol_4_wa', data=self.uv_vol_uvtot_vol_4_wa)
            hf.create_dataset('event_1_wa', data=self.event_1_wa)
            hf.create_dataset('event_2_wa', data=self.event_2_wa)
            hf.create_dataset('event_3_wa', data=self.event_3_wa)
            hf.create_dataset('event_4_wa', data=self.event_4_wa)
            hf.create_dataset('volume_1_wd', data=self.volume_1_wd)
            hf.create_dataset('volume_2_wd', data=self.volume_2_wd)
            hf.create_dataset('volume_3_wd', data=self.volume_3_wd)
            hf.create_dataset('volume_4_wd', data=self.volume_4_wd)
            hf.create_dataset('shap_1_wd', data=self.shap_1_wd)
            hf.create_dataset('shap_2_wd', data=self.shap_2_wd)
            hf.create_dataset('shap_3_wd', data=self.shap_3_wd)
            hf.create_dataset('shap_4_wd', data=self.shap_4_wd)
            hf.create_dataset('shap_1_vol_wd', data=self.shap_1_vol_wd)
            hf.create_dataset('shap_2_vol_wd', data=self.shap_2_vol_wd)
            hf.create_dataset('shap_3_vol_wd', data=self.shap_3_vol_wd)
            hf.create_dataset('shap_4_vol_wd', data=self.shap_4_vol_wd)
            hf.create_dataset('uv_uvtot_1_wd', data=self.uv_uvtot_1_wd)
            hf.create_dataset('uv_uvtot_2_wd', data=self.uv_uvtot_2_wd)
            hf.create_dataset('uv_uvtot_3_wd', data=self.uv_uvtot_3_wd)
            hf.create_dataset('uv_uvtot_4_wd', data=self.uv_uvtot_4_wd)
            hf.create_dataset('vv_vvtot_1_wd', data=self.vv_vvtot_1_wd)
            hf.create_dataset('vv_vvtot_2_wd', data=self.vv_vvtot_2_wd)
            hf.create_dataset('vv_vvtot_3_wd', data=self.vv_vvtot_3_wd)
            hf.create_dataset('vv_vvtot_4_wd', data=self.vv_vvtot_4_wd)
            hf.create_dataset('uu_uutot_1_wd', data=self.uu_uutot_1_wd)
            hf.create_dataset('uu_uutot_2_wd', data=self.uu_uutot_2_wd)
            hf.create_dataset('uu_uutot_3_wd', data=self.uu_uutot_3_wd)
            hf.create_dataset('uu_uutot_4_wd', data=self.uu_uutot_4_wd)
            hf.create_dataset('k_ktot_1_wd', data=self.k_ktot_1_wd)
            hf.create_dataset('k_ktot_2_wd', data=self.k_ktot_2_wd)
            hf.create_dataset('k_ktot_3_wd', data=self.k_ktot_3_wd)
            hf.create_dataset('k_ktot_4_wd', data=self.k_ktot_4_wd)
            hf.create_dataset('k_ktot_1_vol_wd', data=self.k_ktot_1_vol_wd)
            hf.create_dataset('k_ktot_2_vol_wd', data=self.k_ktot_2_vol_wd)
            hf.create_dataset('k_ktot_3_vol_wd', data=self.k_ktot_3_vol_wd)
            hf.create_dataset('k_ktot_4_vol_wd', data=self.k_ktot_4_vol_wd)
            hf.create_dataset('ens_enstot_1_wd', data=self.ens_enstot_1_wd)
            hf.create_dataset('ens_enstot_2_wd', data=self.ens_enstot_2_wd)
            hf.create_dataset('ens_enstot_3_wd', data=self.ens_enstot_3_wd)
            hf.create_dataset('ens_enstot_4_wd', data=self.ens_enstot_4_wd)
            hf.create_dataset('ens_enstot_1_vol_wd', data=self.ens_enstot_1_vol_wd)
            hf.create_dataset('ens_enstot_2_vol_wd', data=self.ens_enstot_2_vol_wd)
            hf.create_dataset('ens_enstot_3_vol_wd', data=self.ens_enstot_3_vol_wd)
            hf.create_dataset('ens_enstot_4_vol_wd', data=self.ens_enstot_4_vol_wd)
            hf.create_dataset('uv_uvtot_1_vol_wd', data=self.uv_uvtot_1_vol_wd)
            hf.create_dataset('uv_uvtot_2_vol_wd', data=self.uv_uvtot_2_vol_wd)
            hf.create_dataset('uv_uvtot_3_vol_wd', data=self.uv_uvtot_3_vol_wd)
            hf.create_dataset('uv_uvtot_4_vol_wd', data=self.uv_uvtot_4_vol_wd)
            hf.create_dataset('vv_vvtot_1_vol_wd', data=self.vv_vvtot_1_vol_wd)
            hf.create_dataset('vv_vvtot_2_vol_wd', data=self.vv_vvtot_2_vol_wd)
            hf.create_dataset('vv_vvtot_3_vol_wd', data=self.vv_vvtot_3_vol_wd)
            hf.create_dataset('vv_vvtot_4_vol_wd', data=self.vv_vvtot_4_vol_wd)
            hf.create_dataset('uu_uutot_1_vol_wd', data=self.uu_uutot_1_vol_wd)
            hf.create_dataset('uu_uutot_2_vol_wd', data=self.uu_uutot_2_vol_wd)
            hf.create_dataset('uu_uutot_3_vol_wd', data=self.uu_uutot_3_vol_wd)
            hf.create_dataset('uu_uutot_4_vol_wd', data=self.uu_uutot_4_vol_wd)
            hf.create_dataset('uv_vol_uvtot_vol_1_wd', data=self.uv_vol_uvtot_vol_1_wd)
            hf.create_dataset('uv_vol_uvtot_vol_2_wd', data=self.uv_vol_uvtot_vol_2_wd)
            hf.create_dataset('uv_vol_uvtot_vol_3_wd', data=self.uv_vol_uvtot_vol_3_wd)
            hf.create_dataset('uv_vol_uvtot_vol_4_wd', data=self.uv_vol_uvtot_vol_4_wd)
            hf.create_dataset('event_1_wd', data=self.event_1_wd)
            hf.create_dataset('event_2_wd', data=self.event_2_wd)
            hf.create_dataset('event_3_wd', data=self.event_3_wd)
            hf.create_dataset('event_4_wd', data=self.event_4_wd)
            hf.create_dataset('voltot', data=self.voltot)
            hf.create_dataset('shapback_list', data=self.shapback_list)
            hf.create_dataset('shapbackvol_list', data=self.shapbackvol_list)
            hf.create_dataset('AR1_grid', data=self.AR1_grid)
            hf.create_dataset('AR2_grid', data=self.AR2_grid)
            hf.create_dataset('SHAP_grid1', data=self.SHAP_grid1)
            hf.create_dataset('SHAP_grid2', data=self.SHAP_grid2)
            hf.create_dataset('SHAP_grid3', data=self.SHAP_grid3)
            hf.create_dataset('SHAP_grid4', data=self.SHAP_grid4)
            hf.create_dataset('SHAP_grid1vol', data=self.SHAP_grid1vol)
            hf.create_dataset('SHAP_grid2vol', data=self.SHAP_grid2vol)
            hf.create_dataset('SHAP_grid3vol', data=self.SHAP_grid3vol)
            hf.create_dataset('SHAP_grid4vol', data=self.SHAP_grid4vol)
            hf.create_dataset('npoin1', data=self.npoin1)
            hf.create_dataset('npoin2', data=self.npoin2)
            hf.create_dataset('npoin3', data=self.npoin3)
            hf.create_dataset('npoin4', data=self.npoin4)
            hf.create_dataset('shap1cum', data=self.shap1cum)
            hf.create_dataset('shap2cum', data=self.shap2cum)
            hf.create_dataset('shap3cum', data=self.shap3cum)
            hf.create_dataset('shap4cum', data=self.shap4cum)
            hf.create_dataset('shapbcum', data=self.shapbcum)
            hf.create_dataset('shap_vol1cum', data=self.shap_vol1cum)
            hf.create_dataset('shap_vol2cum', data=self.shap_vol2cum)
            hf.create_dataset('shap_vol3cum', data=self.shap_vol3cum)
            hf.create_dataset('shap_vol4cum', data=self.shap_vol4cum)
            hf.create_dataset('shap_volbcum', data=self.shap_volbcum)
            hf.create_dataset('shapmax', data=self.shapmax)
            hf.create_dataset('shapmin', data=self.shapmin)
            hf.create_dataset('shapmaxvol', data=self.shapmaxvol)
            hf.create_dataset('shapminvol', data=self.shapminvol)
            hf.create_dataset('cdg_y_1', data=self.cdg_y_1)
            hf.create_dataset('cdg_y_2', data=self.cdg_y_2)
            hf.create_dataset('cdg_y_3', data=self.cdg_y_3)
            hf.create_dataset('cdg_y_4', data=self.cdg_y_4)
            hf.create_dataset('cdg_y_wa', data=self.cdg_y_wa)
            hf.create_dataset('cdg_y_wd', data=self.cdg_y_wd)
            hf.create_dataset('y_plus_min_1', data=self.y_plus_min_1)
            hf.create_dataset('y_plus_min_2', data=self.y_plus_min_2)
            hf.create_dataset('y_plus_min_3', data=self.y_plus_min_3)
            hf.create_dataset('y_plus_min_4', data=self.y_plus_min_4)
            hf.close()
            
            self.cdg_y_1 = (1-np.abs(self.cdg_y_1))*utau/ny # convert to y plus
            self.cdg_y_2 = (1-np.abs(self.cdg_y_2))*utau/ny
            self.cdg_y_3 = (1-np.abs(self.cdg_y_3))*utau/ny
            self.cdg_y_4 = (1-np.abs(self.cdg_y_4))*utau/ny
            
            

    def read_data_simple(self,start,end,step,\
                         # file='../../../data2/cremades/P125_21pi_vu_SHAP_ann4_divide/P125_21pi_vu',\
                         dir_shap = './',
                         dir_struc = './',
                         dir_uvw = './',
                         dir_save = './',
                         dir_grad='./',
                         structure = 'streak',
                         dataset = 'P125_21pi_vu',
                         mode = 'mse',
                         # fileQ='../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu',\
                         # fileuvw='../P125_21pi_vu/P125_21pi_vu',\
                         fileUmean="Umean.txt",
                         filenorm="norm.txt",
                         shapmin=-8,
                         shapmax=5,
                         shapminvol=-8,
                         shapmaxvol=8,
                         nbars=1000,                     
                         absolute=True,
                         volmin=2.7e4,
                         readdata=False,
                         editq3=False,
                         find_files=False):
        """
        Function for reading the data
        """
        import h5py
        import os
        
        
        if mode == 'mse':
            fileread=f'data_plots.h5.{structure}'
            file_shap=dir_shap+f'{dataset}_{structure}_SHAP_mse/{dataset}'
        elif mode == 'cf':
            fileread=f'data_plots_cf.h5.{structure}'
            file_shap=dir_shap+f'{dataset}_{structure}_SHAP_cf/{dataset}'
        
        fileuvw=dir_uvw+f'{dataset}/{dataset}'
        filegrad=dir_grad+f'{dataset}/grad/{dataset}'
        
        
        # check if attributes exist and if not create empty dict
        
        attr = ['volume_wa', 'volume_wd', 'shap_wa', 'shap_wd',
                'shap_wa_vol', 'shap_wd_vol', 'event_wa', 
                'event_wd', 'uv_uvtot_wa', 'uv_uvtot_wd', 
                'vv_vvtot_wa', 'vv_vvtot_wd',
                'uu_uutot_wa', 'uu_uutot_wd',
                'k_ktot_wa', 'k_ktot_wd',
                'ens_enstot_wa', 'ens_enstot_wd',
                'uv_uvtot_wa_vol', 'uv_uvtot_wd_vol', 
                'vv_vvtot_wa_vol', 'vv_vvtot_wd_vol', 
                'uu_uutot_wa_vol', 'uu_uutot_wd_vol',
                'k_ktot_wa_vol', 'k_ktot_wd_vol',
                'ens_enstot_wa_vol', 'ens_enstot_wd_vol',
                'uv_vol_uvtot_vol_wa', 'uv_vol_uvtot_vol_wd', 
                'volume', 'volume_wa', 'shap', 'shap_vol', 
                'uv_uvtot', 'uv_uvtot_vol', 'uv_vol_uvtot_vol', 
                'vv_vvtot', 'vv_vvtot_vol', 'vv_vol_vvtot_vol',
                'uu_uutot', 'uu_uutot_vol', 'uu_vol_uutot_vol',
                'k_ktot', 'k_ktot_vol', 'k_vol_ktot_vol',
                'ens_enstot', 'ens_enstot_vol', 'ens_vol_enstot_vol',
                'event', 'voltot', 'shapback_list', 
                'shapbackvol_list', 'AR1_grid', 'AR2_grid', 
                'SHAP_grid', 'SHAP_gridvol', 'npoin', 'shapcum', 
                'shapbcum', 'shap_volcum', 'shap_volbcum', 
                'shapmax', 'shapmin', 'shapmaxvol', 'shapminvol', 
                'cdg_y', 'cdg_y_wa', 'cdg_y_wd', 'y_plus_min']
        
        for att in attr:
            if not hasattr(self, att):
                exec(f'self.{att}'+' = {}')
            else:
                try:
                    exec(f'self.{att}["test"] = False')
                except:
                    exec(f'self.{att}'+' = {}')
                    
        # self.shapmin = shapmin
        # self.shapmax = shapmax
        # self.shapminvol = shapminvol
        # self.shapmaxvol = shapmaxvol
        # self.nbars = nbars
        # self.volmin = volmin
        if absolute:
            if mode == 'mse':
                self.ylabel_shap = r'$|\phi_{\overrightarrow{u}}| \cdot 10^{-3}$'
                self.ylabel_shap_vol = r'$|\phi_{\overrightarrow{u}}/V^+| \cdot 10^{-9}$'
                self.clabel_shap = r'$|\phi_{\overrightarrow{u}}|$'
                self.clabel_shap_vol = r'$|\phi_{\overrightarrow{u}} /V^+|$'
            elif mode == 'cf':
                self.ylabel_shap = r'$|\phi_{\tau_{w}}| \cdot 10^{-3}$'
                self.ylabel_shap_vol = r'$|\phi_{\tau_{w}}/V^+| \cdot 10^{-9}$'
                self.clabel_shap = r'$|\phi_{\tau_{w}}|$'
                self.clabel_shap_vol = r'$|\phi_{\tau_{w}} /V^+|$'
        else:
            self.ylabel_shap = '$\phi_i \cdot 10^{-3}$'
            self.ylabel_shap_vol = '$\phi_i/V^+ \cdot 10^{-9}$'
            self.clabel_shap = '$\phi_i$'
            self.clabel_shap_vol = '$\phi_i /V^+$'
        import get_data_fun as gd
        normdata = gd.get_data_norm(file_read=fileuvw,
                                    file_grad=filegrad)
        normdata.geom_param(start,1,1,1)
        utau = normdata.vtau
        ny = normdata.ny
        if readdata:
            hf = h5py.File(fileread, 'r')
            self.volume_wa[structure] = np.array(hf['volume_wa'])
            self.volume_wd[structure] = np.array(hf['volume_wd'])
            self.shap_wa[structure] = np.array(hf['shap_wa'])
            self.shap_wd[structure] = np.array(hf['shap_wd'])
            self.shap_wa_vol[structure] = np.array(hf['shap_wa_vol'])
            self.shap_wd_vol[structure] = np.array(hf['shap_wd_vol'])
            self.event_wa[structure] = np.array(hf['event_wa'])
            self.event_wd[structure] = np.array(hf['event_wd'])
            self.uv_uvtot_wa[structure] = np.array(hf['uv_uvtot_wa'])
            self.uv_uvtot_wd[structure] = np.array(hf['uv_uvtot_wd'])
            self.uv_uvtot_wa_vol[structure] = np.array(hf['uv_uvtot_wa_vol'])
            self.uv_uvtot_wd_vol[structure] = np.array(hf['uv_uvtot_wd_vol'])
            self.vv_vvtot_wa[structure] = np.array(hf['vv_vvtot_wa'])
            self.vv_vvtot_wd[structure] = np.array(hf['vv_vvtot_wd'])
            self.vv_vvtot_wa_vol[structure] = np.array(hf['vv_vvtot_wa_vol'])
            self.vv_vvtot_wd_vol[structure] = np.array(hf['vv_vvtot_wd_vol'])
            self.uu_uutot_wa[structure] = np.array(hf['uu_uutot_wa'])
            self.uu_uutot_wd[structure] = np.array(hf['uu_uutot_wd'])
            self.uu_uutot_wa_vol[structure] = np.array(hf['uu_uutot_wa_vol'])
            self.uu_uutot_wd_vol[structure] = np.array(hf['uu_uutot_wd_vol'])
            self.k_ktot_wa[structure] = np.array(hf['k_ktot_wa'])
            self.k_ktot_wd[structure] = np.array(hf['k_ktot_wd'])
            self.k_ktot_wa_vol[structure] = np.array(hf['k_ktot_wa_vol'])
            self.k_ktot_wd_vol[structure] = np.array(hf['k_ktot_wd_vol'])
            self.ens_enstot_wa[structure] = np.array(hf['ens_enstot_wa'])
            self.ens_enstot_wd[structure] = np.array(hf['ens_enstot_wd'])
            self.ens_enstot_wa_vol[structure] = np.array(hf['ens_enstot_wa_vol'])
            self.ens_enstot_wd_vol[structure] = np.array(hf['ens_enstot_wd_vol'])
            self.uv_vol_uvtot_vol_wa[structure] = np.array(hf['uv_vol_uvtot_vol_wa'])
            self.uv_vol_uvtot_vol_wd[structure] = np.array(hf['uv_vol_uvtot_vol_wd'])
            self.volume[structure] = np.array(hf['volume'])
            self.volume_wa[structure] = np.array(hf['volume_wa'])
            self.shap[structure] = np.array(hf['shap'])
            self.shap_vol[structure] = np.array(hf['shap_vol'])
            self.uv_uvtot[structure] = np.array(hf['uv_uvtot'])
            self.uv_uvtot_vol[structure] = np.array(hf['uv_uvtot_vol'])
            self.uv_vol_uvtot_vol[structure] = np.array(hf['uv_vol_uvtot_vol'])
            self.vv_vvtot[structure] = np.array(hf['vv_vvtot'])
            self.vv_vvtot_vol[structure] = np.array(hf['vv_vvtot_vol'])
            self.uu_uutot[structure] = np.array(hf['uu_uutot'])
            self.uu_uutot_vol[structure] = np.array(hf['uu_uutot_vol'])
            self.k_ktot[structure] = np.array(hf['k_ktot'])
            self.k_ktot_vol[structure] = np.array(hf['k_ktot_vol'])
            self.k_vol_ktot_vol[structure] = np.array(hf['k_vol_ktot_vol'])
            self.ens_enstot[structure] = np.array(hf['ens_enstot'])
            self.ens_enstot_vol[structure] = np.array(hf['ens_enstot_vol'])
            self.ens_vol_enstot_vol[structure] = np.array(hf['ens_vol_enstot_vol'])
            self.event[structure] = np.array(hf['event'])
            self.voltot[structure] = np.array(hf['voltot'])
            self.shapback_list[structure] = np.array(hf['shapback_list'])
            self.shapbackvol_list[structure] = np.array(hf['shapbackvol_list'])
            self.AR1_grid[structure] = np.array(hf['AR1_grid'])
            self.AR2_grid[structure] = np.array(hf['AR2_grid'])
            self.SHAP_grid[structure] = np.array(hf['SHAP_grid'])
            self.SHAP_gridvol[structure] = np.array(hf['SHAP_gridvol'])
            self.npoin[structure] = np.array(hf['npoin'])
            self.shapcum[structure] = np.array(hf['shapcum'])
            self.shapbcum[structure] = np.array(hf['shapbcum'])
            self.shap_volcum[structure] = np.array(hf['shap_volcum'])
            self.shap_volbcum[structure] = np.array(hf['shap_volbcum'])
            self.shapmax[structure] = np.array(hf['shapmax'])
            self.shapmin[structure] = np.array(hf['shapmin'])
            self.shapmaxvol[structure] = np.array(hf['shapmaxvol'])
            self.shapminvol[structure] = np.array(hf['shapminvol'])
            self.cdg_y[structure] = (1-np.abs(np.array(hf['cdg_y'])))*utau/ny # convert to y plus
            self.cdg_y_wa[structure] = np.array(hf['cdg_y_wa'])
            self.cdg_y_wd[structure] = np.array(hf['cdg_y_wd'])
            self.y_plus_min[structure] = np.array(hf['y_plus_min'])
            hf.close()
        else:
            volume_wa = []
            volume_wd= []
            shap_wa = []
            shap_wd = []
            shap_wa_vol = []
            shap_wd_vol = []
            event_wa = []
            event_wd = []
            uv_uvtot_wa = []
            uv_uvtot_wd = []
            uv_uvtot_wa_vol = []
            uv_uvtot_wd_vol = []
            vv_vvtot_wa = []
            vv_vvtot_wd = []
            vv_vvtot_wa_vol = []
            vv_vvtot_wd_vol = []
            uu_uutot_wa = []
            uu_uutot_wd = []
            uu_uutot_wa_vol = []
            uu_uutot_wd_vol = []
            k_ktot_wa = []
            k_ktot_wd = []
            k_ktot_wa_vol = []
            k_ktot_wd_vol = []
            ens_enstot_wa = []
            ens_enstot_wd = []
            ens_enstot_wa_vol = []
            ens_enstot_wd_vol = []
            uv_vol_uvtot_vol_wa = []
            uv_vol_uvtot_vol_wd = []
            cdg_y_wa = []
            cdg_y_wd = []
            volume = []
            uv_uvtot = []
            uv_uvtot_vol = []
            uv_vol_uvtot_vol = []
            vv_vvtot = []
            vv_vvtot_vol = []
            vv_vol_vvtot_vol = []
            uu_uutot = []
            uu_uutot_vol = []
            uu_vol_uutot_vol = []
            k_ktot = []
            k_ktot_vol = []
            k_vol_ktot_vol = []
            ens_enstot = []
            ens_enstot_vol = []
            ens_vol_enstot_vol = []
            shap = []
            shap_vol = []
            event = []
            cdg_y = []
            y_plus_min = []
            voltot = np.sum(normdata.vol)
            shapback_list = []
            shapbackvol_list = []
            expmax = 2
            expmin = -1
            ngrid = 50
            AR1_vec = np.linspace(expmin,expmax,ngrid+1)
            AR2_vec = np.linspace(expmin,expmax,ngrid+1)
            AR1_grid = np.zeros((ngrid,ngrid))
            AR2_grid = np.zeros((ngrid,ngrid))
            SHAP_grid = np.zeros((ngrid,ngrid))
            SHAP_gridvol = np.zeros((ngrid,ngrid))
            npoin = np.zeros((ngrid,ngrid))
            
            if find_files:
                indbar = file_shap.rfind('/')
                filesexist = os.listdir(file_shap[:indbar])
                text1 = file_shap[indbar+1:]+'.'
                text2 = '.h5.shap'
                range_shap =[int(x_file.replace(text1,'').replace(text2,'')) for x_file in filesexist]
            else:
                range_shap = range(start,end,step)
            lenrange_shap = len(range_shap)
        
            uv_Qminus = 0
            vol_Qminus = 0
            uv_Qminus_wa = 0
            vol_Qminus_wa = 0
            uvtot_cum = 0
            voltot_cum = 0
            shapcum = 0
            shap_volcum = 0
            shapbcum = 0
            shap_volbcum = 0
        
            try:
                normdata.read_norm()
            except:
                normdata.calc_norm(start,end)
            try:
                normdata.UUmean 
            except:
                normdata.read_Umean()
            for ii_arlim1 in np.arange(ngrid):
                arlim1inf = 10**AR1_vec[ii_arlim1]
                arlim1sup = 10**AR1_vec[ii_arlim1+1]
                for ii_arlim2 in np.arange(ngrid):
                    arlim2inf = 10**AR2_vec[ii_arlim2]
                    arlim2sup = 10**AR2_vec[ii_arlim2+1]
                    AR1_grid[ii_arlim1,ii_arlim2] = \
                    10**((AR1_vec[ii_arlim1]+AR1_vec[ii_arlim1+1])/2)
                    AR2_grid[ii_arlim1,ii_arlim2] =\
                    10**((AR2_vec[ii_arlim2]+AR2_vec[ii_arlim2+1])/2)
            for ii in range_shap:
                try:
                    uu,vv,ww = normdata.read_velocity(ii)
                    phi = normdata.read_enstrophy(ii)
                    uvtot = np.sum(abs(np.multiply(uu,vv)))
                    vvtot = np.sum(abs(np.multiply(vv,vv)))
                    uutot = np.sum(abs(np.multiply(uu,uu)))
                    ktot = 0.5*np.sum(uu**2+vv**2+ww**2)
                    enstot = np.sum(phi) 
                    uv_struc = normdata.read_uvstruc(ii,
                                                     cwd=dir_struc,
                                                     structure=structure)
                    # if editq3:
                    #     for jjind in np.arange(len(uv_struc.event)):
                    #         if uv_struc.cdg_y[jjind]>0.9 and uv_struc.event[jjind]==3:
                    #             uv_struc.event[jjind] = 2
                    voltot = np.sum(normdata.vol)
                    lenstruc = len(uv_struc.vol)
                    uvtot_cum += uvtot
                    voltot_cum += voltot
                    if any(uv_struc.vol>5e6):
                        print(ii)
                    lenstruc = len(uv_struc.vol)
                    if absolute:
                        shapvalues = abs(self.read_shap(ii,file=file_shap))
                        shapback = abs(shapvalues[-1])
                        shapmax = np.max(abs(np.array([shapmin,shapmax])))
                        shapmin = 0
                        shapmaxvol = np.max(abs(np.array([shapminvol,shapmaxvol])))
                        shapminvol = 0
                    else:
                        shapvalues = self.read_shap(ii,file=file_shap)
                        shapback = shapvalues[-1]
                        shapmax = np.max(np.array([shapmin,shapmax]))
                        shapmin = np.min(np.array([shapmin,shapmax]))
                        shapmaxvol = np.max(np.array([shapminvol,shapmaxvol]))
                        shapminvol = np.min(np.array([shapminvol,shapmaxvol]))
                    uv = np.zeros((lenstruc,))
                    v2 = np.zeros((lenstruc,))
                    u2 = np.zeros((lenstruc,))
                    k = np.zeros((lenstruc,))
                    ens = np.zeros((lenstruc,))
                    uv_vol = np.zeros((lenstruc,))
                    vv_vol = np.zeros((lenstruc,))
                    uu_vol = np.zeros((lenstruc,))
                    k_vol = np.zeros((lenstruc,))
                    ens_vol = np.zeros((lenstruc,))
                    
                    for jj in np.arange(lenstruc):
                        indexuv = np.where(uv_struc.mat_segment==jj+1)
                        for kk in np.arange(len(indexuv[0])):
                            uv[jj] += abs(uu[indexuv[0][kk],indexuv[1][kk],\
                                          indexuv[2][kk]]*vv[indexuv[0][kk],\
                                                 indexuv[1][kk],indexuv[2][kk]]) 
                            v2[jj] += abs(vv[indexuv[0][kk],indexuv[1][kk],\
                                          indexuv[2][kk]]*vv[indexuv[0][kk],\
                                                 indexuv[1][kk],indexuv[2][kk]])  
                            u2[jj] += abs(uu[indexuv[0][kk],indexuv[1][kk],\
                                          indexuv[2][kk]]*uu[indexuv[0][kk],\
                                                 indexuv[1][kk],indexuv[2][kk]])                       
                            k[jj] += 0.5*(uu[indexuv[0][kk],
                                             indexuv[1][kk],
                                             indexuv[2][kk]]**2\
                                          +vv[indexuv[0][kk],
                                              indexuv[1][kk],
                                              indexuv[2][kk]]**2\
                                          +ww[indexuv[0][kk],
                                              indexuv[1][kk],
                                              indexuv[2][kk]]**2)
                            ens[jj] += phi[indexuv[0][kk],
                                           indexuv[1][kk],
                                           indexuv[2][kk]]
                        uv_vol[jj] = uv[jj]/uv_struc.vol[jj]
                        vv_vol[jj] = v2[jj]/uv_struc.vol[jj]
                        uu_vol[jj] = u2[jj]/uv_struc.vol[jj]
                        k_vol[jj] = k[jj]/uv_struc.vol[jj]
                        ens_vol[jj] = ens[jj]/uv_struc.vol[jj]
                        uv_back_vol = (uvtot-np.sum(uv))/\
                        (voltot-np.sum(uv_struc.vol))
                        uv_vol_sum = np.sum(uv_vol)+uv_back_vol
                        vv_back_vol = (vvtot-np.sum(v2))/\
                        (voltot-np.sum(uv_struc.vol))
                        vv_vol_sum = np.sum(vv_vol)+vv_back_vol
                        uu_back_vol = (uutot-np.sum(u2))/\
                        (voltot-np.sum(uv_struc.vol))
                        uu_vol_sum = np.sum(uu_vol)+uu_back_vol
                        k_back_vol = (ktot-np.sum(k))/\
                        (voltot-np.sum(uv_struc.vol))
                        k_vol_sum = np.sum(k_vol)+k_back_vol
                        ens_back_vol = (enstot-np.sum(ens))/\
                        (voltot-np.sum(uv_struc.vol))
                        ens_vol_sum = np.sum(ens_vol)+ens_back_vol
                        if uv_struc.cdg_y[jj] <= 0:
                            yplus_min_ii = (1+uv_struc.ymin[jj])*normdata.rey
                        else:
                            yplus_min_ii = (1-uv_struc.ymax[jj])*normdata.rey
                        # if uv_struc.event[jj] == 2 or uv_struc.event[jj] == 4:
                        #     uv_Qminus += uv[jj]
                        #     vol_Qminus += uv_struc.vol[jj]
                        #     if yplus_min_ii < 20:
                        #         uv_Qminus_wa += uv[jj]
                        #         vol_Qminus_wa += uv_struc.vol[jj]
                        uv[jj] /= uvtot
                        v2[jj] /= vvtot
                        u2[jj] /= uutot
                        k[jj] /= ktot
                        ens[jj] /= enstot
                        
                        if uv_struc.vol[jj] > volmin:                  
                            Dy = uv_struc.ymax[jj]-uv_struc.ymin[jj]
                            Dz = uv_struc.dz[jj]
                            Dx = uv_struc.dx[jj]
                            AR1 = Dx/Dy
                            AR2 = Dz/Dy
                            for ii_arlim1 in np.arange(ngrid):
                                arlim1inf = 10**AR1_vec[ii_arlim1]
                                arlim1sup = 10**AR1_vec[ii_arlim1+1]
                                if AR1 >= arlim1inf and AR1<arlim1sup:
                                    for ii_arlim2 in np.arange(ngrid):
                                        arlim2inf = 10**AR2_vec[ii_arlim2]
                                        arlim2sup = 10**AR2_vec[ii_arlim2+1]
                                        if AR2 >= arlim2inf and AR2<arlim2sup:
                                            SHAP_grid[ii_arlim1,ii_arlim2] +=\
                                            shapvalues[jj]
                                            SHAP_gridvol[ii_arlim1,ii_arlim2] +=\
                                            shapvalues[jj]/uv_struc.vol[jj]
                                            npoin[ii_arlim1,ii_arlim2] += 1
                                            
                            if yplus_min_ii < 20:
                                volume_wa.append(uv_struc.vol[jj]/1e6)
                                uv_uvtot_wa.append(uv[jj])
                                uv_uvtot_wa_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                vv_vvtot_wa.append(v2[jj])
                                vv_vvtot_wa_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                k_ktot_wa.append(k[jj])
                                k_ktot_wa_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                ens_enstot_wa.append(ens[jj])
                                ens_enstot_wa_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                uv_vol_uvtot_vol_wa.append(uv_vol[jj]/uv_vol_sum)
                                shap_wa.append(shapvalues[jj]*1e3)
                                shap_wa_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                # self.event_wa.append(uv_struc.event[jj])
                                cdg_y_wa.append(uv_struc.cdg_y[jj])
                            else:
                                volume_wd.append(uv_struc.vol[jj]/1e6)
                                uv_uvtot_wd.append(uv[jj])
                                uv_uvtot_wd_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                                vv_vvtot_wd.append(v2[jj])
                                vv_vvtot_wd_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                                uu_uutot_wd.append(u2[jj])
                                uu_uutot_wd_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                                k_ktot_wd.append(k[jj])
                                k_ktot_wd_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                                ens_enstot_wd.append(ens[jj])
                                ens_enstot_wd_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                                uv_vol_uvtot_vol_wd.append(uv_vol[jj]/uv_vol_sum)
                                shap_wd.append(shapvalues[jj]*1e3)
                                shap_wd_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                                # self.event_wd.append(uv_struc.event[jj])
                                cdg_y_wd.append(uv_struc.cdg_y[jj])
                            
                            volume.append(uv_struc.vol[jj]/1e6)
                            uv_uvtot.append(uv[jj])
                            uv_uvtot_vol.append(uv[jj]/uv_struc.vol[jj]*1e7)
                            uv_vol_uvtot_vol.append(uv_vol[jj]/uv_vol_sum)
                            vv_vvtot.append(v2[jj])
                            vv_vvtot_vol.append(v2[jj]/uv_struc.vol[jj]*1e7)
                            vv_vol_vvtot_vol.append(vv_vol[jj]/uv_vol_sum)
                            uu_uutot.append(u2[jj])
                            uu_uutot_vol.append(u2[jj]/uv_struc.vol[jj]*1e7)
                            uu_vol_uutot_vol.append(uu_vol[jj]/uv_vol_sum)
                            k_ktot.append(k[jj])
                            k_ktot_vol.append(k[jj]/uv_struc.vol[jj]*1e7)
                            k_vol_ktot_vol.append(k_vol[jj]/k_vol_sum)
                            ens_enstot.append(ens[jj])
                            ens_enstot_vol.append(ens[jj]/uv_struc.vol[jj]*1e7)
                            ens_vol_enstot_vol.append(ens_vol[jj]/ens_vol_sum)
                            shap.append(shapvalues[jj]*1e3)
                            shap_vol.append(shapvalues[jj]/uv_struc.vol[jj]*1e9)
                            # self.event.append(uv_struc.event[jj])
                            shapcum += shapvalues[jj]
                            shap_volcum += shapvalues[jj]/uv_struc.vol[jj]
                            cdg_y.append(uv_struc.cdg_y[jj])
                            y_plus_min.append(yplus_min_ii)
                                
                    vol_b = np.sum(normdata.vol)-np.sum(uv_struc.vol)
                    shapbcum += shapvalues[-1]
                    shap_volbcum += shapvalues[-1]/vol_b
                    shapback_list.append(shapback)
                    volback = np.sum(normdata.vol)-np.sum(uv_struc.vol)
                    shapbackvol_list.append(shapback/volback)
                except:
                    print('Missing: '+file_shap+'.'+str(ii)+'.h5.shap')
            for ii_arlim1 in np.arange(ngrid):
                for ii_arlim2 in np.arange(ngrid):
                    if npoin[ii_arlim1,ii_arlim2] == 0:
                        SHAP_grid[ii_arlim1,ii_arlim2] = np.nan
                        SHAP_gridvol[ii_arlim1,ii_arlim2] = np.nan
                    else:
                        SHAP_grid[ii_arlim1,ii_arlim2] /= npoin[ii_arlim1,ii_arlim2]
                        SHAP_gridvol[ii_arlim1,ii_arlim2] /= npoin[ii_arlim1,ii_arlim2]
                    
            # file_save = open('Qinfo.txt', "w+") 
            # file_save.write('uv_Q- '+str(uv_Qminus/uvtot_cum)+'\n')
            # file_save.write('vol_Q- '+str(vol_Qminus/voltot_cum)+'\n')
            # file_save.write('uv_Q-_wa '+str(uv_Qminus_wa/uvtot_cum)+'\n')
            # file_save.write('vol_Q-_wa '+str(vol_Qminus_wa/voltot_cum)+'\n')
            # file_save.close()
        
            self.volume_wa[structure] = volume_wa
            self.volume_wd[structure] = volume_wd
            self.shap_wa[structure] = shap_wa
            self.shap_wd[structure] = shap_wd
            self.shap_wa_vol[structure] = shap_wa_vol
            self.shap_wd_vol[structure] = shap_wd_vol
            self.event_wa[structure] = event_wa
            self.event_wd[structure] = event_wd
            self.uv_uvtot_wa[structure] = uv_uvtot_wa
            self.uv_uvtot_wd[structure] = uv_uvtot_wd
            self.uv_uvtot_wa_vol[structure] = uv_uvtot_wa_vol
            self.uv_uvtot_wd_vol[structure] = uv_uvtot_wd_vol
            self.vv_vvtot_wa[structure] = vv_vvtot_wa
            self.vv_vvtot_wd[structure] = vv_vvtot_wd
            self.vv_vvtot_wa_vol[structure] = vv_vvtot_wa_vol
            self.vv_vvtot_wd_vol[structure] = vv_vvtot_wd_vol
            self.uu_uutot_wa[structure] = uu_uutot_wa
            self.uu_uutot_wd[structure] = uu_uutot_wd
            self.uu_uutot_wa_vol[structure] = uu_uutot_wa_vol
            self.uu_uutot_wd_vol[structure] = uu_uutot_wd_vol
            self.k_ktot_wa[structure] = k_ktot_wa
            self.k_ktot_wd[structure] = k_ktot_wd
            self.k_ktot_wa_vol[structure] = k_ktot_wa_vol
            self.k_ktot_wd_vol[structure] = k_ktot_wd_vol
            self.ens_enstot_wa[structure] = ens_enstot_wa
            self.ens_enstot_wd[structure] = ens_enstot_wd
            self.ens_enstot_wa_vol[structure] = ens_enstot_wa_vol
            self.ens_enstot_wd_vol[structure] = ens_enstot_wd_vol
            self.uv_vol_uvtot_vol_wa[structure] = uv_vol_uvtot_vol_wa
            self.uv_vol_uvtot_vol_wd[structure] = uv_vol_uvtot_vol_wd
            self.volume[structure] = volume
            self.volume_wa[structure] = volume_wa
            self.shap[structure] = shap
            self.shap_vol[structure] = shap_vol
            self.uv_uvtot[structure] = uv_uvtot
            self.uv_uvtot_vol[structure] = uv_uvtot_vol
            self.vv_vvtot[structure] = vv_vvtot
            self.vv_vvtot_vol[structure] = vv_vvtot_vol
            self.uu_uutot[structure] = uu_uutot
            self.uu_uutot_vol[structure] = uu_uutot_vol
            self.uv_vol_uvtot_vol[structure] = uv_vol_uvtot_vol
            self.k_ktot[structure] = k_ktot
            self.k_ktot_vol[structure] = k_ktot_vol
            self.k_vol_ktot_vol[structure] = k_vol_ktot_vol
            self.ens_enstot[structure] = ens_enstot
            self.ens_enstot_vol[structure] = ens_enstot_vol
            self.ens_vol_enstot_vol[structure] = ens_vol_enstot_vol
            self.event[structure] = event
            self.voltot[structure] = voltot
            self.shapback_list[structure] = shapback_list
            self.shapbackvol_list[structure] = shapbackvol_list
            self.AR1_grid[structure] = AR1_grid
            self.AR2_grid[structure] = AR2_grid
            self.SHAP_grid[structure] = SHAP_grid
            self.SHAP_gridvol[structure] = SHAP_gridvol
            self.npoin[structure] = npoin
            self.shapcum[structure] = shapcum
            self.shapbcum[structure] = shapbcum
            self.shap_volcum[structure] = shap_volcum
            self.shap_volbcum[structure] = shap_volbcum
            self.shapmax[structure] = shapmax
            self.shapmin[structure] = shapmin
            self.shapmaxvol[structure] = shapmaxvol
            self.shapminvol[structure] = shapminvol
            self.cdg_y[structure] = (1-np.abs(cdg_y))*utau/ny # convert to y plus
            self.cdg_y_wa[structure] = cdg_y_wa
            self.cdg_y_wd[structure] = cdg_y_wd
            self.y_plus_min[structure] = y_plus_min
            
            hf = h5py.File(fileread, 'w')
            hf.create_dataset('volume_wa', data=volume_wa)
            hf.create_dataset('volume_wd', data=volume_wd)
            hf.create_dataset('shap_wa', data=shap_wa)
            hf.create_dataset('shap_wd', data=shap_wd)
            hf.create_dataset('shap_wa_vol', data=shap_wa_vol)
            hf.create_dataset('shap_wd_vol', data=shap_wd_vol)
            hf.create_dataset('event_wa', data=event_wa)
            hf.create_dataset('event_wd', data=event_wd)
            hf.create_dataset('uv_uvtot_wa', data=uv_uvtot_wa)
            hf.create_dataset('uv_uvtot_wd', data=uv_uvtot_wd)
            hf.create_dataset('uv_uvtot_wa_vol', data=uv_uvtot_wa_vol)
            hf.create_dataset('uv_uvtot_wd_vol', data=uv_uvtot_wd_vol)
            hf.create_dataset('vv_vvtot_wa', data=vv_vvtot_wa)
            hf.create_dataset('vv_vvtot_wd', data=vv_vvtot_wd)
            hf.create_dataset('vv_vvtot_wa_vol', data=vv_vvtot_wa_vol)
            hf.create_dataset('vv_vvtot_wd_vol', data=vv_vvtot_wd_vol)
            hf.create_dataset('uu_uutot_wa', data=uu_uutot_wa)
            hf.create_dataset('uu_uutot_wd', data=uu_uutot_wd)
            hf.create_dataset('uu_uutot_wa_vol', data=uu_uutot_wa_vol)
            hf.create_dataset('uu_uutot_wd_vol', data=uu_uutot_wd_vol)
            hf.create_dataset('k_ktot_wa', data=k_ktot_wa)
            hf.create_dataset('k_ktot_wd', data=k_ktot_wd)
            hf.create_dataset('k_ktot_wa_vol', data=k_ktot_wa_vol)
            hf.create_dataset('k_ktot_wd_vol', data=k_ktot_wd_vol)
            hf.create_dataset('ens_enstot_wa', data=ens_enstot_wa)
            hf.create_dataset('ens_enstot_wd', data=ens_enstot_wd)
            hf.create_dataset('ens_enstot_wa_vol', data=ens_enstot_wa_vol)
            hf.create_dataset('ens_enstot_wd_vol', data=ens_enstot_wd_vol)
            hf.create_dataset('uv_vol_uvtot_vol_wa', data=uv_vol_uvtot_vol_wa)
            hf.create_dataset('uv_vol_uvtot_vol_wd', data=uv_vol_uvtot_vol_wd)
            hf.create_dataset('volume', data=volume)
            hf.create_dataset('shap', data=shap)
            hf.create_dataset('shap_vol', data=shap_vol)
            hf.create_dataset('uv_uvtot', data=uv_uvtot)
            hf.create_dataset('uv_uvtot_vol', data=uv_uvtot_vol)
            hf.create_dataset('uv_vol_uvtot_vol', data=uv_vol_uvtot_vol)
            hf.create_dataset('vv_vvtot', data=vv_vvtot)
            hf.create_dataset('vv_vvtot_vol', data=vv_vvtot_vol)
            hf.create_dataset('vv_vol_vvtot_vol', data=vv_vol_vvtot_vol)
            hf.create_dataset('uu_uutot', data=uu_uutot)
            hf.create_dataset('uu_uutot_vol', data=uu_uutot_vol)
            hf.create_dataset('uu_vol_uutot_vol', data=uu_vol_uutot_vol)
            hf.create_dataset('k_ktot', data=k_ktot)
            hf.create_dataset('k_ktot_vol', data=k_ktot_vol)
            hf.create_dataset('k_vol_ktot_vol', data=k_vol_ktot_vol)
            hf.create_dataset('ens_enstot', data=ens_enstot)
            hf.create_dataset('ens_enstot_vol', data=ens_enstot_vol)
            hf.create_dataset('ens_vol_enstot_vol', data=ens_vol_enstot_vol)
            hf.create_dataset('event', data=event)
            hf.create_dataset('voltot', data=voltot)
            hf.create_dataset('shapback_list', data=shapback_list)
            hf.create_dataset('shapbackvol_list', data=shapbackvol_list)
            hf.create_dataset('AR1_grid', data=AR1_grid)
            hf.create_dataset('AR2_grid', data=AR2_grid)
            hf.create_dataset('SHAP_grid', data=SHAP_grid)
            hf.create_dataset('SHAP_gridvol', data=SHAP_gridvol)
            hf.create_dataset('npoin', data=npoin)
            hf.create_dataset('shapcum', data=shapcum)
            hf.create_dataset('shapbcum', data=shapbcum)
            hf.create_dataset('shap_volcum', data=shap_volcum)
            hf.create_dataset('shap_volbcum', data=shap_volbcum)
            hf.create_dataset('shapmax', data=shapmax)
            hf.create_dataset('shapmin', data=shapmin)
            hf.create_dataset('shapmaxvol', data=shapmaxvol)
            hf.create_dataset('shapminvol', data=shapminvol)
            hf.create_dataset('cdg_y', data=cdg_y)
            hf.create_dataset('cdg_y_wa', data=cdg_y_wa)
            hf.create_dataset('cdg_y_wd', data=cdg_y_wd)
            hf.create_dataset('y_plus_min', data=y_plus_min)
            hf.close()
            
        
    def plot_shaps_pdf(self,
                       colormap='viridis',
                       bin_num=100,
                       lev_val=2.5,
                       alf=0.5,
                       structures=[],
                       mode='mse'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        from matplotlib.colors import ListedColormap
        xhistmin = np.min([np.min(self.volume_1),
                           np.min(self.volume_2),
                           np.min(self.volume_3),
                           np.min(self.volume_4)]\
                          +[np.min(self.volume[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.volume_1),
                           np.max(self.volume_2),
                           np.max(self.volume_3),
                           np.max(self.volume_4)]\
                          +[np.max(self.volume[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        
        histogram1,vol_value1,shap_value1 = np.histogram2d(self.volume_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2,vol_value2,shap_value2 = np.histogram2d(self.volume_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3,vol_value3,shap_value3 = np.histogram2d(self.volume_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4,vol_value4,shap_value4 = np.histogram2d(self.volume_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1 = vol_value1[:-1]+np.diff(vol_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        vol_value2 = vol_value2[:-1]+np.diff(vol_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        vol_value3 = vol_value3[:-1]+np.diff(vol_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        vol_value4 = vol_value4[:-1]+np.diff(vol_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        vol_values = {}
        shap_values = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume[structure],
                                                            self.shap[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            vol_values[structure] = vol_value
            shap_values[structure] = shap_value 
            
        
        min_vol = np.min([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        max_vol = np.max([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        interp_h1 = interp2d(vol_value1,shap_value1,histogram1)
        interp_h2 = interp2d(vol_value2,shap_value2,histogram2)
        interp_h3 = interp2d(vol_value3,shap_value3,histogram3)
        interp_h4 = interp2d(vol_value4,shap_value4,histogram4)
        vec_vol = np.linspace(min_vol,max_vol,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        vol_grid,shap_grid = np.meshgrid(vec_vol,vec_shap)
        histogram_Q1 = interp_h1(vec_vol,vec_shap)
        histogram_Q2 = interp_h2(vec_vol,vec_shap)
        histogram_Q3 = interp_h3(vec_vol,vec_shap)
        histogram_Q4 = interp_h4(vec_vol,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
    
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(vol_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_vol,vec_shap)
        
        fs = 20
        plt.figure()
        cud_colors = np.array(3*['#440154', '#009E73', '#F0E442', '#0072B2', '#E69F00']).reshape((3,5)).transpose()
        if colormap == 'custom':
            cmap = ListedColormap(cud_colors)
        else:
            cmap = colormap
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,2]
        print(type(color21), type(color22), type(color23))
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
            
        # plt.contour(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^6$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        #handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   #mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))    
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        if mode == 'mse':
            plt.tight_layout()
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.tight_layout() #rect=(0.02,0,1,1)
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,7])
        plt.savefig('hist2d_interp_vol_SHAP_'+colormap+str(structures)+'_30+.png')
        
        
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol,vol_value1_vol,shap_value1_vol = np.histogram2d(self.volume_1,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol,vol_value2_vol,shap_value2_vol = np.histogram2d(self.volume_2,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol,vol_value3_vol,shap_value3_vol = np.histogram2d(self.volume_3,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol,vol_value4_vol,shap_value4_vol = np.histogram2d(self.volume_4,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        
        histograms_vol = {}
        vol_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume[structure],
                                                            self.shap_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            vol_values_vol[structure] = vol_value
            shap_values_vol[structure] = shap_value 
        
        vol_value1_vol = vol_value1_vol[:-1]+np.diff(vol_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        vol_value2_vol = vol_value2_vol[:-1]+np.diff(vol_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        vol_value3_vol = vol_value3_vol[:-1]+np.diff(vol_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        vol_value4_vol = vol_value4_vol[:-1]+np.diff(vol_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        min_vol_vol = np.min([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        max_vol_vol = np.max([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_vol_vol = np.min([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # max_vol_vol = np.max([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(vol_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(vol_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(vol_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(vol_value4_vol,shap_value4_vol,histogram4_vol)
        vec_vol_vol = np.linspace(min_vol_vol,max_vol_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        vol_grid_vol,shap_grid_vol = np.meshgrid(vec_vol_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_vol_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(vol_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_vol_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(cmap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(cmap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^{6}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(cmap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
        elif mode == 'cf':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3])
        plt.savefig('hist2d_interp_vol_SHAPvol_'+colormap+str(structures)+'_30+.png')
        
        
        xhistmin = np.min([np.min(self.y_plus_min_1),
                           np.min(self.y_plus_min_2),
                           np.min(self.y_plus_min_3),
                           np.min(self.y_plus_min_4)]\
                          +[np.min(self.y_plus_min[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.y_plus_min_1),
                           np.max(self.y_plus_min_2),
                           np.max(self.y_plus_min_3),
                           np.max(self.y_plus_min_4)]\
                          +[np.max(self.y_plus_min[struc]) for struc in structures]
                          )/1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        
        histogram1,vol_value1,shap_value1 = np.histogram2d(self.y_plus_min_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2,vol_value2,shap_value2 = np.histogram2d(self.y_plus_min_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3,vol_value3,shap_value3 = np.histogram2d(self.y_plus_min_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4,vol_value4,shap_value4 = np.histogram2d(self.y_plus_min_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1 = vol_value1[:-1]+np.diff(vol_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        vol_value2 = vol_value2[:-1]+np.diff(vol_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        vol_value3 = vol_value3[:-1]+np.diff(vol_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        vol_value4 = vol_value4[:-1]+np.diff(vol_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        vol_values = {}
        shap_values = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.y_plus_min[structure],
                                                            self.shap[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            vol_values[structure] = vol_value
            shap_values[structure] = shap_value 
            
        
        min_vol = np.min([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        max_vol = np.max([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        interp_h1 = interp2d(vol_value1,shap_value1,histogram1)
        interp_h2 = interp2d(vol_value2,shap_value2,histogram2)
        interp_h3 = interp2d(vol_value3,shap_value3,histogram3)
        interp_h4 = interp2d(vol_value4,shap_value4,histogram4)
        vec_vol = np.linspace(min_vol,max_vol,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        vol_grid,shap_grid = np.meshgrid(vec_vol,vec_shap)
        histogram_Q1 = interp_h1(vec_vol,vec_shap)
        histogram_Q2 = interp_h2(vec_vol,vec_shap)
        histogram_Q3 = interp_h3(vec_vol,vec_shap)
        histogram_Q4 = interp_h4(vec_vol,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
    
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(vol_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_vol,vec_shap)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
            
        # plt.contour(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$y^+_{min}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        #handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   #mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        if mode == 'mse':
            plt.tight_layout()
            plt.xticks([25,50,75,100])
            plt.xlim([0,95])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.tight_layout() #rect=(0.02,0,1,1)
            plt.xticks([25,50,75,100])
            plt.xlim([0,95])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5])
        plt.savefig('hist2d_interp_ymin_SHAP_'+colormap+str(structures)+'_30+.png')
        
        
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol,vol_value1_vol,shap_value1_vol = np.histogram2d(self.y_plus_min_1,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol,vol_value2_vol,shap_value2_vol = np.histogram2d(self.y_plus_min_2,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol,vol_value3_vol,shap_value3_vol = np.histogram2d(self.y_plus_min_3,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol,vol_value4_vol,shap_value4_vol = np.histogram2d(self.y_plus_min_4,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        
        histograms_vol = {}
        vol_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.y_plus_min[structure],
                                                            self.shap_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            vol_values_vol[structure] = vol_value
            shap_values_vol[structure] = shap_value 
        
        vol_value1_vol = vol_value1_vol[:-1]+np.diff(vol_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        vol_value2_vol = vol_value2_vol[:-1]+np.diff(vol_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        vol_value3_vol = vol_value3_vol[:-1]+np.diff(vol_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        vol_value4_vol = vol_value4_vol[:-1]+np.diff(vol_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        min_vol_vol = np.min([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        max_vol_vol = np.max([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_vol_vol = np.min([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # max_vol_vol = np.max([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(vol_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(vol_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(vol_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(vol_value4_vol,shap_value4_vol,histogram4_vol)
        vec_vol_vol = np.linspace(min_vol_vol,max_vol_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        vol_grid_vol,shap_grid_vol = np.meshgrid(vec_vol_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_vol_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(vol_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_vol_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$y^+_{min}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([25,50,75,100])
            plt.xlim([0,95])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
        elif mode == 'cf':
            plt.xticks([25,50,75,100])
            plt.xlim([0,95])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,2])
        plt.savefig('hist2d_interp_ymin_SHAPvol_'+colormap+str(structures)+'_30+.png')
        
        
        xhistmin = np.min([np.min(self.cdg_y_1),
                           np.min(self.cdg_y_2),
                           np.min(self.cdg_y_3),
                           np.min(self.cdg_y_4)]\
                          +[np.min(self.cdg_y[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.cdg_y_1),
                           np.max(self.cdg_y_2),
                           np.max(self.cdg_y_3),
                           np.max(self.cdg_y_4)]\
                          +[np.max(self.cdg_y[struc]) for struc in structures]
                          )/1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        
        histogram1,vol_value1,shap_value1 = np.histogram2d(self.cdg_y_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2,vol_value2,shap_value2 = np.histogram2d(self.cdg_y_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3,vol_value3,shap_value3 = np.histogram2d(self.cdg_y_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4,vol_value4,shap_value4 = np.histogram2d(self.cdg_y_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1 = vol_value1[:-1]+np.diff(vol_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        vol_value2 = vol_value2[:-1]+np.diff(vol_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        vol_value3 = vol_value3[:-1]+np.diff(vol_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        vol_value4 = vol_value4[:-1]+np.diff(vol_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        vol_values = {}
        shap_values = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.cdg_y[structure],
                                                            self.shap[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            vol_values[structure] = vol_value
            shap_values[structure] = shap_value 
            
        
        min_vol = np.min([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        max_vol = np.max([vol_value1,
                          vol_value2,
                          vol_value3,
                          vol_value4]\
                         +[vol_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        interp_h1 = interp2d(vol_value1,shap_value1,histogram1)
        interp_h2 = interp2d(vol_value2,shap_value2,histogram2)
        interp_h3 = interp2d(vol_value3,shap_value3,histogram3)
        interp_h4 = interp2d(vol_value4,shap_value4,histogram4)
        vec_vol = np.linspace(min_vol,max_vol,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        vol_grid,shap_grid = np.meshgrid(vec_vol,vec_shap)
        histogram_Q1 = interp_h1(vec_vol,vec_shap)
        histogram_Q2 = interp_h2(vec_vol,vec_shap)
        histogram_Q3 = interp_h3(vec_vol,vec_shap)
        histogram_Q4 = interp_h4(vec_vol,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
    
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(vol_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_vol,vec_shap)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
            
        # plt.contour(vol_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$y^+_{mean}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        #handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   #mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        if mode == 'mse':
            plt.tight_layout()
            plt.xticks([25,50,75,100])
            plt.xlim([0,107])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
        elif mode == 'cf':
            plt.tight_layout() # rect=(0.02,0,1,1)
            plt.xticks([25,50,75,100])
            plt.xlim([0,107])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3])
        plt.savefig('hist2d_interp_ymean_SHAP_'+colormap+str(structures)+'_30+.png')
        
        
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol,vol_value1_vol,shap_value1_vol = np.histogram2d(self.cdg_y_1,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol,vol_value2_vol,shap_value2_vol = np.histogram2d(self.cdg_y_2,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol,vol_value3_vol,shap_value3_vol = np.histogram2d(self.cdg_y_3,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol,vol_value4_vol,shap_value4_vol = np.histogram2d(self.cdg_y_4,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        
        histograms_vol = {}
        vol_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.cdg_y[structure],
                                                            self.shap_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            vol_values_vol[structure] = vol_value
            shap_values_vol[structure] = shap_value 
        
        vol_value1_vol = vol_value1_vol[:-1]+np.diff(vol_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        vol_value2_vol = vol_value2_vol[:-1]+np.diff(vol_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        vol_value3_vol = vol_value3_vol[:-1]+np.diff(vol_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        vol_value4_vol = vol_value4_vol[:-1]+np.diff(vol_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        min_vol_vol = np.min([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        max_vol_vol = np.max([vol_value1_vol,
                          vol_value2_vol,
                          vol_value3_vol,
                          vol_value4_vol]\
                         +[vol_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_vol_vol = np.min([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # max_vol_vol = np.max([vol_value1_vol,vol_value2_vol,vol_value3_vol,vol_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(vol_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(vol_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(vol_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(vol_value4_vol,shap_value4_vol,histogram4_vol)
        vec_vol_vol = np.linspace(min_vol_vol,max_vol_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        vol_grid_vol,shap_grid_vol = np.meshgrid(vec_vol_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_vol_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_vol_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(vol_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_vol_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$y^+_{mean}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([25,50,75,100])
            plt.xlim([0,107])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5])
        elif mode == 'cf':
            plt.xticks([25,50,75,100])
            plt.xlim([0,107])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,2])
        plt.savefig('hist2d_interp_ymean_SHAPvol_'+colormap+str(structures)+'_30+.png')
        
  
    def plot_shaps_uv_pdf(self,
                          colormap='viridis',
                          bin_num=100,
                          lev_val=2.5,
                          alf=0.5,
                          structures=[],
                          mode='mse', 
                          switch='uv'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        
        if switch == 'uv':
            xhistmin = np.min([np.min(self.uv_uvtot_1),
                               np.min(self.uv_uvtot_2),
                               np.min(self.uv_uvtot_3),
                               np.min(self.uv_uvtot_4)]\
                              +[np.min(self.uv_uvtot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.uv_uvtot_1),
                               np.max(self.uv_uvtot_2),
                               np.max(self.uv_uvtot_3),
                               np.max(self.uv_uvtot_4)]\
                              +[np.max(self.uv_uvtot[struc]) for struc in structures]
                              )*1.2
                
        elif switch == 'vv':
            xhistmin = np.min([np.min(self.vv_vvtot_1),
                               np.min(self.vv_vvtot_2),
                               np.min(self.vv_vvtot_3),
                               np.min(self.vv_vvtot_4)]\
                              +[np.min(self.vv_vvtot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.vv_vvtot_1),
                               np.max(self.vv_vvtot_2),
                               np.max(self.vv_vvtot_3),
                               np.max(self.vv_vvtot_4)]\
                              +[np.max(self.vv_vvtot[struc]) for struc in structures]
                              )*1.2
                
        elif switch == 'uu':
            xhistmin = np.min([np.min(self.uu_uutot_1),
                               np.min(self.uu_uutot_2),
                               np.min(self.uu_uutot_3),
                               np.min(self.uu_uutot_4)]\
                              +[np.min(self.uu_uutot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.uu_uutot_1),
                               np.max(self.uu_uutot_2),
                               np.max(self.uu_uutot_3),
                               np.max(self.uu_uutot_4)]\
                              +[np.max(self.uu_uutot[struc]) for struc in structures]
                              )*1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        # xhistmin = np.min([np.min(self.uv_uvtot_1),np.min(self.uv_uvtot_2),np.min(self.uv_uvtot_3),np.min(self.uv_uvtot_4)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1),np.max(self.uv_uvtot_2),np.max(self.uv_uvtot_3),np.max(self.uv_uvtot_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        if switch == 'uv':
            histogram1,uv_value1,shap_value1 = np.histogram2d(self.uv_uvtot_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2,uv_value2,shap_value2 = np.histogram2d(self.uv_uvtot_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3,uv_value3,shap_value3 = np.histogram2d(self.uv_uvtot_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4,uv_value4,shap_value4 = np.histogram2d(self.uv_uvtot_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        
        elif switch == 'vv':
            histogram1,uv_value1,shap_value1 = np.histogram2d(self.vv_vvtot_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2,uv_value2,shap_value2 = np.histogram2d(self.vv_vvtot_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3,uv_value3,shap_value3 = np.histogram2d(self.vv_vvtot_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4,uv_value4,shap_value4 = np.histogram2d(self.vv_vvtot_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            
        elif switch == 'uu':
            histogram1,uv_value1,shap_value1 = np.histogram2d(self.uu_uutot_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2,uv_value2,shap_value2 = np.histogram2d(self.uu_uutot_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3,uv_value3,shap_value3 = np.histogram2d(self.uu_uutot_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4,uv_value4,shap_value4 = np.histogram2d(self.uu_uutot_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        
        uv_value1 = uv_value1[:-1]+np.diff(uv_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        uv_value2 = uv_value2[:-1]+np.diff(uv_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        uv_value3 = uv_value3[:-1]+np.diff(uv_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        uv_value4 = uv_value4[:-1]+np.diff(uv_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        uv_values = {}
        shap_values = {}
        for structure in structures:
            if switch == 'uv':
                histogram,uv_value,shap_value = np.histogram2d(self.uv_uvtot[structure],
                                                                self.shap[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            elif switch == 'vv':
                histogram,uv_value,shap_value = np.histogram2d(self.vv_vvtot[structure],
                                                                self.shap[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            elif switch == 'uu':
                histogram,uv_value,shap_value = np.histogram2d(self.uu_uutot[structure],
                                                                self.shap[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            uv_values[structure] = uv_value
            shap_values[structure] = shap_value 
            
        min_uv = np.min([uv_value1,
                         uv_value2,
                         #uv_value3,
                         uv_value4]\
                        +[uv_values[struc] for struc in structures])
            
        max_uv = np.max([uv_value1,
                         uv_value2,
                         #uv_value3,
                         uv_value4]\
                        +[uv_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           # shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           # shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        # min_uv = np.min([uv_value1,uv_value2,uv_value3,uv_value4])
        # max_uv = np.max([uv_value1,uv_value2,uv_value3,uv_value4])
        # min_shap = np.min([shap_value1,shap_value2,shap_value3,shap_value4])
        # max_shap = np.max([shap_value1,shap_value2,shap_value3,shap_value4])
        interp_h1 = interp2d(uv_value1,shap_value1,histogram1)
        interp_h2 = interp2d(uv_value2,shap_value2,histogram2)
        interp_h3 = interp2d(uv_value3,shap_value3,histogram3)
        interp_h4 = interp2d(uv_value4,shap_value4,histogram4)
        vec_uv = np.linspace(min_uv,max_uv,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        uv_grid,shap_grid = np.meshgrid(vec_uv,vec_shap)
        histogram_Q1 = interp_h1(vec_uv,vec_shap)
        histogram_Q2 = interp_h2(vec_uv,vec_shap)
        histogram_Q3 = interp_h3(vec_uv,vec_shap)
        histogram_Q4 = interp_h4(vec_uv,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
        
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(uv_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_uv,vec_shap)
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        if mode == 'mse':
            plt.xticks([0.05, 0.10, 0.15])
            plt.xlim([0,0.2])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.xticks([0.05, 0.10, 0.15])
            plt.xlim([0,0.2])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3.5])
        if switch == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot})$',\
                       fontsize=fs)
        elif switch == 'vv':
            plt.xlabel('$\overline{v²}_e/(\overline{v²}_\mathrm{tot})$',\
                       fontsize=fs)
        elif switch == 'uu':
            plt.xlabel('$\overline{u²}_e/(\overline{u²}_\mathrm{tot})$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles =   [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if switch == 'uv':
            plt.savefig('hist2d_interp_uvuvtot_SHAP_'+colormap+str(structures)+'_30+.png')
        elif switch == 'vv':
            plt.savefig('hist2d_interp_vvvvtot_SHAP_'+colormap+str(structures)+'_30+.png')
        elif switch == 'uu':
            plt.savefig('hist2d_interp_uuuutot_SHAP_'+colormap+str(structures)+'_30+.png')
        
        if switch == 'uv':
            xhistmin = np.min([np.min(self.uv_uvtot_1_vol),
                               np.min(self.uv_uvtot_2_vol),
                               np.min(self.uv_uvtot_3_vol),
                               np.min(self.uv_uvtot_4_vol)]\
                              +[np.min(self.uv_uvtot_vol[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.uv_uvtot_1_vol),
                               np.max(self.uv_uvtot_2_vol),
                               np.max(self.uv_uvtot_3_vol),
                               np.max(self.uv_uvtot_4_vol)]\
                              +[np.max(self.uv_uvtot_vol[struc]) for struc in structures]
                              )*1.2
        elif switch == 'vv':
            xhistmin = np.min([np.min(self.vv_vvtot_1_vol),
                               np.min(self.vv_vvtot_2_vol),
                               np.min(self.vv_vvtot_3_vol),
                               np.min(self.vv_vvtot_4_vol)]\
                              +[np.min(self.vv_vvtot_vol[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.vv_vvtot_1_vol),
                               np.max(self.vv_vvtot_2_vol),
                               np.max(self.vv_vvtot_3_vol),
                               np.max(self.vv_vvtot_4_vol)]\
                              +[np.max(self.vv_vvtot_vol[struc]) for struc in structures]
                              )*1.2
        
        elif switch == 'uu':
            xhistmin = np.min([np.min(self.uu_uutot_1_vol),
                               np.min(self.uu_uutot_2_vol),
                               np.min(self.uu_uutot_3_vol),
                               np.min(self.uu_uutot_4_vol)]\
                              +[np.min(self.uu_uutot_vol[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.uu_uutot_1_vol),
                               np.max(self.uu_uutot_2_vol),
                               np.max(self.uu_uutot_3_vol),
                               np.max(self.uu_uutot_4_vol)]\
                              +[np.max(self.uu_uutot_vol[struc]) for struc in structures]
                              )*1.2
            
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.uv_uvtot_1_vol),np.min(self.uv_uvtot_2_vol),np.min(self.uv_uvtot_3_vol),np.min(self.uv_uvtot_4_vol)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1_vol),np.max(self.uv_uvtot_2_vol),np.max(self.uv_uvtot_3_vol),np.max(self.uv_uvtot_4_vol)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        if switch == 'uv':
            histogram1_vol,uv_value1_vol,shap_value1_vol = np.histogram2d(self.uv_uvtot_1_vol,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_vol,uv_value2_vol,shap_value2_vol = np.histogram2d(self.uv_uvtot_2_vol,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_vol,uv_value3_vol,shap_value3_vol = np.histogram2d(self.uv_uvtot_3_vol,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_vol,uv_value4_vol,shap_value4_vol = np.histogram2d(self.uv_uvtot_4_vol,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        elif switch == 'vv':
            histogram1_vol,uv_value1_vol,shap_value1_vol = np.histogram2d(self.vv_vvtot_1_vol,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_vol,uv_value2_vol,shap_value2_vol = np.histogram2d(self.vv_vvtot_2_vol,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_vol,uv_value3_vol,shap_value3_vol = np.histogram2d(self.vv_vvtot_3_vol,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_vol,uv_value4_vol,shap_value4_vol = np.histogram2d(self.vv_vvtot_4_vol,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        elif switch == 'uu':
            histogram1_vol,uv_value1_vol,shap_value1_vol = np.histogram2d(self.uu_uutot_1_vol,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_vol,uv_value2_vol,shap_value2_vol = np.histogram2d(self.uu_uutot_2_vol,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_vol,uv_value3_vol,shap_value3_vol = np.histogram2d(self.uu_uutot_3_vol,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_vol,uv_value4_vol,shap_value4_vol = np.histogram2d(self.uu_uutot_4_vol,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        uv_value1_vol = uv_value1_vol[:-1]+np.diff(uv_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        uv_value2_vol = uv_value2_vol[:-1]+np.diff(uv_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        uv_value3_vol = uv_value3_vol[:-1]+np.diff(uv_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        uv_value4_vol = uv_value4_vol[:-1]+np.diff(uv_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        histograms_vol = {}
        uv_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            if switch == 'uv':
                histogram,uv_value,shap_value = np.histogram2d(self.uv_uvtot_vol[structure],
                                                                self.shap_vol[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            elif switch == 'vv':
                histogram,uv_value,shap_value = np.histogram2d(self.vv_vvtot_vol[structure],
                                                                self.shap_vol[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            elif switch == 'uu':
                histogram,uv_value,shap_value = np.histogram2d(self.uu_uutot_vol[structure],
                                                                self.shap_vol[structure],
                                                                bins=bin_num,
                                                                range=[[xhistmin,xhistmax],
                                                                       [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            uv_values_vol[structure] = uv_value
            shap_values_vol[structure] = shap_value 
            
        min_uv_vol = np.min([uv_value1_vol,
                         uv_value2_vol,
                         uv_value3_vol,
                         uv_value4_vol]\
                        +[uv_values_vol[struc] for struc in structures])
            
        max_uv_vol = np.max([uv_value1_vol,
                         uv_value2_vol,
                         uv_value3_vol,
                         uv_value4_vol]\
                        +[uv_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_uv_vol = np.min([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # max_uv_vol = np.max([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(uv_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(uv_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(uv_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(uv_value4_vol,shap_value4_vol,histogram4_vol)
        vec_uv_vol = np.linspace(min_uv_vol,max_uv_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        uv_grid_vol,shap_grid_vol = np.meshgrid(vec_uv_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_uv_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_uv_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_uv_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_uv_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(uv_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_uv_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        x0 = 0
        x0b = 0.7
        x1 = 0.08
        x2 = 1.16
        x3 = 1.7
        y0_1 = 2
        y1_1 = 3.9
        y0_2 = 0
        y1_2 = 0.8
        ytop = 4.5
        ytop2 = y1_1
        
        # if mode == 'mse':
        #     plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
        #                       color=cmap_fill(0.9),alpha=0.1)
        #     plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
        #                       [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        #     plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
        #                       color=cmap_fill(0.1),alpha=0.1)
        #     plt.plot([x0,x0],[y0_1,y0_2],color='k')
        #     plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        #     plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        #     plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        #     plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        #     plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        #     plt.plot([x2,x3],[ytop2,ytop2],color='k')
        #     plt.plot([x2,x2],[y1_2,y1_1],color='k')
        #     plt.plot([x2,x2],[ytop,y1_1],color='k')
        #     plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        
        
        if mode == 'mse':
            # plt.text(0.5, 0.1, 'A', fontsize = 20)
            # plt.text(1.5, 2.1, 'B', fontsize = 20)   
            # plt.text(0.5, 4, 'C', fontsize = 20) 
            # plt.ylim([y0_2,ytop])
            # plt.xlim([x0,x3])
            plt.xticks([0.5,1,1.5])
            plt.xlim([0,1.7])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,4.5])
        elif mode == 'cf':
            if switch == 'uv':
                plt.xticks([0.5,1])
                plt.xlim([0,1.5])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            elif switch == 'vv':
                plt.xticks([0.5,1,1.5])
                plt.xlim([0,2])
                plt.yticks([0.5,1,1.5])
                plt.ylim([0,2])
            elif switch == 'uu':
                plt.xticks([0.5,1,1.5])
                plt.xlim([0,2])
                plt.yticks([0.5,1,1.5])
                plt.ylim([0,2])
            
        plt.grid()
        if switch == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot}V^+)\cdot10^{-7}$',\
                   fontsize=fs)
        elif switch == 'vv':
            plt.xlabel('$\overline{v²}_e/(\overline{v²}_\mathrm{tot}V^+)\cdot10^{-7}$',\
                   fontsize=fs)
        elif switch == 'uu':
            plt.xlabel('$\overline{u²}_e/(\overline{u²}_\mathrm{tot}V^+)\cdot10^{-7}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if switch == 'uv':
            plt.savefig('hist2d_interp_uvuvtotvol_SHAPvol_'+colormap+str(structures)+'_30+.png')
        elif switch == 'vv':
            plt.savefig('hist2d_interp_vvvvtotvol_SHAPvol_'+colormap+str(structures)+'_30+.png')
        elif switch == 'uu':
            plt.savefig('hist2d_interp_uuuutotvol_SHAPvol_'+colormap+str(structures)+'_30+.png')
        
        
    def plot_shaps_k_pdf(self,
                          colormap='viridis',
                          bin_num=100,
                          lev_val=2.5,
                          alf=0.5,
                          structures=[],
                          mode='mse'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        
        xhistmin = np.min([np.min(self.k_ktot_1),
                           np.min(self.k_ktot_2),
                           np.min(self.k_ktot_3),
                           np.min(self.k_ktot_4)]\
                          +[np.min(self.k_ktot[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.k_ktot_1),
                           np.max(self.k_ktot_2),
                           np.max(self.k_ktot_3),
                           np.max(self.k_ktot_4)]\
                          +[np.max(self.k_ktot[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        # xhistmin = np.min([np.min(self.uv_uvtot_1),np.min(self.uv_uvtot_2),np.min(self.uv_uvtot_3),np.min(self.uv_uvtot_4)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1),np.max(self.uv_uvtot_2),np.max(self.uv_uvtot_3),np.max(self.uv_uvtot_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        histogram1,k_value1,shap_value1 = np.histogram2d(self.k_ktot_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2,k_value2,shap_value2 = np.histogram2d(self.k_ktot_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3,k_value3,shap_value3 = np.histogram2d(self.k_ktot_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4,k_value4,shap_value4 = np.histogram2d(self.k_ktot_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        k_value1 = k_value1[:-1]+np.diff(k_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        k_value2 = k_value2[:-1]+np.diff(k_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        k_value3 = k_value3[:-1]+np.diff(k_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        k_value4 = k_value4[:-1]+np.diff(k_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        k_values = {}
        shap_values = {}
        for structure in structures:
            histogram,k_value,shap_value = np.histogram2d(self.k_ktot[structure],
                                                            self.shap[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            k_value = k_value[:-1]+np.diff(k_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            k_values[structure] = k_value
            shap_values[structure] = shap_value 
            
        min_k = np.min([k_value1,
                         k_value2,
                         k_value3,
                         k_value4]\
                        +[k_values[struc] for struc in structures])
            
        max_k = np.max([k_value1,
                         k_value2,
                         k_value3,
                         k_value4]\
                        +[k_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
            
        # min_uv = np.min([uv_value1,uv_value2,uv_value3,uv_value4])
        # max_uv = np.max([uv_value1,uv_value2,uv_value3,uv_value4])
        # min_shap = np.min([shap_value1,shap_value2,shap_value3,shap_value4])
        # max_shap = np.max([shap_value1,shap_value2,shap_value3,shap_value4])
        interp_h1 = interp2d(k_value1,shap_value1,histogram1)
        interp_h2 = interp2d(k_value2,shap_value2,histogram2)
        interp_h3 = interp2d(k_value3,shap_value3,histogram3)
        interp_h4 = interp2d(k_value4,shap_value4,histogram4)
        vec_k = np.linspace(min_k,max_k,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        k_grid,shap_grid = np.meshgrid(vec_k,vec_shap)
        histogram_Q1 = interp_h1(vec_k,vec_shap)
        histogram_Q2 = interp_h2(vec_k,vec_shap)
        histogram_Q3 = interp_h3(vec_k,vec_shap)
        histogram_Q4 = interp_h4(vec_k,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
        
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(k_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_k,vec_shap)
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(k_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(k_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(k_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(k_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(k_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
                    
        # plt.contour(k_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(k_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(k_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(k_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(k_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        
        if mode == 'mse':
            plt.xticks([0.05, 0.10, 0.15])
            plt.xlim([0,0.2])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.xticks([0.05, 0.10, 0.15])
            plt.xlim([0,0.2])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3.5])
            
        plt.xlabel('$k_e/(k_\mathrm{tot})$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_kktot_SHAP_'+colormap+str(structures)+'_30+.png')
        
        xhistmin = np.min([np.min(self.k_ktot_1_vol),
                           np.min(self.k_ktot_2_vol),
                           np.min(self.k_ktot_3_vol),
                           np.min(self.k_ktot_4_vol)]\
                          +[np.min(self.k_ktot_vol[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.k_ktot_1_vol),
                           np.max(self.k_ktot_2_vol),
                           np.max(self.k_ktot_3_vol),
                           np.max(self.k_ktot_4_vol)]\
                          +[np.max(self.k_ktot_vol[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.uv_uvtot_1_vol),np.min(self.uv_uvtot_2_vol),np.min(self.uv_uvtot_3_vol),np.min(self.uv_uvtot_4_vol)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1_vol),np.max(self.uv_uvtot_2_vol),np.max(self.uv_uvtot_3_vol),np.max(self.uv_uvtot_4_vol)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol,k_value1_vol,shap_value1_vol = np.histogram2d(self.k_ktot_1_vol,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol,k_value2_vol,shap_value2_vol = np.histogram2d(self.k_ktot_2_vol,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol,k_value3_vol,shap_value3_vol = np.histogram2d(self.k_ktot_3_vol,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol,k_value4_vol,shap_value4_vol = np.histogram2d(self.k_ktot_4_vol,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        k_value1_vol = k_value1_vol[:-1]+np.diff(k_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        k_value2_vol = k_value2_vol[:-1]+np.diff(k_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        k_value3_vol = k_value3_vol[:-1]+np.diff(k_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        k_value4_vol = k_value4_vol[:-1]+np.diff(k_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        histograms_vol = {}
        k_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            histogram,k_value,shap_value = np.histogram2d(self.k_ktot_vol[structure],
                                                            self.shap_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            k_value = k_value[:-1]+np.diff(k_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            k_values_vol[structure] = k_value
            shap_values_vol[structure] = shap_value 
            
        min_k_vol = np.min([k_value1_vol,
                         k_value2_vol,
                         k_value3_vol,
                         k_value4_vol]\
                        +[k_values_vol[struc] for struc in structures])
            
        max_k_vol = np.max([k_value1_vol,
                         k_value2_vol,
                         k_value3_vol,
                         k_value4_vol]\
                        +[k_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_uv_vol = np.min([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # max_uv_vol = np.max([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(k_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(k_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(k_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(k_value4_vol,shap_value4_vol,histogram4_vol)
        vec_k_vol = np.linspace(min_k_vol,max_k_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        k_grid_vol,shap_grid_vol = np.meshgrid(vec_k_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_k_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_k_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_k_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_k_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(k_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_k_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        x0 = 0
        x0b = 0.7
        x1 = 0.08
        x2 = 1.16
        x3 = 1.7
        y0_1 = 2
        y1_1 = 3.9
        y0_2 = 0
        y1_2 = 0.8
        ytop = 4.5
        ytop2 = y1_1
        
        '''
        plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
                         color=cmap_fill(0.9),alpha=0.1)
        plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
                         [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
                         color=cmap_fill(0.1),alpha=0.1)
        plt.plot([x0,x0],[y0_1,y0_2],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        plt.plot([x2,x3],[ytop2,ytop2],color='k')
        plt.plot([x2,x2],[y1_2,y1_1],color='k')
        plt.plot([x2,x2],[ytop,y1_1],color='k')
        plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        '''
        
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(k_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(k_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(k_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(k_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(k_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(k_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(k_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(k_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(k_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(k_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        # plt.text(0.5, 0.1, 'A', fontsize = 20)
        # plt.text(1.5, 2.1, 'B', fontsize = 20)   
        # plt.text(0.5, 4, 'C', fontsize = 20) 
        # plt.ylim([y0_2,ytop])
        # plt.xlim([x0,x3])
        
        if mode == 'mse':
            plt.xticks([0.5,1])
            plt.xlim([0,1.25])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,4.5])
        elif mode == 'cf':
            plt.xticks([0.5,1])
            plt.xlim([0,1.5])
            plt.yticks([0.5,1])
            plt.ylim([0,1.5])
        plt.grid()
        plt.xlabel('$k_e/(k_\mathrm{tot}V^+)\cdot10^{-7}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_kktotvol_SHAPvol_'+colormap+str(structures)+'_30+.png')
        
        
    def plot_shaps_ens_pdf(self,
                          colormap='viridis',
                          bin_num=100,
                          lev_val=2.5,
                          alf=0.5,
                          structures=[],
                          mode='mse'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        
        xhistmin = np.min([np.min(self.ens_enstot_1),
                           np.min(self.ens_enstot_2),
                           np.min(self.ens_enstot_3),
                           np.min(self.ens_enstot_4)]\
                          +[np.min(self.ens_enstot[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.ens_enstot_1),
                           np.max(self.ens_enstot_2),
                           np.max(self.ens_enstot_3),
                           np.max(self.ens_enstot_4)]\
                          +[np.max(self.ens_enstot[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        # xhistmin = np.min([np.min(self.uv_uvtot_1),np.min(self.uv_uvtot_2),np.min(self.uv_uvtot_3),np.min(self.uv_uvtot_4)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1),np.max(self.uv_uvtot_2),np.max(self.uv_uvtot_3),np.max(self.uv_uvtot_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        histogram1,ens_value1,shap_value1 = np.histogram2d(self.ens_enstot_1,self.shap_1,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2,ens_value2,shap_value2 = np.histogram2d(self.ens_enstot_2,self.shap_2,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3,ens_value3,shap_value3 = np.histogram2d(self.ens_enstot_3,self.shap_3,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4,ens_value4,shap_value4 = np.histogram2d(self.ens_enstot_4,self.shap_4,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        ens_value1 = ens_value1[:-1]+np.diff(ens_value1)/2
        shap_value1 = shap_value1[:-1]+np.diff(shap_value1)/2
        ens_value2 = ens_value2[:-1]+np.diff(ens_value2)/2
        shap_value2 = shap_value2[:-1]+np.diff(shap_value2)/2
        ens_value3 = ens_value3[:-1]+np.diff(ens_value3)/2
        shap_value3 = shap_value3[:-1]+np.diff(shap_value3)/2
        ens_value4 = ens_value4[:-1]+np.diff(ens_value4)/2
        shap_value4 = shap_value4[:-1]+np.diff(shap_value4)/2
        
        histograms = {}
        ens_values = {}
        shap_values = {}
        for structure in structures:
            histogram,ens_value,shap_value = np.histogram2d(self.ens_enstot[structure],
                                                            self.shap[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            ens_value = ens_value[:-1]+np.diff(ens_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms[structure] = histogram
            ens_values[structure] = ens_value
            shap_values[structure] = shap_value 
            
        min_ens = np.min([ens_value1,
                         ens_value2,
                         ens_value3,
                         ens_value4]\
                        +[ens_values[struc] for struc in structures])
            
        max_ens = np.max([ens_value1,
                         ens_value2,
                         ens_value3,
                         ens_value4]\
                        +[ens_values[struc] for struc in structures])
            
        min_shap = np.min([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
        
        max_shap = np.max([shap_value1,
                           shap_value2,
                           shap_value3,
                           shap_value4]\
                          +[shap_values[struc] for struc in structures])
            
        # min_uv = np.min([uv_value1,uv_value2,uv_value3,uv_value4])
        # max_uv = np.max([uv_value1,uv_value2,uv_value3,uv_value4])
        # min_shap = np.min([shap_value1,shap_value2,shap_value3,shap_value4])
        # max_shap = np.max([shap_value1,shap_value2,shap_value3,shap_value4])
        interp_h1 = interp2d(ens_value1,shap_value1,histogram1)
        interp_h2 = interp2d(ens_value2,shap_value2,histogram2)
        interp_h3 = interp2d(ens_value3,shap_value3,histogram3)
        interp_h4 = interp2d(ens_value4,shap_value4,histogram4)
        vec_ens = np.linspace(min_ens,max_ens,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        ens_grid,shap_grid = np.meshgrid(vec_ens,vec_shap)
        histogram_Q1 = interp_h1(vec_ens,vec_shap)
        histogram_Q2 = interp_h2(vec_ens,vec_shap)
        histogram_Q3 = interp_h3(vec_ens,vec_shap)
        histogram_Q4 = interp_h4(vec_ens,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
        
        histograms_struc = {}
        for structure in structures:
            interp_h = interp2d(ens_values[structure],
                                shap_values[structure],
                                histograms[structure])
            histograms_struc[structure] = interp_h(vec_ens,vec_shap)
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(ens_grid,shap_grid,histogram_Q1.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(ens_grid,shap_grid,histogram_Q2.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(ens_grid,shap_grid,histogram_Q3.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(ens_grid,shap_grid,histogram_Q4.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(ens_grid,
                         shap_grid,
                         histograms_struc[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
                    
        # plt.contour(ens_grid,shap_grid,histogram_Q1.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(ens_grid,shap_grid,histogram_Q2.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(ens_grid,shap_grid,histogram_Q3.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(ens_grid,shap_grid,histogram_Q4.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(ens_grid,
                        shap_grid,
                        histograms_struc[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        if mode == 'mse':
            plt.xticks([0.02, 0.04])
            plt.xlim([0,0.05])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.xticks([0.02, 0.04])
            plt.xlim([0,0.05])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,7])
        plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot})$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_ensenstot_SHAP_'+colormap+str(structures)+'_30+.png')
        
        xhistmin = np.min([np.min(self.ens_enstot_1_vol),
                           np.min(self.ens_enstot_2_vol),
                           np.min(self.ens_enstot_3_vol),
                           np.min(self.ens_enstot_4_vol)]\
                          +[np.min(self.ens_enstot_vol[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.ens_enstot_1_vol),
                           np.max(self.ens_enstot_2_vol),
                           np.max(self.ens_enstot_3_vol),
                           np.max(self.ens_enstot_4_vol)]\
                          +[np.max(self.ens_enstot_vol[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.uv_uvtot_1_vol),np.min(self.uv_uvtot_2_vol),np.min(self.uv_uvtot_3_vol),np.min(self.uv_uvtot_4_vol)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1_vol),np.max(self.uv_uvtot_2_vol),np.max(self.uv_uvtot_3_vol),np.max(self.uv_uvtot_4_vol)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol,ens_value1_vol,shap_value1_vol = np.histogram2d(self.ens_enstot_1_vol,self.shap_1_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol,ens_value2_vol,shap_value2_vol = np.histogram2d(self.ens_enstot_2_vol,self.shap_2_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol,ens_value3_vol,shap_value3_vol = np.histogram2d(self.ens_enstot_3_vol,self.shap_3_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol,ens_value4_vol,shap_value4_vol = np.histogram2d(self.ens_enstot_4_vol,self.shap_4_vol,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        ens_value1_vol = ens_value1_vol[:-1]+np.diff(ens_value1_vol)/2
        shap_value1_vol = shap_value1_vol[:-1]+np.diff(shap_value1_vol)/2
        ens_value2_vol = ens_value2_vol[:-1]+np.diff(ens_value2_vol)/2
        shap_value2_vol = shap_value2_vol[:-1]+np.diff(shap_value2_vol)/2
        ens_value3_vol = ens_value3_vol[:-1]+np.diff(ens_value3_vol)/2
        shap_value3_vol = shap_value3_vol[:-1]+np.diff(shap_value3_vol)/2
        ens_value4_vol = ens_value4_vol[:-1]+np.diff(ens_value4_vol)/2
        shap_value4_vol = shap_value4_vol[:-1]+np.diff(shap_value4_vol)/2
        
        histograms_vol = {}
        ens_values_vol = {}
        shap_values_vol = {}
        for structure in structures:
            histogram,ens_value,shap_value = np.histogram2d(self.ens_enstot_vol[structure],
                                                            self.shap_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            ens_value = ens_value[:-1]+np.diff(ens_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol[structure] = histogram
            ens_values_vol[structure] = ens_value
            shap_values_vol[structure] = shap_value 
            
        min_ens_vol = np.min([ens_value1_vol,
                         ens_value2_vol,
                         ens_value3_vol,
                         ens_value4_vol]\
                        +[ens_values_vol[struc] for struc in structures])
            
        max_ens_vol = np.max([ens_value1_vol,
                         ens_value2_vol,
                         ens_value3_vol,
                         ens_value4_vol]\
                        +[ens_values_vol[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol,
                           shap_value2_vol,
                           shap_value3_vol,
                           shap_value4_vol]\
                          +[shap_values_vol[struc] for struc in structures])
        
        # min_uv_vol = np.min([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # max_uv_vol = np.max([uv_value1_vol,uv_value2_vol,uv_value3_vol,uv_value4_vol])
        # min_shap_vol = np.min([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        # max_shap_vol = np.max([shap_value1_vol,shap_value2_vol,shap_value3_vol,shap_value4_vol])
        interp_h1_vol = interp2d(ens_value1_vol,shap_value1_vol,histogram1_vol)
        interp_h2_vol = interp2d(ens_value2_vol,shap_value2_vol,histogram2_vol)
        interp_h3_vol = interp2d(ens_value3_vol,shap_value3_vol,histogram3_vol)
        interp_h4_vol = interp2d(ens_value4_vol,shap_value4_vol,histogram4_vol)
        vec_ens_vol = np.linspace(min_ens_vol,max_ens_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        ens_grid_vol,shap_grid_vol = np.meshgrid(vec_ens_vol,vec_shap_vol)
        histogram_Q1_vol = interp_h1_vol(vec_ens_vol,vec_shap_vol)
        histogram_Q2_vol = interp_h2_vol(vec_ens_vol,vec_shap_vol)
        histogram_Q3_vol = interp_h3_vol(vec_ens_vol,vec_shap_vol)
        histogram_Q4_vol = interp_h4_vol(vec_ens_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol = {}
        for structure in structures:
            interp_h = interp2d(ens_values_vol[structure],
                                shap_values_vol[structure],
                                histograms_vol[structure])
            histograms_struc_vol[structure] = interp_h(vec_ens_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        x0 = 0
        x0b = 0.7
        x1 = 0.08
        x2 = 1.16
        x3 = 1.7
        y0_1 = 2
        y1_1 = 3.9
        y0_2 = 0
        y1_2 = 0.8
        ytop = 4.5
        ytop2 = y1_1
        '''
        plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
                         color=cmap_fill(0.9),alpha=0.1)
        plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
                         [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
                         color=cmap_fill(0.1),alpha=0.1)
        plt.plot([x0,x0],[y0_1,y0_2],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        plt.plot([x2,x3],[ytop2,ytop2],color='k')
        plt.plot([x2,x2],[y1_2,y1_1],color='k')
        plt.plot([x2,x2],[ytop,y1_1],color='k')
        plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        '''
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(ens_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(ens_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(ens_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(ens_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(ens_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(ens_grid_vol,shap_grid_vol,histogram_Q1_vol.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(ens_grid_vol,shap_grid_vol,histogram_Q2_vol.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(ens_grid_vol,shap_grid_vol,histogram_Q3_vol.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(ens_grid_vol,shap_grid_vol,histogram_Q4_vol.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(ens_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        '''
        plt.text(0.5, 0.1, 'A', fontsize = 20)
        plt.text(1.5, 2.1, 'B', fontsize = 20)   
        plt.text(0.5, 4, 'C', fontsize = 20) 
        
        
        plt.ylim([y0_2,ytop])
        plt.xlim([x0,x3])
        '''
        
        if mode == 'mse':
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,2.5])
        elif mode == 'cf':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,2.5])
            plt.yticks([0.5,1,1.5,2])
            plt.ylim([0,2.5])
        
        plt.grid()
        plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot}V^+)\cdot10^{-7}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
        #           mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'streak_high_vel':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks\n (High Velocity)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                    
            
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_ensenstotvol_SHAPvol_'+colormap+str(structures)+'_30+.png')
          
          
    def plot_shaps_total_noback(self,start=1,end=2,step=1,\
                                file='../SHAP_fields_io/PIV',\
                                fileQ='../Q_fields_io/PIV',\
                                fileuvw='../uv_fields_io/PIV',\
                                filenorm="norm.txt",\
                                colormap='viridis',absolute=False,testcases=False,\
                                filetest='ind_val.txt',numfield=-1,fieldini=0,dx=1,dy=1,\
                                volfilt=900):
        """
        Function for calculating the SHAP contribution of each kind of 
        structure
        """
        shap1 = self.shap1cum
        shap2 = self.shap2cum
        shap3 = self.shap3cum
        shap4 = self.shap4cum
        shap_vol1 = self.shap_vol1cum
        shap_vol2 = self.shap_vol2cum
        shap_vol3 = self.shap_vol3cum
        shap_vol4 = self.shap_vol4cum
        shap_sum = shap1+shap2+shap3+shap4
        shap_vol_sum = shap_vol1+shap_vol2+shap_vol3+shap_vol4
        shap1 /= shap_sum
        shap2 /= shap_sum
        shap3 /= shap_sum
        shap4 /= shap_sum
        shap_vol1 /= shap_vol_sum
        shap_vol2 /= shap_vol_sum
        shap_vol3 /= shap_vol_sum
        shap_vol4 /= shap_vol_sum
        import matplotlib.pyplot as plt
        fs = 20
        plt.figure()
        ax = plt.axes()
        plt.bar([0.5,1.5],[shap1,shap_vol1],\
                color=plt.cm.get_cmap(colormap,4).colors[0,:],\
                label='Outward\ninteraction',edgecolor='black')
        plt.bar([0.5,1.5],[shap2,shap_vol2],bottom=[shap1,shap_vol1],\
                color=plt.cm.get_cmap(colormap,4).colors[1,:],\
                label='Ejection',edgecolor='black')
        plt.bar([0.5,1.5],[shap3,shap_vol3],\
                bottom=[shap1+shap2,shap_vol1+shap_vol2],\
                color=plt.cm.get_cmap(colormap,4).colors[2,:],\
                label='Inward\ninteraction',edgecolor='black')
        plt.bar([0.5,1.5],[shap4,shap_vol4],bottom=[shap1+shap2+shap3,\
                shap_vol1+shap_vol2+shap_vol3],\
        color=plt.cm.get_cmap(colormap,4).colors[3,:],\
                label='Sweeps',edgecolor='black')
        ax.set_ylabel('Fraction of the total',fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)   
        ax.set_position([0.2,0.2,0.35,0.65]) 
        ax.tick_params(axis='both', which='major', labelsize=fs)
        ax.set_xticks([0.5,1.5])
        ax.set_xticklabels(['$\Phi_e/\Phi_T$','$\Phi^V_e/\Phi^V_T$'])
        ax.xaxis.get_offset_text().set_fontsize(fs)
        plt.legend(fontsize=fs, bbox_to_anchor=(1, 1))
        ax.grid()  
        plt.savefig('bar_SHAP_noback_'+colormap+'_30+.png')
        file_save = open('bar_SHAP_noback.txt', "w+") 
        file_save.write('SHAP percentage: \nOutward Int. '+str(shap1)+\
                        '\nEjections '+str(shap2)+'\nInward Int. '+str(shap3)+\
                        '\nSweeps '+str(shap4)+'\n')
        file_save.write('SHAP per volume percentage: \nOutward Int. '+\
                        str(shap_vol1)+'\nEjections '+str(shap_vol2)+\
                        '\nInward Int. '+str(shap_vol3)+'\nSweeps '+\
                        str(shap_vol4)+'\n')
        file_save.close()            
            
        
    def plot_shaps_pdf_wall(self,
                            colormap='viridis',
                            bin_num=100,
                            lev_val=2.5,
                            alf=0.5,
                            structures=[],
                            mode='mse'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        
        xhistmin = np.min([np.min(self.volume_1),
                           np.min(self.volume_2),
                           np.min(self.volume_3),
                           np.min(self.volume_4)]\
                          +[np.min(self.volume[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.volume_1),
                           np.max(self.volume_2),
                           np.max(self.volume_3),
                           np.max(self.volume_4)]\
                          +[np.max(self.volume[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1),
                           np.min(self.shap_2),
                           np.min(self.shap_3),
                           np.min(self.shap_4)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1),
                           np.max(self.shap_2),
                           np.max(self.shap_3),
                           np.max(self.shap_4)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        
        
        histogram1_wa,vol_value1_wa,shap_value1_wa = np.histogram2d(self.volume_1_wa,self.shap_1_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_wa,vol_value2_wa,shap_value2_wa = np.histogram2d(self.volume_2_wa,self.shap_2_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_wa,vol_value3_wa,shap_value3_wa = np.histogram2d(self.volume_3_wa,self.shap_3_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_wa,vol_value4_wa,shap_value4_wa = np.histogram2d(self.volume_4_wa,self.shap_4_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1_wa = vol_value1_wa[:-1]+np.diff(vol_value1_wa)/2
        shap_value1_wa = shap_value1_wa[:-1]+np.diff(shap_value1_wa)/2
        vol_value2_wa = vol_value2_wa[:-1]+np.diff(vol_value2_wa)/2
        shap_value2_wa = shap_value2_wa[:-1]+np.diff(shap_value2_wa)/2
        vol_value3_wa = vol_value3_wa[:-1]+np.diff(vol_value3_wa)/2
        shap_value3_wa = shap_value3_wa[:-1]+np.diff(shap_value3_wa)/2
        vol_value4_wa = vol_value4_wa[:-1]+np.diff(vol_value4_wa)/2
        shap_value4_wa = shap_value4_wa[:-1]+np.diff(shap_value4_wa)/2
        
        histograms_wa = {}
        vol_values_wa = {}
        shap_values_wa = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume_wa[structure],
                                                            self.shap_wa[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_wa[structure] = histogram
            vol_values_wa[structure] = vol_value
            shap_values_wa[structure] = shap_value 
            
            
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        histogram1_wd,vol_value1_wd,shap_value1_wd = np.histogram2d(self.volume_1_wd,self.shap_1_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_wd,vol_value2_wd,shap_value2_wd = np.histogram2d(self.volume_2_wd,self.shap_2_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_wd,vol_value3_wd,shap_value3_wd = np.histogram2d(self.volume_3_wd,self.shap_3_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_wd,vol_value4_wd,shap_value4_wd = np.histogram2d(self.volume_4_wd,self.shap_4_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1_wd = vol_value1_wd[:-1]+np.diff(vol_value1_wd)/2
        shap_value1_wd = shap_value1_wd[:-1]+np.diff(shap_value1_wd)/2
        vol_value2_wd = vol_value2_wd[:-1]+np.diff(vol_value2_wd)/2
        shap_value2_wd = shap_value2_wd[:-1]+np.diff(shap_value2_wd)/2
        vol_value3_wd = vol_value3_wd[:-1]+np.diff(vol_value3_wd)/2
        shap_value3_wd = shap_value3_wd[:-1]+np.diff(shap_value3_wd)/2
        vol_value4_wd = vol_value4_wd[:-1]+np.diff(vol_value4_wd)/2
        shap_value4_wd = shap_value4_wd[:-1]+np.diff(shap_value4_wd)/2
        
        histograms_wd = {}
        vol_values_wd = {}
        shap_values_wd = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume_wd[structure],
                                                            self.shap_wd[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_wd[structure] = histogram
            vol_values_wd[structure] = vol_value
            shap_values_wd[structure] = shap_value 
        
        min_vol = np.min([vol_value1_wa,
                          vol_value2_wa,
                          vol_value3_wa,
                          vol_value4_wa,
                          vol_value1_wd,
                          vol_value2_wd,
                          vol_value3_wd,
                          vol_value4_wd]\
                         +[vol_values_wa[struc] for struc in structures]\
                         +[vol_values_wd[struc] for struc in structures])
            
        max_vol = np.max([vol_value1_wa,
                          vol_value2_wa,
                          vol_value3_wa,
                          vol_value4_wa,
                          vol_value1_wd,
                          vol_value2_wd,
                          vol_value3_wd,
                          vol_value4_wd]\
                         +[vol_values_wa[struc] for struc in structures]\
                         +[vol_values_wd[struc] for struc in structures])
            
        min_shap = np.min([shap_value1_wa,
                           shap_value2_wa,
                           shap_value3_wa,
                           shap_value4_wa,
                           shap_value1_wd,
                           shap_value2_wd,
                           shap_value3_wd,
                           shap_value4_wd]\
                          +[shap_values_wa[struc] for struc in structures]\
                          +[shap_values_wd[struc] for struc in structures])
        
        max_shap = np.max([shap_value1_wa,
                           shap_value2_wa,
                           shap_value3_wa,
                           shap_value4_wa,
                           shap_value1_wd,
                           shap_value2_wd,
                           shap_value3_wd,
                           shap_value4_wd]\
                          +[shap_values_wa[struc] for struc in structures]\
                          +[shap_values_wd[struc] for struc in structures])
        
        # min_vol = np.min([vol_value1_wa,vol_value2_wa,vol_value3_wa,vol_value4_wa,vol_value1_wd,vol_value2_wd,vol_value3_wd,vol_value4_wd])
        # max_vol = np.max([vol_value1_wa,vol_value2_wa,vol_value3_wa,vol_value4_wa,vol_value1_wd,vol_value2_wd,vol_value3_wd,vol_value4_wd])
        # min_shap = np.min([shap_value1_wa,shap_value2_wa,shap_value3_wa,shap_value4_wa,shap_value1_wd,shap_value2_wd,shap_value3_wd,shap_value4_wd])
        # max_shap = np.max([shap_value1_wa,shap_value2_wa,shap_value3_wa,shap_value4_wa,shap_value1_wd,shap_value2_wd,shap_value3_wd,shap_value4_wd])
        interp_h1_wa = interp2d(vol_value1_wa,shap_value1_wa,histogram1_wa)
        interp_h2_wa = interp2d(vol_value2_wa,shap_value2_wa,histogram2_wa)
        interp_h3_wa = interp2d(vol_value3_wa,shap_value3_wa,histogram3_wa)
        interp_h4_wa = interp2d(vol_value4_wa,shap_value4_wa,histogram4_wa)
        interp_h1_wd = interp2d(vol_value1_wd,shap_value1_wd,histogram1_wd)
        interp_h2_wd = interp2d(vol_value2_wd,shap_value2_wd,histogram2_wd)
        interp_h3_wd = interp2d(vol_value3_wd,shap_value3_wd,histogram3_wd)
        interp_h4_wd = interp2d(vol_value4_wd,shap_value4_wd,histogram4_wd)
        vec_vol = np.linspace(min_vol,max_vol,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        vol_grid,shap_grid = np.meshgrid(vec_vol,vec_shap)
        histogram_Q1_wa = interp_h1_wa(vec_vol,vec_shap)
        histogram_Q2_wa = interp_h2_wa(vec_vol,vec_shap)
        histogram_Q3_wa = interp_h3_wa(vec_vol,vec_shap)
        histogram_Q4_wa = interp_h4_wa(vec_vol,vec_shap)
        histogram_Q1_wd = interp_h1_wd(vec_vol,vec_shap)
        histogram_Q2_wd = interp_h2_wd(vec_vol,vec_shap)
        histogram_Q3_wd = interp_h3_wd(vec_vol,vec_shap)
        histogram_Q4_wd = interp_h4_wd(vec_vol,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
        
        histograms_struc_wa = {}
        for structure in structures:
            interp_h = interp2d(vol_values_wa[structure],
                                shap_values_wa[structure],
                                histograms_wa[structure])
            histograms_struc_wa[structure] = interp_h(vec_vol,vec_shap)
            
        histograms_struc_wd = {}
        for structure in structures:
            interp_h = interp2d(vol_values_wd[structure],
                                shap_values_wd[structure],
                                histograms_wd[structure])
            histograms_struc_wd[structure] = interp_h(vec_vol,vec_shap)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        # plt.contourf(vol_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val],colors=[(color11,color12,color13)],linestyles='dashed')
        plt.contour(vol_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val],colors=[(color21,color22,color23)],linestyles='dashed')
        # plt.contour(vol_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val],colors=[(color31,color32,color33)],linestyles='dashed')
        plt.contour(vol_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val],colors=[(color41,color42,color43)],linestyles='dashed')
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)],
                        linestyles='dashed')
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^6$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        handles += [mpl.lines.Line2D([0],[0],linestyle='solid',color='k'),
                    mpl.lines.Line2D([0],[0],linestyle='dashed',color='k')]
        labels += ['W-A','W-D']
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        
        if mode =='mse':
            plt.tight_layout()
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode =='cf':
            plt.tight_layout() # rect=(0.02,0,1,1)
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3])
        plt.savefig('hist2d_interp_vol_SHAP_'+colormap+str(structures)+'_30+_wall.png')
        
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^6$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        if mode == 'mse':
            plt.tight_layout()
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([2,4,6])
            plt.ylim([0,7])
        elif mode == 'cf':
            plt.tight_layout() #rect=(0.02,0,1,1)
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,3])
        plt.savefig('hist2d_interp_vol_SHAP_'+colormap+str(structures)+'_30+_wallattach.png')
        
        
        if mode == 'cf':
            shap_grid *= 10
        vol_grid *= 10
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(vol_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid,
                         shap_grid,
                         histograms_struc_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(vol_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(vol_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid,
                        shap_grid,
                        histograms_struc_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^5$',\
                   fontsize=fs)
        plt.ylabel(self.ylabel_shap,fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        if mode == 'mse':
            plt.tight_layout() # rect=(0.02,0,1,1)
            plt.xticks([2,4,6,8])
            plt.xlim([0.2,10])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0.02,2])
        elif mode == 'cf':
            plt.tight_layout()
            plt.xticks([2,4,6])
            plt.xlim([0.3,8])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0.3,6])    
        plt.savefig('hist2d_interp_vol_SHAP_'+colormap+str(structures)+'_30+_walldetach.png')
        
        xhistmin = np.min([np.min(self.volume_1),
                           np.min(self.volume_2),
                           np.min(self.volume_3),
                           np.min(self.volume_4)]\
                          +[np.min(self.volume[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(self.volume_1),
                           np.max(self.volume_2),
                           np.max(self.volume_3),
                           np.max(self.volume_4)]\
                          +[np.max(self.volume[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol_wa,vol_value1_vol_wa,shap_value1_vol_wa = np.histogram2d(self.volume_1_wa,self.shap_1_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol_wa,vol_value2_vol_wa,shap_value2_vol_wa = np.histogram2d(self.volume_2_wa,self.shap_2_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol_wa,vol_value3_vol_wa,shap_value3_vol_wa = np.histogram2d(self.volume_3_wa,self.shap_3_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol_wa,vol_value4_vol_wa,shap_value4_vol_wa = np.histogram2d(self.volume_4_wa,self.shap_4_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1_vol_wa = vol_value1_vol_wa[:-1]+np.diff(vol_value1_vol_wa)/2
        shap_value1_vol_wa = shap_value1_vol_wa[:-1]+np.diff(shap_value1_vol_wa)/2
        vol_value2_vol_wa = vol_value2_vol_wa[:-1]+np.diff(vol_value2_vol_wa)/2
        shap_value2_vol_wa = shap_value2_vol_wa[:-1]+np.diff(shap_value2_vol_wa)/2
        vol_value3_vol_wa = vol_value3_vol_wa[:-1]+np.diff(vol_value3_vol_wa)/2
        shap_value3_vol_wa = shap_value3_vol_wa[:-1]+np.diff(shap_value3_vol_wa)/2
        vol_value4_vol_wa = vol_value4_vol_wa[:-1]+np.diff(vol_value4_vol_wa)/2
        shap_value4_vol_wa = shap_value4_vol_wa[:-1]+np.diff(shap_value4_vol_wa)/2
        
        histograms_vol_wa = {}
        vol_values_vol_wa = {}
        shap_values_vol_wa = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume_wa[structure],
                                                            self.shap_wa_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol_wa[structure] = histogram
            vol_values_vol_wa[structure] = vol_value
            shap_values_vol_wa[structure] = shap_value
        
        # xhistmin = np.min([np.min(self.volume_1),np.min(self.volume_2),np.min(self.volume_3),np.min(self.volume_4)])/1.2
        # xhistmax = np.max([np.max(self.volume_1),np.max(self.volume_2),np.max(self.volume_3),np.max(self.volume_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        histogram1_vol_wd,vol_value1_vol_wd,shap_value1_vol_wd = np.histogram2d(self.volume_1_wd,self.shap_1_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol_wd,vol_value2_vol_wd,shap_value2_vol_wd = np.histogram2d(self.volume_2_wd,self.shap_2_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol_wd,vol_value3_vol_wd,shap_value3_vol_wd = np.histogram2d(self.volume_3_wd,self.shap_3_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol_wd,vol_value4_vol_wd,shap_value4_vol_wd = np.histogram2d(self.volume_4_wd,self.shap_4_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        vol_value1_vol_wd = vol_value1_vol_wd[:-1]+np.diff(vol_value1_vol_wd)/2
        shap_value1_vol_wd = shap_value1_vol_wd[:-1]+np.diff(shap_value1_vol_wd)/2
        vol_value2_vol_wd = vol_value2_vol_wd[:-1]+np.diff(vol_value2_vol_wd)/2
        shap_value2_vol_wd = shap_value2_vol_wd[:-1]+np.diff(shap_value2_vol_wd)/2
        vol_value3_vol_wd = vol_value3_vol_wd[:-1]+np.diff(vol_value3_vol_wd)/2
        shap_value3_vol_wd = shap_value3_vol_wd[:-1]+np.diff(shap_value3_vol_wd)/2
        vol_value4_vol_wd = vol_value4_vol_wd[:-1]+np.diff(vol_value4_vol_wd)/2
        shap_value4_vol_wd = shap_value4_vol_wd[:-1]+np.diff(shap_value4_vol_wd)/2
        
        histograms_vol_wd = {}
        vol_values_vol_wd = {}
        shap_values_vol_wd = {}
        for structure in structures:
            histogram,vol_value,shap_value = np.histogram2d(self.volume_wd[structure],
                                                            self.shap_wd_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            vol_value = vol_value[:-1]+np.diff(vol_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol_wd[structure] = histogram
            vol_values_vol_wd[structure] = vol_value
            shap_values_vol_wd[structure] = shap_value 
            
        min_vol_vol = np.min([vol_value1_vol_wa,
                          vol_value2_vol_wa,
                          vol_value3_vol_wa,
                          vol_value4_vol_wa,
                          vol_value1_vol_wd,
                          vol_value2_vol_wd,
                          vol_value3_vol_wd,
                          vol_value4_vol_wd]\
                         +[vol_values_vol_wa[struc] for struc in structures]\
                         +[vol_values_vol_wd[struc] for struc in structures])
            
        max_vol_vol = np.max([vol_value1_vol_wa,
                          vol_value2_vol_wa,
                          vol_value3_vol_wa,
                          vol_value4_vol_wa,
                          vol_value1_vol_wd,
                          vol_value2_vol_wd,
                          vol_value3_vol_wd,
                          vol_value4_vol_wd]\
                         +[vol_values_vol_wa[struc] for struc in structures]\
                         +[vol_values_vol_wd[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol_wa,
                           shap_value2_vol_wa,
                           shap_value3_vol_wa,
                           shap_value4_vol_wa,
                           shap_value1_vol_wd,
                           shap_value2_vol_wd,
                           shap_value3_vol_wd,
                           shap_value4_vol_wd]\
                          +[shap_values_vol_wa[struc] for struc in structures]\
                          +[shap_values_vol_wd[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol_wa,
                           shap_value2_vol_wa,
                           shap_value3_vol_wa,
                           shap_value4_vol_wa,
                           shap_value1_vol_wd,
                           shap_value2_vol_wd,
                           shap_value3_vol_wd,
                           shap_value4_vol_wd]\
                          +[shap_values_vol_wa[struc] for struc in structures]\
                          +[shap_values_vol_wd[struc] for struc in structures])
        
        # min_vol_vol = np.min([vol_value1_vol_wa,vol_value2_vol_wa,vol_value3_vol_wa,vol_value4_vol_wa,vol_value1_vol_wd,vol_value2_vol_wd,vol_value3_vol_wd,vol_value4_vol_wd])
        # max_vol_vol = np.max([vol_value1_vol_wa,vol_value2_vol_wa,vol_value3_vol_wa,vol_value4_vol_wa,vol_value1_vol_wd,vol_value2_vol_wd,vol_value3_vol_wd,vol_value4_vol_wd])
        # min_shap_vol = np.min([shap_value1_vol_wa,shap_value2_vol_wa,shap_value3_vol_wa,shap_value4_vol_wa,shap_value1_vol_wd,shap_value2_vol_wd,shap_value3_vol_wd,shap_value4_vol_wd])
        # max_shap_vol = np.max([shap_value1_vol_wa,shap_value2_vol_wa,shap_value3_vol_wa,shap_value4_vol_wa,shap_value1_vol_wd,shap_value2_vol_wd,shap_value3_vol_wd,shap_value4_vol_wd])
        interp_h1_vol_wa = interp2d(vol_value1_vol_wa,shap_value1_vol_wa,histogram1_vol_wa)
        interp_h2_vol_wa = interp2d(vol_value2_vol_wa,shap_value2_vol_wa,histogram2_vol_wa)
        interp_h3_vol_wa = interp2d(vol_value3_vol_wa,shap_value3_vol_wa,histogram3_vol_wa)
        interp_h4_vol_wa = interp2d(vol_value4_vol_wa,shap_value4_vol_wa,histogram4_vol_wa)
        interp_h1_vol_wd = interp2d(vol_value1_vol_wd,shap_value1_vol_wd,histogram1_vol_wd)
        interp_h2_vol_wd = interp2d(vol_value2_vol_wd,shap_value2_vol_wd,histogram2_vol_wd)
        interp_h3_vol_wd = interp2d(vol_value3_vol_wd,shap_value3_vol_wd,histogram3_vol_wd)
        interp_h4_vol_wd = interp2d(vol_value4_vol_wd,shap_value4_vol_wd,histogram4_vol_wd)
        vec_vol_vol = np.linspace(min_vol_vol,max_vol_vol,1000) 
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        vol_grid_vol,shap_grid_vol = np.meshgrid(vec_vol_vol,vec_shap_vol)
        histogram_Q1_vol_wa = interp_h1_vol_wa(vec_vol_vol,vec_shap_vol)
        histogram_Q2_vol_wa = interp_h2_vol_wa(vec_vol_vol,vec_shap_vol)
        histogram_Q3_vol_wa = interp_h3_vol_wa(vec_vol_vol,vec_shap_vol)
        histogram_Q4_vol_wa = interp_h4_vol_wa(vec_vol_vol,vec_shap_vol)
        histogram_Q1_vol_wd = interp_h1_vol_wd(vec_vol_vol,vec_shap_vol)
        histogram_Q2_vol_wd = interp_h2_vol_wd(vec_vol_vol,vec_shap_vol)
        histogram_Q3_vol_wd = interp_h3_vol_wd(vec_vol_vol,vec_shap_vol)
        histogram_Q4_vol_wd = interp_h4_vol_wd(vec_vol_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol_wa = {}
        for structure in structures:
            interp_h = interp2d(vol_values_vol_wa[structure],
                                shap_values_vol_wa[structure],
                                histograms_vol_wa[structure])
            histograms_struc_vol_wa[structure] = interp_h(vec_vol_vol,vec_shap_vol)
            
        histograms_struc_vol_wd = {}
        for structure in structures:
            interp_h = interp2d(vol_values_vol_wd[structure],
                                shap_values_vol_wd[structure],
                                histograms_vol_wd[structure])
            histograms_struc_vol_wd[structure] = interp_h(vec_vol_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val],colors=[(color11,color12,color13)],linestyles='dashed')
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val],colors=[(color31,color32,color33)],linestyles='dashed')
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val],colors=[(color21,color22,color23)],linestyles='dashed')
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val],colors=[(color41,color42,color43)],linestyles='dashed')
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)],
                        linestyles='dashed')
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^{6}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        handles += [mpl.lines.Line2D([0],[0],linestyle='solid',color='k'),
                    mpl.lines.Line2D([0],[0],linestyle='dashed',color='k')]
        labels += ['W-A','W-D']
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3.5])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
        elif mode == 'cf':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,4])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,2])
        plt.savefig('hist2d_interp_vol_SHAPvol_'+colormap+str(structures)+'_30+_wall.png')
        
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^{6}$',\
                   fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels= ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,3.5])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,5.5])
        elif mode == 'cf':
            plt.xticks([1,2,3,4,5,6,7,8,9])
            plt.xlim([0,4])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,2])
        plt.savefig('hist2d_interp_vol_SHAPvol_'+colormap+str(structures)+'_30+_wallattach.png')
        
        if mode == 'cf':
            shap_grid_vol *= 10
        vol_grid_vol *= 10 
        
        
        fs = 20
        plt.figure()
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        # plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        plt.contourf(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(vol_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val],colors=[(color11,color12,color13)])
        # plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val],colors=[(color21,color22,color23)])
        plt.contour(vol_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(vol_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        plt.xlabel('$V^+\cdot10^{5}$',\
                   fontsize=fs) 
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','8'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels= ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        if mode == 'mse':
            plt.xticks([2,4,6,8])
            plt.xlim([0.1,10])
            plt.yticks([1,2,3,4,5,6,7,8,9])
            plt.ylim([0,4.5])
        elif mode == 'cf':
            plt.xticks([2,4,6,8])
            plt.xlim([0.3,8])
            plt.yticks([2,4,6,8])
            plt.ylim([0.1,8])
        plt.savefig('hist2d_interp_vol_SHAPvol_'+colormap+str(structures)+'_30+_walldetach.png')
        
        
    def plot_shaps_x_pdf_wall(self,
                               colormap='viridis',
                               bin_num=100,
                               lev_val=2.5,
                               alf=0.5,
                               structures=[],
                               x='uv',
                               mode='mse'):
        """ 
        Function for plotting the results of the SHAP vs the Reynolds stress
        """
        import matplotlib.pyplot as plt
        import matplotlib as mpl    
        from scipy.interpolate import interp2d
        
        if x == 'uv':
            xhistmin = np.min([np.min(self.uv_uvtot_1),
                               np.min(self.uv_uvtot_2),
                               np.min(self.uv_uvtot_3),
                               np.min(self.uv_uvtot_4)]\
                              +[np.min(self.uv_uvtot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.uv_uvtot_1),
                               np.max(self.uv_uvtot_2), 
                               np.max(self.uv_uvtot_3),
                               np.max(self.uv_uvtot_4)]\
                              +[np.max(self.uv_uvtot[struc]) for struc in structures]
                              )*1.2
                
            yhistmin = np.min([np.min(self.shap_1),
                               np.min(self.shap_2),
                               np.min(self.shap_3),
                               np.min(self.shap_4)]\
                              +[np.min(self.shap[struc]) for struc in structures]
                              )/1.2
            
            yhistmax = np.max([np.max(self.shap_1),
                               np.max(self.shap_2),
                               np.max(self.shap_3),
                               np.max(self.shap_4)]
                              +[np.max(self.shap[struc]) for struc in structures]
                              )*1.2
            
            histogram1_wa,uv_value1_wa,shap_value1_wa = np.histogram2d(self.uv_uvtot_1_wa,self.shap_1_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_wa,uv_value2_wa,shap_value2_wa = np.histogram2d(self.uv_uvtot_2_wa,self.shap_2_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_wa,uv_value3_wa,shap_value3_wa = np.histogram2d(self.uv_uvtot_3_wa,self.shap_3_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_wa,uv_value4_wa,shap_value4_wa = np.histogram2d(self.uv_uvtot_4_wa,self.shap_4_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            
        elif x == 'k':
            xhistmin = np.min([np.min(self.k_ktot_1),
                               np.min(self.k_ktot_2),
                               np.min(self.k_ktot_3),
                               np.min(self.k_ktot_4)]\
                              +[np.min(self.k_ktot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.k_ktot_1),
                               np.max(self.k_ktot_2), 
                               np.max(self.k_ktot_3),
                               np.max(self.k_ktot_4)]\
                              +[np.max(self.k_ktot[struc]) for struc in structures]
                              )*1.2
                
            yhistmin = np.min([np.min(self.shap_1),
                               np.min(self.shap_2),
                               np.min(self.shap_3),
                               np.min(self.shap_4)]\
                              +[np.min(self.shap[struc]) for struc in structures]
                              )/1.2
            
            yhistmax = np.max([np.max(self.shap_1),
                               np.max(self.shap_2),
                               np.max(self.shap_3),
                               np.max(self.shap_4)]
                              +[np.max(self.shap[struc]) for struc in structures]
                              )*1.2
            
            histogram1_wa,uv_value1_wa,shap_value1_wa = np.histogram2d(self.k_ktot_1_wa,self.shap_1_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_wa,uv_value2_wa,shap_value2_wa = np.histogram2d(self.k_ktot_2_wa,self.shap_2_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_wa,uv_value3_wa,shap_value3_wa = np.histogram2d(self.k_ktot_3_wa,self.shap_3_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_wa,uv_value4_wa,shap_value4_wa = np.histogram2d(self.k_ktot_4_wa,self.shap_4_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            
        elif x == 'ens':
            xhistmin = np.min([np.min(self.ens_enstot_1),
                               np.min(self.ens_enstot_2),
                               np.min(self.ens_enstot_3),
                               np.min(self.ens_enstot_4)]\
                              +[np.min(self.ens_enstot[struc]) for struc in structures]
                              )/1.2
                
            xhistmax = np.max([np.max(self.ens_enstot_1),
                               np.max(self.ens_enstot_2), 
                               np.max(self.ens_enstot_3),
                               np.max(self.ens_enstot_4)]\
                              +[np.max(self.ens_enstot[struc]) for struc in structures]
                              )*1.2
                
            yhistmin = np.min([np.min(self.shap_1),
                               np.min(self.shap_2),
                               np.min(self.shap_3),
                               np.min(self.shap_4)]\
                              +[np.min(self.shap[struc]) for struc in structures]
                              )/1.2
            
            yhistmax = np.max([np.max(self.shap_1),
                               np.max(self.shap_2),
                               np.max(self.shap_3),
                               np.max(self.shap_4)]
                              +[np.max(self.shap[struc]) for struc in structures]
                              )*1.2
            
            histogram1_wa,uv_value1_wa,shap_value1_wa = np.histogram2d(self.ens_enstot_1_wa,self.shap_1_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram2_wa,uv_value2_wa,shap_value2_wa = np.histogram2d(self.ens_enstot_2_wa,self.shap_2_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram3_wa,uv_value3_wa,shap_value3_wa = np.histogram2d(self.ens_enstot_3_wa,self.shap_3_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            histogram4_wa,uv_value4_wa,shap_value4_wa = np.histogram2d(self.ens_enstot_4_wa,self.shap_4_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
            
        # xhistmin = np.min([np.min(self.uv_uvtot_1),np.min(self.uv_uvtot_2),np.min(self.uv_uvtot_3),np.min(self.uv_uvtot_4)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1),np.max(self.uv_uvtot_2),np.max(self.uv_uvtot_3),np.max(self.uv_uvtot_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        
        uv_value1_wa = uv_value1_wa[:-1]+np.diff(uv_value1_wa)/2
        shap_value1_wa = shap_value1_wa[:-1]+np.diff(shap_value1_wa)/2
        uv_value2_wa = uv_value2_wa[:-1]+np.diff(uv_value2_wa)/2
        shap_value2_wa = shap_value2_wa[:-1]+np.diff(shap_value2_wa)/2
        uv_value3_wa = uv_value3_wa[:-1]+np.diff(uv_value3_wa)/2
        shap_value3_wa = shap_value3_wa[:-1]+np.diff(shap_value3_wa)/2
        uv_value4_wa = uv_value4_wa[:-1]+np.diff(uv_value4_wa)/2
        shap_value4_wa = shap_value4_wa[:-1]+np.diff(shap_value4_wa)/2
        
        histograms_wa = {}
        uv_values_wa = {}
        shap_values_wa = {}
        for structure in structures:
            if x == 'uv':
                x_xtot_wa = self.uv_uvtot_wa[structure]
            elif x == 'k':
                x_xtot_wa = self.k_ktot_wa[structure]
            elif x == 'ens':
                x_xtot_wa = self.ens_enstot_wa[structure]
            histogram,uv_value,shap_value = np.histogram2d(x_xtot_wa,
                                                           self.shap_wa[structure],
                                                           bins=bin_num,
                                                           range=[[xhistmin,xhistmax],
                                                                  [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_wa[structure] = histogram
            uv_values_wa[structure] = uv_value
            shap_values_wa[structure] = shap_value 
        
        # xhistmin = np.min([np.min(self.uv_uvtot_1),np.min(self.uv_uvtot_2),np.min(self.uv_uvtot_3),np.min(self.uv_uvtot_4)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1),np.max(self.uv_uvtot_2),np.max(self.uv_uvtot_3),np.max(self.uv_uvtot_4)])*1.2
        # yhistmin = np.min([np.min(self.shap_1),np.min(self.shap_2),np.min(self.shap_3),np.min(self.shap_4)])/1.2
        # yhistmax = np.max([np.max(self.shap_1),np.max(self.shap_2),np.max(self.shap_3),np.max(self.shap_4)])*1.2
        if x == 'uv':
            x_xtot_1_wd = self.uv_uvtot_1_wd
            x_xtot_2_wd = self.uv_uvtot_2_wd
            x_xtot_3_wd = self.uv_uvtot_3_wd
            x_xtot_4_wd = self.uv_uvtot_4_wd
        elif x == 'k':
            x_xtot_1_wd = self.k_ktot_1_wd
            x_xtot_2_wd = self.k_ktot_2_wd
            x_xtot_3_wd = self.k_ktot_3_wd
            x_xtot_4_wd = self.k_ktot_4_wd
        elif x == 'ens':
            x_xtot_1_wd = self.ens_enstot_1_wd
            x_xtot_2_wd = self.ens_enstot_2_wd
            x_xtot_3_wd = self.ens_enstot_3_wd
            x_xtot_4_wd = self.ens_enstot_4_wd
        histogram1_wd,uv_value1_wd,shap_value1_wd = np.histogram2d(x_xtot_1_wd,self.shap_1_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_wd,uv_value2_wd,shap_value2_wd = np.histogram2d(x_xtot_2_wd,self.shap_2_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_wd,uv_value3_wd,shap_value3_wd = np.histogram2d(x_xtot_3_wd,self.shap_3_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_wd,uv_value4_wd,shap_value4_wd = np.histogram2d(x_xtot_4_wd,self.shap_4_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        uv_value1_wd = uv_value1_wd[:-1]+np.diff(uv_value1_wd)/2
        shap_value1_wd = shap_value1_wd[:-1]+np.diff(shap_value1_wd)/2
        uv_value2_wd = uv_value2_wd[:-1]+np.diff(uv_value2_wd)/2
        shap_value2_wd = shap_value2_wd[:-1]+np.diff(shap_value2_wd)/2
        uv_value3_wd = uv_value3_wd[:-1]+np.diff(uv_value3_wd)/2
        shap_value3_wd = shap_value3_wd[:-1]+np.diff(shap_value3_wd)/2
        uv_value4_wd = uv_value4_wd[:-1]+np.diff(uv_value4_wd)/2
        shap_value4_wd = shap_value4_wd[:-1]+np.diff(shap_value4_wd)/2
        
        histograms_wd = {}
        uv_values_wd = {}
        shap_values_wd = {}
        for structure in structures:
            if x == 'uv':
                x_xtot_wd = self.uv_uvtot_wd[structure]
            elif x == 'k':
                x_xtot_wd = self.k_ktot_wd[structure]
            elif x == 'ens':
                x_xtot_wd = self.ens_enstot_wd[structure]
            histogram,uv_value,shap_value = np.histogram2d(x_xtot_wd,
                                                            self.shap_wd[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_wd[structure] = histogram
            uv_values_wd[structure] = uv_value
            shap_values_wd[structure] = shap_value 
            
        
        min_uv = np.min([uv_value1_wa,
                          uv_value2_wa,
                          uv_value3_wa,
                          uv_value4_wa,
                          uv_value1_wd,
                          uv_value2_wd,
                          uv_value3_wd,
                          uv_value4_wd]\
                         +[uv_values_wa[struc] for struc in structures]\
                         +[uv_values_wd[struc] for struc in structures])
            
        max_uv = np.max([uv_value1_wa,
                          uv_value2_wa,
                          uv_value3_wa,
                          uv_value4_wa,
                          uv_value1_wd,
                          uv_value2_wd,
                          uv_value3_wd,
                          uv_value4_wd]\
                         +[uv_values_wa[struc] for struc in structures]\
                         +[uv_values_wd[struc] for struc in structures])
            
        min_shap = np.min([shap_value1_wa,
                           shap_value2_wa,
                           shap_value3_wa,
                           shap_value4_wa,
                           shap_value1_wd,
                           shap_value2_wd,
                           shap_value3_wd,
                           shap_value4_wd]\
                          +[shap_values_wa[struc] for struc in structures]\
                          +[shap_values_wd[struc] for struc in structures])
        
        max_shap = np.max([shap_value1_wa,
                           shap_value2_wa,
                           shap_value3_wa,
                           shap_value4_wa,
                           shap_value1_wd,
                           shap_value2_wd,
                           shap_value3_wd,
                           shap_value4_wd]\
                          +[shap_values_wa[struc] for struc in structures]\
                          +[shap_values_wd[struc] for struc in structures])
        
        # min_uv = np.min([uv_value1_wa,uv_value2_wa,uv_value3_wa,uv_value4_wa,uv_value1_wd,uv_value2_wd,uv_value3_wd,uv_value4_wd])
        # max_uv = np.max([uv_value1_wa,uv_value2_wa,uv_value3_wa,uv_value4_wa,uv_value1_wd,uv_value2_wd,uv_value3_wd,uv_value4_wd])
        # min_shap = np.min([shap_value1_wa,shap_value2_wa,shap_value3_wa,shap_value4_wa,shap_value1_wd,shap_value2_wd,shap_value3_wd,shap_value4_wd])
        # max_shap = np.max([shap_value1_wa,shap_value2_wa,shap_value3_wa,shap_value4_wa,shap_value1_wd,shap_value2_wd,shap_value3_wd,shap_value4_wd])
        interp_h1_wa = interp2d(uv_value1_wa,shap_value1_wa,histogram1_wa)
        interp_h2_wa = interp2d(uv_value2_wa,shap_value2_wa,histogram2_wa)
        interp_h3_wa = interp2d(uv_value3_wa,shap_value3_wa,histogram3_wa)
        interp_h4_wa = interp2d(uv_value4_wa,shap_value4_wa,histogram4_wa)
        interp_h1_wd = interp2d(uv_value1_wd,shap_value1_wd,histogram1_wd)
        interp_h2_wd = interp2d(uv_value2_wd,shap_value2_wd,histogram2_wd)
        interp_h3_wd = interp2d(uv_value3_wd,shap_value3_wd,histogram3_wd)
        interp_h4_wd = interp2d(uv_value4_wd,shap_value4_wd,histogram4_wd)
        vec_uv = np.linspace(min_uv,max_uv,1000)
        vec_shap = np.linspace(min_shap,max_shap,1000)
        uv_grid,shap_grid = np.meshgrid(vec_uv,vec_shap)
        histogram_Q1_wa = interp_h1_wa(vec_uv,vec_shap)
        histogram_Q2_wa = interp_h2_wa(vec_uv,vec_shap)
        histogram_Q3_wa = interp_h3_wa(vec_uv,vec_shap)
        histogram_Q4_wa = interp_h4_wa(vec_uv,vec_shap)
        histogram_Q1_wd = interp_h1_wd(vec_uv,vec_shap)
        histogram_Q2_wd = interp_h2_wd(vec_uv,vec_shap)
        histogram_Q3_wd = interp_h3_wd(vec_uv,vec_shap)
        histogram_Q4_wd = interp_h4_wd(vec_uv,vec_shap)
        
        if mode == 'cf':
            shap_grid /= 10
        
        histograms_struc_wa = {}
        for structure in structures:
            interp_h = interp2d(uv_values_wa[structure],
                                shap_values_wa[structure],
                                histograms_wa[structure])
            histograms_struc_wa[structure] = interp_h(vec_uv,vec_shap)
            
        histograms_struc_wd = {}
        for structure in structures:
            interp_h = interp2d(uv_values_wd[structure],
                                shap_values_wd[structure],
                                histograms_wd[structure])
            histograms_struc_wd[structure] = interp_h(vec_uv,vec_shap)
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid,
                         shap_grid,
                         histograms_struc_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid,
                        shap_grid,
                        histograms_struc_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        # plt.contourf(uv_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid,
                         shap_grid,
                         histograms_struc_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val],colors=[(color11,color12,color13)],linestyles='dashed')
        plt.contour(uv_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val],colors=[(color21,color22,color23)],linestyles='dashed')
        # plt.contour(uv_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val],colors=[(color31,color32,color33)],linestyles='dashed')
        plt.contour(uv_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val],colors=[(color41,color42,color43)],linestyles='dashed')
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid,
                        shap_grid,
                        histograms_struc_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)],
                        linestyles='dashed')
        
        plt.grid()
    
        if mode == 'mse':
            if x == 'ens':
                plt.xticks([0.02,0.04])
                plt.xlim([0,0.05])
                plt.yticks([2,4,6])
                plt.ylim([0,7])
            else:
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([2,4,6])
                plt.ylim([0,7])
                
        elif mode == 'cf':
            if x == 'ens':
                plt.xticks([0.02,0.04])
                plt.xlim([0,0.05])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,2.5])
            elif x == 'k':
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,3.5])
            elif x == 'uv':
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,3.5])
            
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot})$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        handles += [mpl.lines.Line2D([0],[0],linestyle='solid',color='k'),
                    mpl.lines.Line2D([0],[0],linestyle='dashed',color='k')]
        labels += ['W-A','W-D']
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'tot_SHAP_'+colormap+str(structures)+'_30+_wall.png')
        
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid,
                         shap_grid,
                         histograms_struc_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid,shap_grid,histogram_Q1_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid,shap_grid,histogram_Q2_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid,shap_grid,histogram_Q3_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid,shap_grid,histogram_Q4_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid,
                        shap_grid,
                        histograms_struc_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        
        if mode == 'mse':
            if x == 'ens':
                plt.xticks([0.02, 0.04])
                plt.xlim([0,0.05])
                plt.yticks([2,4,6])
                plt.ylim([0,7])
            else:
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([2,4,6])
                plt.ylim([0,7])
            
        elif mode == 'cf':
            if x == 'ens':
                plt.xticks([0.02, 0.04])
                plt.xlim([0,0.05])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,2.5])
            elif x == 'k':
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,3.5])
            elif x == 'uv':
                plt.xticks([0.05, 0.10, 0.15])
                plt.xlim([0,0.2])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,3.5])
            
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot})$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap.replace('3','2'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'tot_SHAP_'+colormap+str(structures)+'_30+_wallattach.png')
        
        if mode == 'cf':
            shap_grid *= 10
        
        fs = 20
        plt.figure()        
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid,
                         shap_grid,
                         histograms_struc_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid,shap_grid,histogram_Q1_wd.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid,shap_grid,histogram_Q2_wd.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid,shap_grid,histogram_Q3_wd.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid,shap_grid,histogram_Q4_wd.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid,
                        shap_grid,
                        histograms_struc_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        plt.grid()
        
        if mode == 'mse':
            if x == 'ens':
                plt.xlim([0.0006,0.005])
                plt.yticks([0.5,1])
                plt.ylim([0.05,1.5])
            elif x == 'k':
                plt.xlim([0.001,0.025])
                plt.yticks([0.5,1])
                plt.ylim([0.05,1.3])
            elif x == 'uv':
                plt.xlim([0.0005,0.05])
                plt.yticks([0.5,1])
                plt.ylim([0.05,1.3])
                
        elif mode == 'cf':
            if x == 'ens':
                plt.xticks([0.001,0.002,0.003])
                plt.xlim([0.0006,0.004])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0.25,6])
            elif x == 'k':
                plt.xlim([0.001,0.025])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0.3,5])
            elif x == 'uv':
                plt.xlim([0.001,0.05])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0.3,5])
            
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot})$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot})$',\
                       fontsize=fs)
        plt.ylabel(self.ylabel_shap,fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'tot_SHAP_'+colormap+str(structures)+'_30+_walldetach.png')
        
        if x == 'uv':
            x_xtot_1_vol = self.uv_uvtot_1_vol
            x_xtot_2_vol = self.uv_uvtot_2_vol
            x_xtot_3_vol = self.uv_uvtot_3_vol
            x_xtot_4_vol = self.uv_uvtot_4_vol
            if len(structures) > 0:
                x_xtot_vol = self.uv_uvtot_vol
            
        elif x == 'k':
            x_xtot_1_vol = self.k_ktot_1_vol
            x_xtot_2_vol = self.k_ktot_2_vol
            x_xtot_3_vol = self.k_ktot_3_vol
            x_xtot_4_vol = self.k_ktot_4_vol
            if len(structures) > 0:
                x_xtot_vol = self.k_ktot_vol
        
        elif x == 'ens':
            x_xtot_1_vol = self.ens_enstot_1_vol
            x_xtot_2_vol = self.ens_enstot_2_vol
            x_xtot_3_vol = self.ens_enstot_3_vol
            x_xtot_4_vol = self.ens_enstot_4_vol
            if len(structures) > 0:
                x_xtot_vol = self.ens_enstot_vol
        
        xhistmin = np.min([np.min(x_xtot_1_vol),
                           np.min(x_xtot_2_vol),
                           np.min(x_xtot_3_vol),
                           np.min(x_xtot_4_vol)]\
                          +[np.min(x_xtot_vol[struc]) for struc in structures]
                          )/1.2
            
        xhistmax = np.max([np.max(x_xtot_1_vol),
                           np.max(x_xtot_2_vol), 
                           np.max(x_xtot_3_vol),
                           np.max(x_xtot_4_vol)]\
                          +[np.max(x_xtot_vol[struc]) for struc in structures]
                          )*1.2
            
        yhistmin = np.min([np.min(self.shap_1_vol),
                           np.min(self.shap_2_vol),
                           np.min(self.shap_3_vol),
                           np.min(self.shap_4_vol)]\
                          +[np.min(self.shap_vol[struc]) for struc in structures]
                          )/1.2
        
        yhistmax = np.max([np.max(self.shap_1_vol),
                           np.max(self.shap_2_vol),
                           np.max(self.shap_3_vol),
                           np.max(self.shap_4_vol)]
                          +[np.max(self.shap_vol[struc]) for struc in structures]
                          )*1.2
        
        # xhistmin = np.min([np.min(self.uv_uvtot_1_vol),np.min(self.uv_uvtot_2_vol),np.min(self.uv_uvtot_3_vol),np.min(self.uv_uvtot_4_vol)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1_vol),np.max(self.uv_uvtot_2_vol),np.max(self.uv_uvtot_3_vol),np.max(self.uv_uvtot_4_vol)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        
        if x == 'uv':
            x_xtot_1_vol_wa = self.uv_uvtot_1_vol_wa
            x_xtot_2_vol_wa = self.uv_uvtot_2_vol_wa
            x_xtot_3_vol_wa = self.uv_uvtot_3_vol_wa
            x_xtot_4_vol_wa = self.uv_uvtot_4_vol_wa
        elif x == 'k':
            x_xtot_1_vol_wa = self.k_ktot_1_vol_wa
            x_xtot_2_vol_wa = self.k_ktot_2_vol_wa
            x_xtot_3_vol_wa = self.k_ktot_3_vol_wa
            x_xtot_4_vol_wa = self.k_ktot_4_vol_wa
        elif x == 'ens':
            x_xtot_1_vol_wa = self.ens_enstot_1_vol_wa
            x_xtot_2_vol_wa = self.ens_enstot_2_vol_wa
            x_xtot_3_vol_wa = self.ens_enstot_3_vol_wa
            x_xtot_4_vol_wa = self.ens_enstot_4_vol_wa
        
        histogram1_vol_wa,uv_value1_vol_wa,shap_value1_vol_wa = np.histogram2d(x_xtot_1_vol_wa,self.shap_1_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol_wa,uv_value2_vol_wa,shap_value2_vol_wa = np.histogram2d(x_xtot_2_vol_wa,self.shap_2_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol_wa,uv_value3_vol_wa,shap_value3_vol_wa = np.histogram2d(x_xtot_3_vol_wa,self.shap_3_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol_wa,uv_value4_vol_wa,shap_value4_vol_wa = np.histogram2d(x_xtot_4_vol_wa,self.shap_4_vol_wa,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        uv_value1_vol_wa = uv_value1_vol_wa[:-1]+np.diff(uv_value1_vol_wa)/2
        shap_value1_vol_wa = shap_value1_vol_wa[:-1]+np.diff(shap_value1_vol_wa)/2
        uv_value2_vol_wa = uv_value2_vol_wa[:-1]+np.diff(uv_value2_vol_wa)/2
        shap_value2_vol_wa = shap_value2_vol_wa[:-1]+np.diff(shap_value2_vol_wa)/2
        uv_value3_vol_wa = uv_value3_vol_wa[:-1]+np.diff(uv_value3_vol_wa)/2
        shap_value3_vol_wa = shap_value3_vol_wa[:-1]+np.diff(shap_value3_vol_wa)/2
        uv_value4_vol_wa = uv_value4_vol_wa[:-1]+np.diff(uv_value4_vol_wa)/2
        shap_value4_vol_wa = shap_value4_vol_wa[:-1]+np.diff(shap_value4_vol_wa)/2
        
        histograms_vol_wa = {}
        uv_values_vol_wa = {}
        shap_values_vol_wa = {}
        for structure in structures:
            if x == 'uv':
                x_xtot_vol_wa = self.uv_uvtot_wa_vol[structure]
            elif x == 'k':
                x_xtot_vol_wa = self.k_ktot_wa_vol[structure]
            elif x == 'ens':
                x_xtot_vol_wa = self.ens_enstot_wa_vol[structure]
            histogram,uv_value,shap_value = np.histogram2d(x_xtot_vol_wa,
                                                           self.shap_wa_vol[structure],
                                                           bins=bin_num,
                                                           range=[[xhistmin,xhistmax],
                                                                  [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol_wa[structure] = histogram
            uv_values_vol_wa[structure] = uv_value
            shap_values_vol_wa[structure] = shap_value 
            
        # xhistmin = np.min([np.min(self.uv_uvtot_1_vol),np.min(self.uv_uvtot_2_vol),np.min(self.uv_uvtot_3_vol),np.min(self.uv_uvtot_4_vol)])/1.2
        # xhistmax = np.max([np.max(self.uv_uvtot_1_vol),np.max(self.uv_uvtot_2_vol),np.max(self.uv_uvtot_3_vol),np.max(self.uv_uvtot_4_vol)])*1.2
        # yhistmin = np.min([np.min(self.shap_1_vol),np.min(self.shap_2_vol),np.min(self.shap_3_vol),np.min(self.shap_4_vol)])/1.2
        # yhistmax = np.max([np.max(self.shap_1_vol),np.max(self.shap_2_vol),np.max(self.shap_3_vol),np.max(self.shap_4_vol)])*1.2
        
        if x == 'uv':
            x_xtot_1_vol_wd = self.uv_uvtot_1_vol_wd
            x_xtot_2_vol_wd = self.uv_uvtot_2_vol_wd
            x_xtot_3_vol_wd = self.uv_uvtot_3_vol_wd
            x_xtot_4_vol_wd = self.uv_uvtot_4_vol_wd
        elif x == 'k':
            x_xtot_1_vol_wd = self.k_ktot_1_vol_wd
            x_xtot_2_vol_wd = self.k_ktot_2_vol_wd
            x_xtot_3_vol_wd = self.k_ktot_3_vol_wd
            x_xtot_4_vol_wd = self.k_ktot_4_vol_wd
        elif x == 'ens':
            x_xtot_1_vol_wd = self.ens_enstot_1_vol_wd
            x_xtot_2_vol_wd = self.ens_enstot_2_vol_wd
            x_xtot_3_vol_wd = self.ens_enstot_3_vol_wd
            x_xtot_4_vol_wd = self.ens_enstot_4_vol_wd
        
        histogram1_vol_wd,uv_value1_vol_wd,shap_value1_vol_wd = np.histogram2d(x_xtot_1_vol_wd,self.shap_1_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram2_vol_wd,uv_value2_vol_wd,shap_value2_vol_wd = np.histogram2d(x_xtot_2_vol_wd,self.shap_2_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram3_vol_wd,uv_value3_vol_wd,shap_value3_vol_wd = np.histogram2d(x_xtot_3_vol_wd,self.shap_3_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        histogram4_vol_wd,uv_value4_vol_wd,shap_value4_vol_wd = np.histogram2d(x_xtot_4_vol_wd,self.shap_4_vol_wd,bins=bin_num,range=[[xhistmin,xhistmax],[yhistmin,yhistmax]])
        uv_value1_vol_wd = uv_value1_vol_wd[:-1]+np.diff(uv_value1_vol_wd)/2
        shap_value1_vol_wd = shap_value1_vol_wd[:-1]+np.diff(shap_value1_vol_wd)/2
        uv_value2_vol_wd = uv_value2_vol_wd[:-1]+np.diff(uv_value2_vol_wd)/2
        shap_value2_vol_wd = shap_value2_vol_wd[:-1]+np.diff(shap_value2_vol_wd)/2
        uv_value3_vol_wd = uv_value3_vol_wd[:-1]+np.diff(uv_value3_vol_wd)/2
        shap_value3_vol_wd = shap_value3_vol_wd[:-1]+np.diff(shap_value3_vol_wd)/2
        uv_value4_vol_wd = uv_value4_vol_wd[:-1]+np.diff(uv_value4_vol_wd)/2
        shap_value4_vol_wd = shap_value4_vol_wd[:-1]+np.diff(shap_value4_vol_wd)/2
        
        histograms_vol_wd = {}
        uv_values_vol_wd = {}
        shap_values_vol_wd = {}
        for structure in structures:
            if x == 'uv':
                x_xtot_vol_wd = self.uv_uvtot_wd_vol[structure]
            elif x == 'k':
                x_xtot_vol_wd = self.k_ktot_wd_vol[structure]
            elif x == 'ens':
                x_xtot_vol_wd = self.ens_enstot_wd_vol[structure]
            histogram,uv_value,shap_value = np.histogram2d(x_xtot_vol_wd,
                                                            self.shap_wd_vol[structure],
                                                            bins=bin_num,
                                                            range=[[xhistmin,xhistmax],
                                                                   [yhistmin,yhistmax]])
            uv_value = uv_value[:-1]+np.diff(uv_value)/2
            shap_value = shap_value[:-1]+np.diff(shap_value)/2
            histograms_vol_wd[structure] = histogram
            uv_values_vol_wd[structure] = uv_value
            shap_values_vol_wd[structure] = shap_value 
        
        min_uv_vol = np.min([uv_value1_vol_wa,
                         uv_value2_vol_wa,
                         uv_value3_vol_wa,
                          uv_value4_vol_wa,
                          uv_value1_vol_wd,
                          uv_value2_vol_wd,
                          uv_value3_vol_wd,
                          uv_value4_vol_wd]\
                         +[uv_values_vol_wa[struc] for struc in structures]\
                         +[uv_values_vol_wd[struc] for struc in structures])
            
        max_uv_vol = np.max([uv_value1_vol_wa,
                          uv_value2_vol_wa,
                          uv_value3_vol_wa,
                          uv_value4_vol_wa,
                          uv_value1_vol_wd,
                          uv_value2_vol_wd,
                          uv_value3_vol_wd,
                          uv_value4_vol_wd]\
                         +[uv_values_vol_wa[struc] for struc in structures]\
                         +[uv_values_vol_wd[struc] for struc in structures])
            
        min_shap_vol = np.min([shap_value1_vol_wa,
                           shap_value2_vol_wa,
                           shap_value3_vol_wa,
                           shap_value4_vol_wa,
                           shap_value1_vol_wd,
                           shap_value2_vol_wd,
                           shap_value3_vol_wd,
                           shap_value4_vol_wd]\
                          +[shap_values_vol_wa[struc] for struc in structures]\
                          +[shap_values_vol_wd[struc] for struc in structures])
        
        max_shap_vol = np.max([shap_value1_vol_wa,
                           shap_value2_vol_wa,
                           shap_value3_vol_wa,
                           shap_value4_vol_wa,
                           shap_value1_vol_wd,
                           shap_value2_vol_wd,
                           shap_value3_vol_wd,
                           shap_value4_vol_wd]\
                          +[shap_values_vol_wa[struc] for struc in structures]\
                          +[shap_values_vol_wd[struc] for struc in structures])
        
        # min_uv_vol = np.min([uv_value1_vol_wa,uv_value2_vol_wa,uv_value3_vol_wa,uv_value4_vol_wa,uv_value1_vol_wd,uv_value2_vol_wd,uv_value3_vol_wd,uv_value4_vol_wd])
        # max_uv_vol = np.max([uv_value1_vol_wa,uv_value2_vol_wa,uv_value3_vol_wa,uv_value4_vol_wa,uv_value1_vol_wd,uv_value2_vol_wd,uv_value3_vol_wd,uv_value4_vol_wd])
        # min_shap_vol = np.min([shap_value1_vol_wa,shap_value2_vol_wa,shap_value3_vol_wa,shap_value4_vol_wa,shap_value1_vol_wd,shap_value2_vol_wd,shap_value3_vol_wd,shap_value4_vol_wd])
        # max_shap_vol = np.max([shap_value1_vol_wa,shap_value2_vol_wa,shap_value3_vol_wa,shap_value4_vol_wa,shap_value1_vol_wd,shap_value2_vol_wd,shap_value3_vol_wd,shap_value4_vol_wd])
        interp_h1_vol_wa = interp2d(uv_value1_vol_wa,shap_value1_vol_wa,histogram1_vol_wa)
        interp_h2_vol_wa = interp2d(uv_value2_vol_wa,shap_value2_vol_wa,histogram2_vol_wa)
        interp_h3_vol_wa = interp2d(uv_value3_vol_wa,shap_value3_vol_wa,histogram3_vol_wa)
        interp_h4_vol_wa = interp2d(uv_value4_vol_wa,shap_value4_vol_wa,histogram4_vol_wa)
        interp_h1_vol_wd = interp2d(uv_value1_vol_wd,shap_value1_vol_wd,histogram1_vol_wd)
        interp_h2_vol_wd = interp2d(uv_value2_vol_wd,shap_value2_vol_wd,histogram2_vol_wd)
        interp_h3_vol_wd = interp2d(uv_value3_vol_wd,shap_value3_vol_wd,histogram3_vol_wd)
        interp_h4_vol_wd = interp2d(uv_value4_vol_wd,shap_value4_vol_wd,histogram4_vol_wd)
        vec_uv_vol = np.linspace(min_uv_vol,max_uv_vol,1000)
        vec_shap_vol = np.linspace(min_shap_vol,max_shap_vol,1000)
        uv_grid_vol,shap_grid_vol = np.meshgrid(vec_uv_vol,vec_shap_vol)
        histogram_Q1_vol_wa = interp_h1_vol_wa(vec_uv_vol,vec_shap_vol)
        histogram_Q2_vol_wa = interp_h2_vol_wa(vec_uv_vol,vec_shap_vol)
        histogram_Q3_vol_wa = interp_h3_vol_wa(vec_uv_vol,vec_shap_vol)
        histogram_Q4_vol_wa = interp_h4_vol_wa(vec_uv_vol,vec_shap_vol)
        histogram_Q1_vol_wd = interp_h1_vol_wd(vec_uv_vol,vec_shap_vol)
        histogram_Q2_vol_wd = interp_h2_vol_wd(vec_uv_vol,vec_shap_vol)
        histogram_Q3_vol_wd = interp_h3_vol_wd(vec_uv_vol,vec_shap_vol)
        histogram_Q4_vol_wd = interp_h4_vol_wd(vec_uv_vol,vec_shap_vol)
        
        if mode == 'cf':
            shap_grid_vol /= 100
        
        histograms_struc_vol_wa = {}
        for structure in structures:
            interp_h = interp2d(uv_values_vol_wa[structure],
                                shap_values_vol_wa[structure],
                                histograms_vol_wa[structure])
            histograms_struc_vol_wa[structure] = interp_h(vec_uv_vol,vec_shap_vol)
            
        histograms_struc_vol_wd = {}
        for structure in structures:
            interp_h = interp2d(uv_values_vol_wd[structure],
                                shap_values_vol_wd[structure],
                                histograms_vol_wd[structure])
            histograms_struc_vol_wd[structure] = interp_h(vec_uv_vol,vec_shap_vol)
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        x0 = 0
        x0b = 0.7
        x1 = 0.08
        x2 = 1.16
        x3 = 1.7
        y0_1 = 2
        y1_1 = 3.9
        y0_2 = 0
        y1_2 = 0.8
        ytop = 4.5
        ytop2 = y1_1
        '''
        plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
                         color=cmap_fill(0.9),alpha=0.1)
        plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
                         [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
                         color=cmap_fill(0.1),alpha=0.1)
        plt.plot([x0,x0],[y0_1,y0_2],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        plt.plot([x2,x3],[ytop2,ytop2],color='k')
        plt.plot([x2,x2],[y1_2,y1_1],color='k')
        plt.plot([x2,x2],[ytop,y1_1],color='k')
        plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        '''
        
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val],colors=[(color11,color12,color13)],linestyles='dashed')
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val],colors=[(color21,color22,color23)],linestyles='dashed')
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val],colors=[(color31,color32,color33)],linestyles='dashed')
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val],colors=[(color41,color42,color43)],linestyles='dashed')
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)],
                        linestyles='dashed')
        '''
        plt.text(0.5, 0.1, 'A', fontsize = 20)
        plt.text(1.5, 2.1, 'B', fontsize = 20)   
        plt.text(0.5, 4, 'C', fontsize = 20) 
        plt.ylim([y0_2,ytop])
        plt.xlim([x0,x3])'''
        
        if mode == 'mse':
            if x == 'ens':
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,2.5])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,5.5])
            elif x == 'k':
                plt.xticks([0.5,1])
                plt.xlim([0,1.25])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4.5])
            elif x == 'uv':
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,1.7])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4.5])
                
        if mode == 'cf':
            if x == 'ens':
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,3])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            elif x == 'k':
                plt.xticks([0.5,1])
                plt.xlim([0,1.5])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            elif x == 'uv':
                plt.xticks([0.5,1])
                plt.xlim([0,1.5])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            
        plt.grid()
            
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
        handles += [mpl.lines.Line2D([0],[0],linestyle='solid',color='k'),
                    mpl.lines.Line2D([0],[0],linestyle='dashed',color='k')]
        labels += ['W-A','W-D']
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'totvol_SHAPvol_'+colormap+str(structures)+'_30+_wall.png')
        
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        
        '''
        plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
                         color=cmap_fill(0.9),alpha=0.1)
        plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
                         [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
                         color=cmap_fill(0.1),alpha=0.1)
        plt.plot([x0,x0],[y0_1,y0_2],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        plt.plot([x2,x3],[ytop2,ytop2],color='k')
        plt.plot([x2,x2],[y1_2,y1_1],color='k')
        plt.plot([x2,x2],[ytop,y1_1],color='k')
        plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        '''
        
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wa[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wa.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wa.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wa.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wa.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wa[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        '''
        plt.text(0.5, 0.1, 'A', fontsize = 20)
        plt.text(1.5, 2.1, 'B', fontsize = 20)   
        plt.text(0.5, 4, 'C', fontsize = 20) 
        plt.ylim([y0_2,ytop])
        plt.xlim([x0,x3])
        '''
            
        if mode == 'mse':
            if x == 'ens':
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,2.5])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,5.5])
            elif x == 'k':
                plt.xticks([0.5,1])
                plt.xlim([0,1.25])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4.5])
            elif x == 'uv':
                plt.xticks([0.5,1,1.5])
                plt.xlim([0,1.7])
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4.5])
                
        if mode == 'cf':
            if x == 'ens':
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,3])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            elif x == 'k':
                plt.xticks([0.5,1])
                plt.xlim([0,1.5])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
            elif x == 'uv':
                plt.xticks([0.5,1])
                plt.xlim([0,1.5])
                plt.yticks([0.5,1])
                plt.ylim([0,1.5])
        
        plt.grid()
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot}V^+)\cdot10^{-7}$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','7'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'totvol_SHAPvol_'+colormap+str(structures)+'_30+_wallattach.png')
        
        
        if mode == 'cf':
            shap_grid_vol *= 10
        uv_grid_vol *= 10
        
        x0 = 0
        x0b = 0.5
        x1 = 0.1
        x2 = 0.3
        x3 = 0.8
        y0_1 = 2.5
        y1_1 = 3.5
        y0_2 = 0
        y1_2 = 0.8
        ytop = 4
        ytop2 = y1_1
        
        fs = 20
        plt.figure()
        cmap_fill = plt.cm.get_cmap('viridis', 10)
        
        '''
        plt.fill_between([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],[y0_2,y0_2,y0_2,y1_2],\
                         color=cmap_fill(0.9),alpha=0.1)
        plt.fill_between([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],\
                         [y1_1,y1_1,y1_1],color=cmap_fill(0.5),alpha=0.1)
        plt.fill_between([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],[ytop2,ytop2],\
                         color=cmap_fill(0.1),alpha=0.1)
        plt.plot([x0,x0],[y0_1,y0_2],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_1,y1_1,y1_1,y1_1],color='k')
        plt.plot([x0,x1,x0b,x2],[y0_2,y0_2,y0_2,y1_2],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,ytop,ytop],color='k')
        plt.plot([x1,(ytop-y1_1)*(x1-x0)/(y1_1-y0_1)+x1,x2],[y1_1,y1_1,y1_1],color='k')
        plt.plot([x2,x3],[y1_2,y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2)],color='k')
        plt.plot([x2,x3],[ytop2,ytop2],color='k')
        plt.plot([x2,x2],[y1_2,y1_1],color='k')
        plt.plot([x2,x2],[ytop,y1_1],color='k')
        plt.plot([x3,x3],[y1_2+(y1_2-y0_2)/(x2-x0b)*(x3-x2),ytop2],color='k')
        '''
        
        # color11 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,0]
        # color12 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,1]
        # color13 = plt.cm.get_cmap(colormap,4+len(structures)).colors[0,2]
        color21 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,0]
        color22 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,1]
        color23 = plt.cm.get_cmap(colormap,2+len(structures)).colors[0,2]
        # color31 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,0]
        # color32 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,1]
        # color33 = plt.cm.get_cmap(colormap,4+len(structures)).colors[2,2]
        color41 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,0]
        color42 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,1]
        color43 = plt.cm.get_cmap(colormap,2+len(structures)).colors[1,2]
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color11,color12,color13)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color21,color22,color23)],alpha=alf)
        # plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color31,color32,color33)],alpha=alf)
        plt.contourf(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val,1e5*lev_val],colors=[(color41,color42,color43)],alpha=alf)
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contourf(uv_grid_vol,
                         shap_grid_vol,
                         histograms_struc_vol_wd[structure].T,
                         levels=[lev_val,1e5*lev_val],
                         colors=[(colorx1,colorx2,colorx3)],
                         alpha=alf)
        
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q1_vol_wd.T,levels=[lev_val],colors=[(color11,color12,color13)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q2_vol_wd.T,levels=[lev_val],colors=[(color21,color22,color23)])
        # plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q3_vol_wd.T,levels=[lev_val],colors=[(color31,color32,color33)])
        plt.contour(uv_grid_vol,shap_grid_vol,histogram_Q4_vol_wd.T,levels=[lev_val],colors=[(color41,color42,color43)])
        
        for ii, structure in enumerate(structures):
            colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
            colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
            colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
            plt.contour(uv_grid_vol,
                        shap_grid_vol,
                        histograms_struc_vol_wd[structure].T,
                        levels=[lev_val],
                        colors=[(colorx1,colorx2,colorx3)])
        '''
        plt.text(0.5, 0.1, 'A', fontsize = 20)
        plt.text(1.5, 2.1, 'B', fontsize = 20)   
        plt.text(0.5, 4, 'C', fontsize = 20) 
        
        
        plt.ylim([y0_2,ytop])
        plt.xlim([x0,x3])
        '''
        
        if mode == 'mse':
            if x == 'ens':
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4.5])
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0.2,2.5])
            elif x == 'k':
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4])
                plt.xticks([2,4,6])
                plt.xlim([0,6])
            elif x == 'uv':
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,4])
                plt.xticks([2,4,6,8])
                plt.xlim([0,10])
            
        elif mode == 'cf':
            if x == 'ens':
                plt.yticks([2,4,6])
                plt.ylim([0,7.5])
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0.2,2.5])
            elif x == 'k':
                plt.yticks([2,4,6])
                plt.ylim([0,7])
                plt.xticks([1,2,3,4,5,6,7,8,9])
                plt.xlim([0,6])
            elif x == 'uv':
                plt.yticks([1,2,3,4,5,6,7,8,9])
                plt.ylim([0,5])
                plt.xticks([2,4,6,8,10])
                plt.xlim([0,11])
        
        plt.grid()
        if x == 'uv':
            plt.xlabel('$\overline{uv}_e/(\overline{uv}_\mathrm{tot}V^+)\cdot10^{-8}$',\
                       fontsize=fs)
        elif x == 'k':
            plt.xlabel('$k_e/(k_\mathrm{tot}V^+)\cdot10^{-8}$',\
                       fontsize=fs)
        elif x == 'ens':
            plt.xlabel('$\Omega_e/(\Omega_\mathrm{tot}V^+)\cdot10^{-8}$',\
                       fontsize=fs)
        if mode == 'mse':
            plt.ylabel(self.ylabel_shap_vol,fontsize=fs)
        elif mode == 'cf':
            plt.ylabel(self.ylabel_shap_vol.replace('9','8'),fontsize=fs)
        plt.tick_params(axis='both', which='major', labelsize=fs)
        # handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color11,color12,color13,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color11,color12,color13,alf)),\
        handles = [mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color21,color22,color23,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color21,color22,color23,alf)),\
                   # mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color31,color32,color33,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color31,color32,color33,alf)),\
                   mpl.lines.Line2D([0],[0],marker='o',markeredgecolor=(color41,color42,color43,1),markersize=15, ls='',markeredgewidth=1,markerfacecolor=(color41,color42,color43,alf))]
        # labels = ['Outward\ninteractions','Ejections','Inward\ninteractions','Sweeps']
        labels = ['Ejections','Sweeps']
        
        for ii, structure in enumerate(structures):
            if structure == 'streak':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Streaks')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
                
            elif structure == 'chong':
                colorx1 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,0]
                colorx2 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,1]
                colorx3 = plt.cm.get_cmap(colormap,2+len(structures)).colors[ii+2,2]
                labels.append('Vortices\n (Chong)')
                handles.append(mpl.lines.Line2D([0],
                                                [0],
                                                marker='o',
                                                markeredgecolor=(colorx1,colorx2,colorx3,1),
                                                markersize=15, 
                                                ls='',markeredgewidth=1,
                                                markerfacecolor=(colorx1,colorx2,colorx3,alf)))
        
        plt.legend(handles,labels,fontsize=fs-4,loc='center left', bbox_to_anchor=(1, 0.5))
        plt.tight_layout()
        plt.savefig('hist2d_interp_'+x+x+'totvol_SHAPvol_'+colormap+str(structures)+'_30+_walldettach.png')
        