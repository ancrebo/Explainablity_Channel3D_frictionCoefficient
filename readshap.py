# -*- coding: utf-8 -*-
"""
Created on Tue Jun 27 10:30:46 2023

@author: andres cremades botella

function for calculating the shap values
"""

import shap_config as sc
start = 7000 #1000#1200 7000 #
end = 9998 #9999 #1201 7970 #
step = 1
shap = sc.shap_conf()
shap.read_data(start,end,step,\
               file='/data2/cremades/P125_21pi_vu_SHAP_ann4_divide/P125_21pi_vu',
               fileQ='/data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu',
               fileuvw='/data/cremades/P125_21pi_vu/P125_21pi_vu',
               absolute=True,readdata=True,fileread='data_plots_wall.h5.Q')

shap.read_data_simple(start,
                      end,
                      step,
                      dir_shap='/media/nils/Elements/',
                      dir_struc='/data2/nils/',
                      dir_uvw='/data/cremades/',
                      dir_grad='/media/nils/Elements/',
                      structure='streak',
                      absolute=True,
                      readdata=True)

shap.read_data_simple(start,
                      end,
                      step,
                      dir_shap='/media/nils/Elements/',
                      dir_struc='/data2/nils/',
                      dir_uvw='/data/cremades/',
                      dir_grad='/media/nils/Elements/',
                      structure='chong',
                      absolute=True,
                      readdata=True)

shap.read_data_simple(start,
                      end,
                      step,
                      dir_shap='/media/nils/Elements/',
                      dir_struc='/data2/nils/',
                      dir_uvw='/data/cremades/',
                      dir_grad='/media/nils/Elements/',
                      structure='streak_high_vel',
                      absolute=True,
                      readdata=True)

# shap.plot_shaps(colormap='viridis')#
shap.plot_shaps_pdf(colormap='custom',bin_num=70,lev_val=2,structures=['streak', 'chong', 'streak_high_vel']) #lev_val=5.1
# shap.plot_shaps_pdf_probability(colormap='viridis',bin_num=70,lev_val=5.1)
# shap.plot_shaps_pdf_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
# shap.plot_shaps_pdf_wall_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
# shap.plot_shaps_pdf_wall(colormap='viridis',bin_num=70,lev_val=2, structures=['streak', 'chong'])
# shap.plot_shaps_uv(colormap='viridis')
# shap.plot_shaps_uv_ejectionover(colormap='viridis')
# shap.plot_shaps_uv_pdf(colormap='viridis',bin_num=70,lev_val=2, structures=['streak','chong'], switch='uu') #lev_val=5.1
# shap.plot_shaps_k_pdf(colormap='viridis', bin_num=70,lev_val=2, structures=['streak','chong'])
shap.plot_shaps_ens_pdf(colormap='custom',bin_num=70,lev_val=2,structures=['streak','chong', 'streak_high_vel'])
# shap.plot_shaps_uv_pdf_probability(colormap='viridis',bin_num=70,lev_val=5.1)
# shap.plot_shaps_uv_pdf_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
# shap.plot_shaps_x_pdf_wall(colormap='viridis',bin_num=70,lev_val=2, x='uv', structures=['streak','chong'])
# shap.plot_shaps_x_pdf_wall(colormap='viridis',bin_num=70,lev_val=2, x='k', structures=['streak','chong'])
# shap.plot_shaps_x_pdf_wall(colormap='viridis',bin_num=70,lev_val=2, x='ens', structures=['streak','chong'])
# shap.plot_shaps_uv_pdf_wall_perclim(colormap='viridis',bin_num=70,per_val=0.1,per_val_vol=0.1,log=True)
# shap.plot_shaps_uv_pdf_hist(colormap='viridis')
# shap.plot_shaps_kde(colormap='viridis')
# shap.plot_shaps_AR(colormap='viridis')
# shap.plot_shaps_total(colormap='viridis')
# shap.plot_shaps_total_noback(colormap='viridis')

#'../../../data2/cremades/P125_21pi_vu_SHAP_ann4_divide/P125_21pi_vu',\
#'../../../data2/cremades/P125_21pi_vu_Q_divide/P125_21pi_vu',\
