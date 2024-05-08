import h5py 
import numpy as np
from tqdm import tqdm



path_grad = '/media/nils/Elements/P125_21pi_vu/grad/P125_21pi_vu'
path_cf = '/media/nils/Elements/P125_21pi_vu_cf/P125_21pi_vu'
# path_grad = './P125_21pi_vu/grad/P125_21pi_vu'
# path_cf = './P125_21pi_vu_cf/P125_21pi_vu'

start = 1000
end = 9999
step = 1

ny = 0.0004761904761904762
U_bulk = 0.8988189365467902

c_f_var = np.zeros((1, int(np.floor((end-start)/step)+1)))

for ii in tqdm(range(start, end, step)):
    # file = h5py.File(path_grad+f'.{ii}.grad', 'r+')
    # dudy = np.array(file['dudy'])
    # file.close()
    file2 = h5py.File(path_cf+f'.{ii}.cf', 'r+')
    # dudy_wall = np.zeros((2,dudy.shape[1],dudy.shape[2]))
    # dudy_wall[0,:,:] = dudy[0,:,:]
    # dudy_wall[1,:,:] = -dudy[-1,:,:]
    
    # c_f = 2*ny*dudy_wall/(U_bulk**2)
    # file2.create_dataset('c_f', data=c_f)
    # file2.close()
    
    c_f = np.array(file2['c_f'])
    # if 'c_f_mean' in file2:
    #     del file2['c_f_mean']
    # file2.create_dataset('c_f_mean', data=np.mean(c_f))
    c_f_var[0, int(np.floor((ii-start)/step))] = np.var(c_f)
    file2.close()
    
    print(np.sqrt(np.mean(c_f_var)))

    