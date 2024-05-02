import h5py 
import numpy as np
from tqdm import tqdm


path = '/media/nils/Elements/P125_21pi_vu/grad/P125_21pi_vu'

start = 1000
end = 9999
step = 1

for ii in tqdm(range(start, end, step)):
    file = h5py.File(path+f'.{ii}.grad', 'r+')
    dudy = np.array(file['dudy'])
    file.close()
    file2 = h5py.File(path+f'.{ii}.dudy', 'w')
    dudy_wall = np.zeros((2,dudy.shape[1],dudy.shape[2]))
    dudy_wall[0,:,:] = dudy[0,:,:]
    dudy_wall[1,:,:] = dudy[-1,:,:]
    file2.create_dataset('dudy', data=dudy_wall)
    file2.close()
    
