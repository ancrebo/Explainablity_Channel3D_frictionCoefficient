import h5py
import os

folder = '/data2/nils/P125_21pi_vu'

missing = {}

for structure in ['hunt', 'chong', 'streak']:
    path = folder+f'_{structure}/'
    num_files = len([name for name in os.listdir(path) if os.path.isfile(os.path.join(path, name))])
    files_missing = []
    for ii in range(1000,num_files+1050):
        if not os.path.isfile(path+f'P125_21pi_vu.{ii}.h5.{structure}'):
            files_missing.append(ii)
    missing[structure] = files_missing
    
    
print(missing)
        
        

