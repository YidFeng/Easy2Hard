import os
import random
import shutil


output_dir = './oversmoothed'
os.makedirs(output_dir, exist_ok=True)
dirs = os.listdir('./results/')

for file in os.listdir('./results/'+dirs[0]):
    rand_idx = random.randint(0,len(dirs)-1)
    mom_dir = dirs[rand_idx]
    src_path = os.path.join('./results',mom_dir,file)
    dst_path = os.path.join(output_dir,file)
    shutil.copy(src_path, dst_path)
