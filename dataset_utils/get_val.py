import shutil
import os
import random
import numpy as np




if __name__ == "__main__":


    src_dir1 = './train/'

    dst_dir1 = './val/'

    os.makedirs(dst_dir1, exist_ok=True)

    file_list = os.listdir(src_dir1)

    total_num = len(file_list)
    val_num = int(total_num*0.1)
    idxs = random.sample(range(total_num), val_num)
    for id in idxs:
        file = file_list[id]

        src_path1 = os.path.join(src_dir1, file)
        dst_path1 = os.path.join(dst_dir1, file)

        shutil.move(src_path1, dst_path1)
