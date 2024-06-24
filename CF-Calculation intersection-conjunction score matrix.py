# 计算两个节目的观众交集/并集。输入数据为每个节目的观众列表。由于矩阵较大，采用多进程的方式计算
import pandas as pd
import numpy as np
import pickle
import multiprocessing
from tqdm import tqdm

def init_pool(weight):
    global glob_weight    # 子进程初始化，定义全局变量
    glob_weight = weight

def subprocess_compute_weight(i,j, length, twod_list):
    comm_viewer_a = twod_list[i]
    for k in range(104*j, min(104*(j+1),length)):
        if i<k:
            comm_viewer_b = twod_list[k]
            glob_weight [i*length+k] = len(comm_viewer_a.intersection(comm_viewer_b))/len(comm_viewer_a.union(comm_viewer_b))   # i,k为weight的行列

def main(path, worker_num):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    length = len(data)
    twod_list = multiprocessing.Manager().list(data)
    del data

    weight = multiprocessing.Array('f', length*length)
    pool1 = multiprocessing.Pool(worker_num, initializer=init_pool, initargs=(weight,))

    pbar = tqdm(total=length*30)
    pbar.set_description('Process')
    update = lambda *args: pbar.update()
    for i in range(length):
        for j in range(30):
            pool1.apply_async(subprocess_compute_weight, args=(i,j,length, twod_list,),callback=update)
    pool1.close()
    pool1.join()
    del pool1
    del twod_list

    print("所有多进程已完成")
    weight_shared = np.frombuffer(weight.get_obj(), dtype=np.float32).reshape((length, length))
    print("weight读出来了")
    np.save('矩阵.npy',weight_shared)
    print("矩阵已存完")

if __name__ == "__main__":
    main('矩阵.pkl', worker_num=40)