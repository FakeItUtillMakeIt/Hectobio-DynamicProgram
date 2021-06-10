
import numpy as np
from dtw import *
import matplotlib.pyplot as plt
import argparse
import os
import h5py
import json
import scipy.io as sio



def dtw_find_seq(path_A=None, path_B=None):
    # 动态规划对齐两个序列
    # path_A='ATCGGGGGTACACCTTT'
    # path_B='AAAAAAAATTTTTCGGGGGGGGGGGGTAAAACCCA'

    mat_path = np.zeros([len(path_B), len(path_A)])

    print('path_A len:{}'.format(len(path_A)))
    sum_loss = 0
    B_start = 0
    A_corr = 0
    for A_inx in range(len(path_A)):
        A = path_A[A_inx]
        B_corr = B_start
        for B_inx in range(B_corr, len(path_B)):
            B = path_B[B_inx]
            if np.abs(ord(B)-ord(A)) < 1:
                sum_loss += np.abs(ord(B)-ord(A))

                mat_path[B_inx][A_inx] = 1

                B_start = B_inx

                try:
                    if path_A[A_inx+1] == path_B[B_inx+1]:
                        A_corr += 1
                        print('next equal {}:{}'.format(B_inx+1, A_inx+1))
                        break

                    else:
                        pass
                except:
                    pass
            else:
                pass
        if B_inx+1 == len(path_B) and path_A[A_inx+1] != path_B[B_inx]:
            break
    print(A_corr)
    # print(mat_path)
    if A_corr >= len(path_A)//2:
        # A，B具有相关性，A可视为B的子序列
        return (A_corr+1)/len(path_A), True

    if sum_loss < 1:
        return mat_path, True
    else:
        return mat_path, False

# 动态规划寻找两个相似序列，相似率大于阈值视为相似
# 该函数使用递归寻找,调用对其函数
# 仅适用于无跳过，且开始序列相同


def dtw_find_ralative_seq(path_A, path_B):
    ret = dtw_find_seq(path_A, path_B)
    print(ret)
    pass


def test1():
    path_A = 'ATCGGGGGTACACCTT'
    path_B = 'ATCGGGG'

    dtw_find_ralative_seq(path_A, path_B)


# 序列信号动态时间规整

path_A = 'GGTACACC'
path_B = 'ATCGGGGGTACACCTT'


def dynamic_find_sub_seq(path_A, path_B):
    loss_mat = np.zeros((len(path_A)+1, len(path_B)+1))

    for i in range(1, len(path_A)+1):
        for j in range(1, len(path_B)+1):
            mki = loss_mat[i-1][j]
            mkj = loss_mat[i][j-1]

            if path_A[i-1] == path_B[j-1]:
                mij = loss_mat[i-1][j-1]+1
            else:
                mij = loss_mat[i-1][j-1]-1

            loss_mat[i][j] = max(mki, mkj, mij, 0)
    return loss_mat


def mat_trace_back(ref_seq, find_seq, loss_mat):
    x, y = np.where(np.max(loss_mat) == loss_mat)
    print(x[0], y[0])


def test2():
    loss_mat = dynamic_find_sub_seq(path_A, path_B)
    mat_trace_back(path_A, path_B, loss_mat)


# 电平信号动态时间扭曲,信号
def dynamic_find_pass(ref_seq, find_seq):
    INF = 100000000
    loss_mat = np.zeros([len(ref_seq), len(find_seq)])*INF
    # 花费矩阵，cost
    for row in range(len(ref_seq)):
        for column in range(len(find_seq)):
            current_level_sub = np.abs(ref_seq[row]-find_seq[column])
            if row == 0 and column == 0:
                loss_mat[row][column] = current_level_sub
            elif row == 0 and column != 0:
                loss_mat[row][column] = current_level_sub + \
                    loss_mat[row][column-1]
            elif row != 0 and column == 0:
                loss_mat[row][column] = current_level_sub + \
                    loss_mat[row-1][column]
            else:
                loss_mat[row][column] = current_level_sub+min(
                    loss_mat[row-1][column-1], loss_mat[row-1][column], loss_mat[row][column-1])

    #print(loss_mat)
    return loss_mat


def trace_back_pass_path(ref_seq, find_seq, loss_mat):
    # 回溯找路径
    # 最右下角即为最小距离，根据它反推途径
    INF = 1000000000000
    
    path_min = []
    min_cost = []

    row = len(ref_seq)-1
    column = len(find_seq)-1
    min_cost.append(loss_mat[row][column])
    # path_min.append((row,column))
    path_min = [[row, column]]+path_min
    while(row != 0 or column != 0):
        Mij = loss_mat[row-1][column-1]
        Mki = loss_mat[row-1][column]
        Mkj = loss_mat[row][column-1]
        if min(Mij, Mki, Mkj) == Mij:
            # min_cost=Mij
            min_cost.append(Mij)
            # path_min.append((row-1,column-1))
            path_min = [[row-1, column-1]]+path_min
            row = row-1
            column = column-1
        elif min(Mij, Mki, Mkj) == Mki:
            # min_cost=Mki
            min_cost.append(Mki)
            # path_min.append((row-1,column))
            path_min = [[row-1, column]]+path_min
            row = row-1
            column = column
        elif min(Mij, Mki, Mkj) == Mkj:
            # min_cost=Mkj
            min_cost.append(Mkj)
            # path_min.append((row,column-1))
            path_min = [[row, column-1]]+path_min
            row = row
            column = column-1
        else:
            # min_cost=min(Mij,Mki,Mkj)
            min_cost.append(Mij)
            # path_min.append(((row-1,column-1),(row-1,column),(row,column-1)))
            path_min = [[row-1, column-1]]+path_min
            row = row-1
            column = column-1
        if column==0:
            break

    # print(min_cost)
    # print(path_min)
    # ref
    x = np.array(path_min)[:, 1]
    # find
    y = np.array(path_min)[:, 0]
    # print(x,y)

    return path_min,x,y


#根据最小路径对齐find和ref序列
def align_ref_find(ref_seq, find_seq, path_min):
    align_ref = []
    align_find = []
    for ref_inx, find_inx in path_min:
        align_ref.append(ref_seq[ref_inx])
        align_find.append(find_seq[find_inx])
    print('ref原始信号长度：{},find原始信号长度：{}'.format(len(ref_seq), len(find_seq)))
    print('ref_align长度：{},find_align长度：{}'.format(
        len(align_ref), len(align_find)))

    return align_ref,align_find
    pass

def get_region_signal():
    read = h5py.File('/mnt/raid/lla/hectobio/basecalling/read_1_section_1.mat','r')
    x_f = read['read/x_f'][()].T
    region_signal=x_f[0]
    read.close()
    return region_signal

def get_reference_signal():

    return

def get_selfalign_signal():
    read=sio.loadmat('/mnt/raid/lijin/Hectobio/testcode/DynamicPrograming/self_align.mat')
    x_tf=read['read']['x_tf']
    self_align_signal=x_tf[0][0][0]
    
    return self_align_signal





if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='dynamic programing for sequence or signal')
    parser.add_argument('--ref-seq', help='reference sequence or signal')
    parser.add_argument('--find-seq', help='perparing find sequence or signal')

    args = parser.parse_args()


    #使用原始信号和自对齐信号进行规整
    ref_seq=get_selfalign_signal()
    find_seq=get_region_signal()

    loss_mat=dynamic_find_pass(ref_seq, find_seq)
    
    print('cost:{}'.format(loss_mat[-1][-1]))

    path_pass,x,y = trace_back_pass_path(
        ref_seq, find_seq, loss_mat)

    align_ref_find(ref_seq, find_seq, path_pass)
    pass
