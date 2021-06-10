#实际电信号和参考电信号进行对齐
#Increasing DNA 论文中第6节和matlab源码中的processCV-->sequenceCV-->calculateSequenceVV。
import math
import json
import os 
import sys
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.defchararray import find, partition
align_module_path=os.path.abspath("../../findleveling/C_python")

sys.path.append(align_module_path)
import align
import self_alignment
# module_path=os.path.abspath('../..')
# sys.path.append(module_path)
import dynamic_wrap
import flickerfilter
import calc_ref_act_scoremat

abs_dir_path=os.path.abspath('.')

json_path=os.path.join(abs_dir_path,'x1k1x2k2.json')

json_dict=defaultdict()
with open(json_path,'r') as json_f:
    json_dict=json.load(json_f)
    pass

print(json_dict.keys())

act_level=json_dict['X1']
act_conf=json_dict['K1']
ref_level=json_dict['X2']
ref_conf=json_dict['K2']


act_level=np.array(act_level)
act_conf=np.array(act_conf)
ref_level=np.array(ref_level)
ref_conf=np.array(ref_conf)

region_ref=ref_level
region_act=act_level
#删除闪烁状态
########################################
########################################
act_index=flickerfilter.filterFlicker(act_level,first_pf_thres=3.0,second_pf_thres=8.0,first_step=2,second_step=4)
act_level=np.array([act_level[idx] for idx in act_index])
act_conf=np.array([act_conf[idx] for idx in act_index])

#利用matlab中的方法计算得分矩阵
#shape=act.shape x ref.shape  193x182
score_mat=calc_ref_act_scoremat.calc_scoremat(act_level,act_conf,ref_level,ref_conf)   
score_cutoff=-3
#
p_step,p_skip,p_ext,p_bad=0.8,0.3,0.3,0.2

p_list=[p_skip*np.power(p_ext,zhishu) for zhishu in range(11)]
p=np.array([p_step]+p_list)

p_ind_given_backstep=0.975
p_backstep=0.025
did_backstep=np.zeros(shape=act_level.shape)
did_nobackstep=np.ones(shape=act_level.shape)

p_ind_given_nobackstep=0.5-(p_ind_given_backstep*p_backstep)
prior_given_backstep=np.ones(shape=ref_level.shape)*0.5
prior_given_nobackstep=np.ones(shape=ref_level.shape)*0.5

did_backstep=did_backstep.reshape(did_backstep.shape[0],1)
did_nobackstep=did_nobackstep.reshape(did_nobackstep.shape[0],1)
prior_given_backstep=prior_given_backstep.reshape(1,prior_given_backstep.shape[0])
prior_given_nobackstep=prior_given_nobackstep.reshape(1,prior_given_nobackstep.shape[0])

alignment_matrix=score_mat+np.dot(did_backstep,prior_given_backstep)+\
    np.dot(did_nobackstep,prior_given_nobackstep)

#校准对齐矩阵
tmp=np.max(alignment_matrix,axis=1)+score_cutoff
tmp.reshape(tmp.shape[0],1)
candidate_matches=np.zeros(shape=alignment_matrix.shape)
#p为实际序列中每个电平的所代表的step|skip|等概率
#p: act.shape[0]x4x12
p=p.reshape(1,1,12)

P=np.repeat(p,4,axis=1)
P=np.repeat(P,act_level.shape[0],axis=0)
print(P.shape)
Pbad=p_bad*np.ones(shape=(1,act_level.shape[0]))

#对应Hel308 poremodel CV 的属性
###############################################
#转换信息的结构（哪些映射状态可以转换为哪些其他映射状态，以及与哪些相关的惩罚）
transition_info=None
##############################################

p_vals=None
total_probability=None
if P.shape[2]==3:
    pass
elif P.shape[2]==12:
    p_vals=P
    for cB in range(p_vals.shape[1]):
        total_probability=np.sum(p_vals[:,cB,:],axis=1)
        total_probability=total_probability.reshape(total_probability.shape[0],1)
        total_probability=np.repeat(total_probability,12,axis=1)
        total_probability.reshape(total_probability.shape[0],1,total_probability.shape[1])
        p_vals[:,cB,:]=p_vals[:,cB,:]/total_probability
    pass
############################################
#p_vals=np.log(np.reshape(np.sum(np.dot(p_vals,np.reshape(transition_info.p_combos,shape=(2,3,1,0),axis=2),shape=(0,1,3,2)))))
############################################
#find the penalty to be applied for each bad level
p_good=np.log(1-Pbad)
p_bad=np.log(Pbad)
max_sequential_bad=p.shape[1]-1

#舍去小于零的元素，以-inf填充，说明这个点的索引不可达
for axis in range(tmp.shape[0]):
    candidate_matches[axis][:]=alignment_matrix[axis]>tmp[axis]

num_candidate_matches=np.sum(candidate_matches,axis=1)


for xaxis in range(candidate_matches.shape[0]):
    for yaxis in range(candidate_matches.shape[1]):
        if candidate_matches[xaxis][yaxis]==0:
            # alignment_matrix[xaxis][yaxis]=-np.inf
            alignment_matrix[xaxis][yaxis]=0
        if alignment_matrix[xaxis][yaxis]<0:
            alignment_matrix[xaxis][yaxis]=0

#原信号计算损失矩阵
loss_mat=dynamic_wrap.dynamic_find_pass(ref_seq=ref_level,find_seq=act_level)
pass_path,path_x,path_y=dynamic_wrap.trace_back_pass_path(ref_seq=ref_level,find_seq=act_level,loss_mat=loss_mat)
print(loss_mat.shape)
print(alignment_matrix.shape)

#alignment_matrix=alignment_matrix+loss_mat.reshape(alignment_matrix.shape)


#原信号根据matlab中的scorematrix计算得分矩阵
path_min,pass_x,pass_y=dynamic_wrap.trace_back_pass_path(ref_seq=act_level,find_seq=ref_level,loss_mat=-alignment_matrix)
print(path_min)
align_act,align_ref=dynamic_wrap.align_ref_find(ref_seq=act_level,find_seq=ref_level,path_min=path_min)

plt.figure(1)

# plt.plot(np.array(act_level)+0.2,color='red',linewidth=0.5)
# plt.plot(np.array(ref_level)+0.3,color='blue',linewidth=0.5)

# plt.plot(np.array(align_act)+0.1,color='red',linewidth=0.5,alpha=0.9)
# plt.plot(np.array(align_ref),color='blue',linewidth=0.5,alpha=0.9)

plt.step(np.arange(len(region_ref)),np.array(region_ref)+0.75,color='blue',linewidth=0.5,label='region_ref')
plt.step(np.arange(len(region_act)),np.array(region_act)+0.6,color='red',linewidth=0.5,label='region_act')

plt.step(np.arange(len(ref_level)),np.array(ref_level)+0.45,color='blue',linewidth=0.5,label='filterfilcker_ref')
plt.step(np.arange(len(act_level)),np.array(act_level)+0.3,color='red',linewidth=0.5,label='filterflicker_act')

plt.step(np.arange(len(align_ref)),np.array(align_ref),color='blue',linewidth=0.5,alpha=0.9,label='levelalign_ref')
plt.step(np.arange(len(align_act)),np.array(align_act)+0.15,color='red',linewidth=0.5,alpha=0.9,label='levelalign_act')

plt.legend()
plt.xlabel('level')
plt.ylabel('current/pA')

savename='alignmentCV1.png'
plt.savefig(os.path.join(abs_dir_path,savename),dpi=300)
plt.close()

#####################################
#计算皮尔森相关度
####################################
def pearson(vector1, vector2):
    n = len(vector1)
    #simple sums
    sum1 = sum(float(vector1[i]) for i in range(n))
    sum2 = sum(float(vector2[i]) for i in range(n))
    #sum up the squares
    sum1_pow = sum([pow(v, 2.0) for v in vector1])
    sum2_pow = sum([pow(v, 2.0) for v in vector2])
    #sum up the products
    p_sum = sum([vector1[i]*vector2[i] for i in range(n)])
    #分子num，分母den
    num = p_sum - (sum1*sum2/n)
    den = math.sqrt((sum1_pow-pow(sum1, 2)/n)*(sum2_pow-pow(sum2, 2)/n))
    if den == 0:
        return 0.0
    return num/den
convar_ref_act=pearson(align_ref,align_act)
region_ref_act=pearson(region_ref,region_act)
print('region convriance is {} % \npearson convriance is {} %'.format(region_ref_act*100,convar_ref_act*100))



def trace_process(alignment_matrix):

    traceback_matrix_rows=np.zeros(alignment_matrix.shape)
    traceback_matrix_cols=np.zeros(alignment_matrix.shape)

    #计算转移矩阵
    transition_submatrix=None
    for cl in range(1,alignment_matrix.shape[0]):
        this_row_best_score=np.ones(shape=(1,int(num_candidate_matches[(cl)])))*(-np.inf)
        this_row_best_row_from=np.zeros(shape=(1,int(num_candidate_matches[cl])))
        this_row_best_col_from=np.zeros(shape=(1,int(num_candidate_matches[cl])))
        #对每个row计算转移矩阵，包括badlevel惩罚 对应matlab calculateSequenceVV中209行
        for cfrom in range(np.min(max_sequential_bad,cl-1)):
            transition_submatrix=np.reshape(p_vals[cl,cfrom,transition_info.perm_matrix[candidate_matches[cl-cfrom,:],candidate_matches[cl,:]]]+\
                np.sum(p_bad[(cl-cfrom):cl]),num_candidate_matches[cl-cfrom],num_candidate_matches[cl],p_good[cl-cfrom])
            #为每个to找到最好的from
            from_score,where_from=np.max(np.transpose(transition_submatrix+alignment_matrix[cl-cfrom,candidate_matches[cl-cfrom,:]]),axis=0)
            #找到比目前最好元素更好的to元素
            better_score=from_score>this_row_best_score
            #获取非0元素索引和值
            from_indices=find(candidate_matches[cl-cfrom,:])
            #更新最好的序列路径
            this_row_best_col_from[better_score]=from_indices[where_from[better_score]]
            this_row_best_row_from[better_score]=cl-cfrom
            this_row_best_score[better_score]=from_score[better_score]
        #使用最佳选项更新alignment和traceback矩阵
        alignment_matrix[cl,:]=this_row_best_score+alignment_matrix[cl,candidate_matches[cl,:]]
        traceback_matrix_rows[cl,candidate_matches[cl,:]]=this_row_best_row_from
        traceback_matrix_cols[cl,candidate_matches[cl,:]]=this_row_best_col_from

    #找到开始点，相当于dtw中的最右下角的点
    total_score,kmer=np.max(alignment_matrix[-1,:])
    #???为什么要log
    alignment_matrix=alignment_matrix-np.log(alignment_matrix)


