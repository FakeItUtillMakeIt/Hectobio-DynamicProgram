####
#时序数据中的闪烁点（也就是明显与数据不连续的孤立部分）
from typing import Tuple
import numpy as np
from numpy.core.defchararray import index
from numpy.core.fromnumeric import repeat

#data=level data
# data=np.array(np.random.random(size=100))


def filterFlicker(data,first_pf_thres=2.0,second_pf_thres=2.0,first_step=2,second_step=5):
    pfs=np.array([first_pf_thres,second_pf_thres])
    windows=np.array([first_step,second_step])

    bad_data=True
    w=windows[0]
    pf=pfs[0]
    filter_data=data

    new_index=[]
    ii=np.arange(len(data))
    while np.any(bad_data):
        #对每个数据点计算周围值中值和标准差，判断是否为闪烁状态
        #tmp1=np.array([-w,-1,1,w])
        tmp1=np.concatenate((np.arange(-w,-1+1),np.arange(1,w+1)))
        tmp2=len(filter_data)-2*w
        tmp3=np.array([item for item in range(w,len(filter_data)-w)])
        tmp3=np.transpose(tmp3)

        tmp1=np.expand_dims(tmp1,axis=0)
        tmp4=repeat(tmp1,tmp2,axis=0)
        tmp3=np.expand_dims(tmp3,axis=1)
        tmp5=repeat(tmp3,2*w,axis=1)
        
        ix=tmp4+tmp5

        #计算每个点周围2个点的中值和标准差，作为判断是否为孤立状态的评判依据
        arr=np.arange(2*w)
        m_ix=[]
        s_ix=[]
        for vpack in ix:
            tmp_med=[]
            tmp_std=[]
            for i in range(2*w):
                tmp_med.append(filter_data[vpack[i]])
                tmp_std.append(filter_data[vpack[i]])
            m_ix.append(tmp_med)
            s_ix.append(tmp_std)

        #m_ix=[[filter_data[idx1],filter_data[idx2],filter_data[idx3],filter_data[idx4]] for idx1,idx2,idx3,idx4 in ix]
        data_med=np.median(m_ix,axis=1)
        #s_ix=[[filter_data[idx1],filter_data[idx2],filter_data[idx3],filter_data[idx4]] for idx1,idx2,idx3,idx4 in ix]
        data_std=np.std(s_ix,axis=1)

        a=filter_data[w:-w]
        a=np.transpose(a)
        data_devition=(a-data_med)/data_std
        data_devition=np.abs(data_devition)

        

        #查找超出阈值的数据
        # bad_data1=[]
        # for value in data_devition:
        #     if value<pf:
        #         bad_data1.append(False)
        #     else:
        #         bad_data1.append(True)
        # bad_data=np.array(bad_data1)
        bad_data=np.array(data_devition>pf)
        bad_data=data_std>0.04
        #删除超出阈值的数据索引
        ii=np.array([ii[idx] for idx,v in enumerate(bad_data) if v==False])
        filter_data=np.array([filter_data[idx] for idx,v in enumerate(bad_data) if v==False])
        # new_index=[]
        # for idx,v in enumerate(bad_data):
        #     if v==True:
        #         pass
        #     else:
        #         new_index.append(idx)

        break    
        pass
        #ix=np.repeat()
    new_index=[]
    for idx,v in enumerate(bad_data):
        if v==True:
            pass
        else:
            new_index.append(idx)

    bad_data=True
    w=windows[1]
    pf=pfs[1]


    while np.any(bad_data):
        #对每个数据点计算周围值中值和标准差，判断是否为闪烁状态
        #tmp1=np.array([-w,-1,1,w])
        tmp1=np.concatenate((np.arange(-w,-1+1),np.arange(1,w+1)))
        tmp2=len(filter_data)-2*w
        tmp3=np.array([item for item in range(w,len(filter_data)-w)])
        tmp3=np.transpose(tmp3)

        tmp1=np.expand_dims(tmp1,axis=0)
        tmp4=repeat(tmp1,tmp2,axis=0)
        tmp3=np.expand_dims(tmp3,axis=1)
        tmp5=repeat(tmp3,2*w,axis=1)
        
        ix=tmp4+tmp5

        #计算每个点周围5个点的中值和标准差，作为判断是否为孤立状态的评判依据
        arr=np.arange(2*w)
        m_ix=[]
        s_ix=[]
        for vpack in ix:
            tmp_med=[]
            tmp_std=[]
            for i in range(2*w):
                tmp_med.append(filter_data[vpack[i]])
                tmp_std.append(filter_data[vpack[i]])
            m_ix.append(tmp_med)
            s_ix.append(tmp_std)
        #m_ix=[[filter_data[idx1],filter_data[idx2],filter_data[idx3],filter_data[idx4]] for idx1,idx2,idx3,idx4 in ix]
        data_med=np.median(m_ix,axis=1)
        #s_ix=[[filter_data[idx1],filter_data[idx2],filter_data[idx3],filter_data[idx4]] for idx1,idx2,idx3,idx4 in ix]
        data_std=np.std(s_ix,axis=1)

        a=filter_data[w:-w]
        a=np.transpose(a)
        data_devition=(a-data_med)/data_std
        data_devition=np.abs(data_devition)
        #查找超出阈值的数据
        # bad_data2=[]
        # for value in data_devition:
        #     if value<pf:
        #         bad_data2.append(False)
        #     else:
        #         bad_data2.append(True)
        # bad_data=np.array(bad_data2)
        bad_data=np.array(data_devition>pf)
        #删除超出阈值的数据索引
        ii=np.array([ii[idx] for idx,v in enumerate(bad_data) if v==False])
        filter_data=np.array([filter_data[idx] for idx,v in enumerate(bad_data) if v==False])
        # new_index=[]
        # for idx,v in enumerate(bad_data):
        #     if v==True:
        #         pass
        #     else:
        #         new_index.append(idx)

        break
        pass
    print(ii)
    #返回删除闪烁状态的索引
    return ii

    #标记删除的点为坏点
    # bad_points=np.ones(shape=(1,len(data)))*True
    # for idx,value in enumerate( new_index):
    #     if value==True:
    #         bad_points[value]=False

    # print(bad_points)













