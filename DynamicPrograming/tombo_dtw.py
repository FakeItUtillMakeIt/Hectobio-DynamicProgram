import matplotlib.pyplot as plt 
import numpy as np
from ont_fast5_api.fast5_interface import get_fast5_file
import ruptures as rpt
from dtw import *
import math


file='/mnt/raid5/lijin/Hectobio/testcode/classfy_barcode/fast5/fast5_train/1/1aa2dd4b-2779-4a84-ba65-c6e5db6a16db.fast5'
id_l=[]
f5=get_fast5_file(file, mode='r')
count=0
for read in f5.get_reads():
    count+=1
    print(read.read_id)
    if count>=5:
        break
# print(dir(read))
data=read.get_analysis_dataset('RawGenomeCorrected_000','BaseCalled_template')
offset=dict(list(data['Events'].attrs.items()))['read_start_rel_to_raw']
value=data['Events'].value


template_file='/mnt/raid5/lijin/Hectobio/testcode/classfy_barcode/fast5/fast5_train/1/Bacteria.txt'
fid=open(template_file,'r')
template=[]
level_l=[]
dur_l=[]
for line in fid:
    level_l.append(float(line.split('\t')[0]))
    dur_l.append(math.floor(float(line.split('\t')[1].split('\n')[0])))
temp_l=[]
for num,index in enumerate(level_l):
    temp_l+=[index]*dur_l[num]
raw_data=f5.get_read(read.read_id)
raw_data=raw_data.get_raw_data(read.read_id)

#画台阶
algo=rpt.KernelCPD(kernel='rbf',min_size=4).fit(raw_data)
changpoints_list=algo.predict(pen=5)
segments=np.split(raw_data,changpoints_list)[:-1]
#取每一段台阶的中位数
segment_m=[np.median(s) for s in segments]
print(len(segment_m))
plt.step(changpoints_list,segment_m)
plt.show()

#画原始信号的f标记
plt.plot(raw_data[offset:offset+list(value[22])[2]])
log_l=[]
c=0
dur_l=[]
for i in range(22):
    start=list(value[i])[2]+offset
    stop=list(value[i])[3]+start
    c+=1
    plt.hlines(np.median(raw_data[start:stop]),start-offset,stop-offset,label='true',colors='r')
    dur_l.append(stop-start)
    #plt.hlines(np.median(raw_data[start:stop]),start,stop,label='true',colors='r')
    # if i>1 and c<len(prob_data):
    #     plt.hlines(np.median(prob_data[i-1]),start-offset,stop-offset,label='prob',colors='black')
    #     #plt.hlines(np.median(prob_data[i-1]),start,stop,label='prob',colors='black')
    log_l.append(str(list(value[i])[4])[2])

plt.plot(temp_l)

#plt.text(0,0,str(log_l))
plt.xlabel(str(log_l))
plt.show()




#dtw
#temp_l=list(raw_data[offset+30:offset+200])
align=dtw(segment_m,temp_l,keep_internals=True,step_pattern=asymmetric, open_end=True,open_begin=True)
align.plot(type='threeway')
plt.savefig('/mnt/raid5/lijin/Hectobio/testcode/DynamicPrograming/tombo_example.png',dpi=300)
print(offset)

print(align.distance)
print(align.stepsTaken)
print(align.index1)
print(align.index2)