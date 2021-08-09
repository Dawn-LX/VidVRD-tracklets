import os
index = ["{:02d}".format(i) for i in range(1,15,1)]
print(index)
source = "configs/MEGA/inference/VidORtrain_freq1_2k2999.yaml"
for idx in index:
    dest = "configs/MEGA/partxx/VidORtrain_freq1_part{}.yaml".format(idx)
    os.system("cp {} {}".format(source,dest))