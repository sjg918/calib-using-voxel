
import numpy as np
import random

nlist = []

for i in range(100):
    n = random.randint(1, 35989-100)
    while n in nlist:
        n = random.randint(1, 35989-100)
    nlist.append(n)
print(nlist)

with open('/home/jnu-ie/kew/calib-using-voxel/gendata/100number-oxford.txt', 'w') as writer:
    for i in range(100):
        n = nlist[i]
        if i==99:
            writer.write(str(n))
            continue
        writer.write(str(n) + '\n')
        continue
