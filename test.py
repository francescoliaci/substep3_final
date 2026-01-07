import numpy as np

data = np.load("output_subtask_1.npy", allow_pickle=True)
data = data.item()   # ⬅️ IMPORTANTISSIMO

print(type(data))
print(len(data))

for i, (k, v) in enumerate(data.items()):
    print(k, v.shape)
    if i == 10:
        break

key = list(data.keys())[0]
print("KEY:", key)
print("VECTOR SHAPE:", data[key].shape)
print("FIRST 10 VALUES:", data[key][:10])

import pprint
pprint.pprint(list(data.items())[:3])
