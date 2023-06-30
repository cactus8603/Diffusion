import json

with open('./cfgs/three_corner.json', 'r') as f:
    stroke = json.load(f)

# with open('./word_4807.txt', 'w') as f:
#     for chr in stroke.keys():
#         f.write(chr + '\n')

with open('word_4807.txt', 'r') as f:
    data = f.readlines()
    data = [value.strip() for value in data]

print(data[0])
print(stroke[data[0]])

# for chr in stroke.keys():
#     print(chr)


from glob import glob
import os
import shutil

# data_path = '/data/Font/stroke_and_corner_12937'
# total_font = glob(os.path.join(data_path, '*'))

# dest = '/data/Font/stroke_and_corner_12937/tmp'

# f = []
# tmp = 0
# for path in total_font:
#     total_chr = glob(os.path.join(path, '*'))
#     if len(total_chr) != 12937:
#         f.append(path)
#         print(path)
#         print(len(total_chr))
#         shutil.move(path, dest)
#         tmp += 1

# print(tmp)

# with open('12937.txt', 'w') as file:
#     for t in f:
#         file.write(t+'\n')





