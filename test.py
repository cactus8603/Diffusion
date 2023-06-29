import json

with open('three_corner.json', 'r') as f:
    stroke = json.load(f)
print(len(stroke))