import json
import os
import tqdm

p = 'C:/Users/Aleksander/source/repos/C/insilico/blender/PlantGenerator/render2/render'
for f in os.listdir(p):
    if f.endswith("_meta_plants.txt"):
        with open(os.path.join(p, f), 'r') as fs:
            j = json.load(fs)
        for plant in j:
            bb = plant['bounding_box']
            assert "x" in bb
            assert "y" in bb
            assert "w" in bb
            assert "h" in bb
        # print(j)
