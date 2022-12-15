import json
import random

coco_json = '/media/nicola/SSD/Datasets/COCO_2014/annotations_trainval2014/captions_val2014.json'
output_txt = 'alad/extraction/analytics/output/queries_from_coco.txt'

with open(coco_json, 'r') as f:
    data = json.load(f)

sentences = [d["caption"] for d in data["annotations"]]
sentences = [s.replace('\n', ' ') for s in sentences]
print(f'Fetched {len(sentences)} sentences.')
random.shuffle(sentences)
sentences = sentences[:5000]
with open(output_txt, 'w') as f:
    f.write('\n'.join(sentences))
