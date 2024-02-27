import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import random
import cv2
import numpy as np
import pandas as pd
import torch
from functools import partial
from flask import Flask, jsonify
from flask import make_response
from matplotlib import pyplot as plt
import torchxrayvision as xrv
from univdt.components import NIH
from univdt.components.nih import MAPPER

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False  # 이 설정을 추가

def generate_random_index(max_index):
    generated_indexes = set()
    while True:
        random_index = random.randint(0, max_index - 1)
        if random_index not in generated_indexes:
            generated_indexes.add(random_index)
            return random_index

root_dir = 'C:\\nih\\nih'  # TODO: 루트 디렉토리 설정
nih = partial(NIH, root_dir=root_dir, transform=None)

nih_test = nih(split='test')
assert len(nih_test) == 25596

@app.route('/random_image', methods=['GET'])
def get_random_image():
    random_index = generate_random_index(len(nih_test))
    data = nih_test.load_data(random_index)
    image = data['image']
    label = data['label']
    image_path = data['path']  # assuming the 'path' key contains the image path
    

    if isinstance(label, np.ndarray):
        label = label.tolist()

    image_resized = cv2.resize(image, (512, 512))
    image_resized = np.expand_dims(image_resized, axis=2)
    image_resized = xrv.datasets.normalize(image_resized, 255)
    image_resized = image_resized.mean(2)[None, ...]
    image_resized = image_resized.astype(dtype=np.float32)
    image_resized = torch.from_numpy(image_resized)

    model = xrv.models.ResNet(weights="resnet50-res512-all")
    outputs = model(image_resized[None, ...])

    result = {}
    result['label'] = label
    result['diseases'] = []

    if isinstance(label, list):
        for l in label:
            l = int(l)
            if l not in MAPPER.values():
                print(f"Label {l} not found in MAPPER.")
            else:
                disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(l)]
                result['diseases'].append(disease_name)
    else:
        label = int(label)
        if label not in MAPPER.values():
            print(f"Label {label} not found in MAPPER.")
            result['diseases'].append("Unknown Disease")
        else:
            disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(label)]
            result['diseases'].append(disease_name)

    unique_diseases = list(set(result['diseases']))
    output_probabilities_dict = dict(zip(unique_diseases, outputs[0].detach().numpy()))

    disease_probability_list = []
    for disease in unique_diseases:
        if disease in output_probabilities_dict:
            probability = output_probabilities_dict[disease]
            probability_percent = round(probability * 100, 1)
            disease_probability_list.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")

    result['probabilities'] = disease_probability_list
    
    return jsonify(result)

if __name__ == '__main__':
    app.run()
