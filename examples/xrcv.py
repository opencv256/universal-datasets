# torch, cv2, numpy, functools, random, matplotlib.pyplot, torchxrayvision, univdt.components에서 사용할 모듈 import
import torch
import cv2
import numpy as np
from functools import partial
import random
from matplotlib import pyplot as plt
import torchxrayvision as xrv
from univdt.components.nih import MAPPER


def generate_random_index(max_index):
    # 중복을 피하기 위해 생성된 인덱스를 추적하기 위한 빈 집합 생성
    generated_indexes = set()  
    while True:
        # 주어진 범위 내에서 무작위 인덱스 생성
        random_index = random.randint(0, max_index - 1)
        if random_index not in generated_indexes:  # 중복되지 않은 인덱스인 경우
            generated_indexes.add(random_index)  # 생성된 인덱스 집합에 추가하고
            return random_index  # 해당 인덱스 반환

def generate_random_index_with_path(max_index, dataset):
    generated_indexes = set()  
    while True:
        random_index = random.randint(0, max_index - 1)
        if random_index not in generated_indexes:
            generated_indexes.add(random_index)
            image_path = dataset.raw_data[random_index]['path']
            return random_index, image_path  

# 이미지 크기를 조정하는 함수 정의
def resize_image(image):
    # 이미지 크기를 조정하고 그림을 그리기 위해 그림 크기 설정
    plt.figure(figsize=(15, 5))
    # OpenCV를 사용하여 이미지 크기 조정
    image_resized = cv2.resize(image, (512, 512))
    # 이미지 차원을 확장하여 채널을 추가
    image_resized = np.expand_dims(image_resized, axis=2)
    return image_resized

# 이미지를 정규화하는 함수 정의
def normalize_image(image_resized):
    # XRV 패키지의 데이터셋을 사용하여 이미지를 정규화하고 255로 나눔
    image_resized = xrv.datasets.normalize(image_resized, 255)
    # 이미지의 각 픽셀값의 평균을 계산하고 새로운 차원을 추가하여 반환
    image_resized = image_resized.mean(2)[None, ...]
    # 이미지 데이터 타입을 32비트 부동소수점으로 변환
    image_resized = image_resized.astype(dtype=np.float32)
    # 이미지 데이터를 Torch 텐서로 변환하여 반환
    image_resized = torch.from_numpy(image_resized)
    return image_resized

# 이미지를 처리하는 함수 정의
def process_image(image_resized):
    # XRV 패키지의 ResNet 모델을 로드하고 입력 이미지를 처리하여 결과 반환
    model = xrv.models.ResNet(weights="resnet50-res512-all")
    outputs = model(image_resized[None, ...])
    return outputs

# 이미지와 라벨을 함께 시각화하는 함수 정의
def plot_image(image, label, outputs):
    # 이미지를 출력하기 위한 서브플롯 설정
    plt.subplot(1, 3, 2)
    # 그레이스케일 이미지로 변환하여 출력
    plt.imshow(image, cmap='gray')

    # 다중 라벨인 경우
    if isinstance(label, np.ndarray):
        diseases = []  # 각 라벨에 해당하는 질병 이름을 저장하기 위한 리스트
        # 각 라벨에 대해 해당하는 질병 이름을 매핑하고 리스트에 저장
        for l in label:
            l = int(l) 
            if l not in MAPPER.values():  # 매핑된 질병이 없는 경우
                print(f"Label {l} not found in MAPPER.")
            else:
                disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(l)]
                diseases.append(disease_name)
        # 중복되지 않는 질병 이름을 추출하여 제목으로 사용
        unique_diseases = set(diseases)
        title = f'Disease: {", ".join(unique_diseases)}'

    # 단일 라벨인 경우
    else:
        label = int(label)  
        # 라벨에 해당하는 질병 이름을 매핑하고 제목으로 사용
        if label not in MAPPER.values():
            print(f"Label {label} not found in MAPPER.")
            title = "Unknown Disease"
        else:
            disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(label)]
            title = f"Disease: {disease_name}"

    # 출력 결과와 해당 질병의 확률을 표시
    output_probabilities_dict = dict(zip(diseases, outputs[0].detach().numpy()))
    result = []  # 결과를 저장할 리스트
    for disease in unique_diseases:
        if disease in output_probabilities_dict:
            probability = output_probabilities_dict[disease]
            probability_percent = round(probability * 100, 1)
            result.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")  # 결과를 리스트에 추가
    

    # 이미지와 함께 출력
    plt.show()