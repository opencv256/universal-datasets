import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import requests
import cv2
import numpy as np
import torch
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchxrayvision as xrv
import boto3
import copy

from flask import Flask, jsonify, request
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from univdt.components.nih import MAPPER

app = Flask(__name__)

# AWS 설정
bucket_name = 'medit-static-files'
AWS_ACCESS_KEY_ID = 'AKIA5GP7S5DFKJJHEPAF'
AWS_SECRET_ACCESS_KEY = '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'

aws_region = 'ap-northeast-2'
s3 = boto3.client('s3', 
                  region_name=aws_region,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

# 새로운 배열
new_pathologies = {'normal': 0, 'Atelectasis': 3, 'Consolidation': 5, 'Infiltration': 13, 'Pneumothorax': 10, 
                   'Edema': 4, 'Emphysema': 2, 'Fibrosis': 12, 'Effusion': 1, 'Pneumonia': 11, 
                   'Pleural_Thickening': 6, 'Nodule': 9, 'Mass': 8, 'Hernia': 7}

new_idx_mapping = {}
for pathology, idx in new_pathologies.items():
    if pathology in MAPPER:
        new_idx_mapping[pathology] = MAPPER[pathology]
    else:
        new_idx_mapping[pathology] = idx

# new_idx_mapping을 MAPPER에 업데이트
MAPPER.update(new_idx_mapping)

model = xrv.models.ResNet(weights="resnet50-res512-all")

cam = GradCAM(model=model, target_layers=model.model.layer4)

# 이미지 다운로드 함수
def download_image_from_url(url, file_name='image.jpg'):
    try:
        response = requests.get(url)
        response.raise_for_status()  
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded image URL: {url}")
        return True
    except Exception as e:
        print(f"Failed to download the image: {e}")
        return False

# S3에 이미지 업로드 함수
def upload_file_to_s3(image_path, bucket_name, s3_file_name):
    s3.upload_file(image_path, bucket_name, s3_file_name)
    print(f"Uploaded image to S3: {s3_file_name}")
    file_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_name}"
    return file_url

# 이미지 처리 함수
def process_image(image_resized, model):
    outputs = model(image_resized[None, ...])
    pathology_probabilities = dict(zip(model.pathologies, outputs[0].detach().numpy()))

    result = {}
    for disease, probability in pathology_probabilities.items():
        if disease.lower() in MAPPER:
            if MAPPER[disease.lower()] == new_idx_mapping:
                result[disease.lower()] = probability
    return result

def min_max_norm(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def post_process_cam(prob, actmap: np.ndarray, image: np.ndarray, threshold: float = 0.5, color_contour=(255,0,255)) -> np.ndarray:
    if prob < threshold:
        draw = copy.deepcopy(image)
        mask = np.zeros_like(image)
        return draw, mask
    actmap = min_max_norm(actmap)
    new_cam = copy.deepcopy(actmap)

    # 가우시안 블러 적용
    new_cam = cv2.GaussianBlur(new_cam, (5, 5), 0)

    # for calc contours
    new_cam2 = np.where(new_cam > threshold, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(new_cam2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    new_cam = np.where(new_cam > threshold, new_cam, 0)

    filtered_contours = []
    for contour in contours:
        # 컨투어 내부의 최대 값 찾기
        max_value = np.max(new_cam * (cv2.drawContours(np.zeros_like(new_cam), [contour], 0, 1, thickness=cv2.FILLED)))
        # 최대 값이 0.8 이상인 경우만 남기기
        if max_value >= 0.8:
            filtered_contours.append(contour)
    # 원본에 컨투어 그리기
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    draw = cv2.drawContours(copy.deepcopy(image), filtered_contours, -1, color_contour, 1)
    mask = np.zeros_like(image)
    
    # 컨투어 내부를 히트맵으로 채우기
    heatmap = cv2.applyColorMap((new_cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    mask = cv2.drawContours(mask, filtered_contours, -1, (255, 255, 255), -1)
    mask = cv2.bitwise_and(heatmap, mask)

    return draw, mask

# 이미지에 대한 연산을 수행하고 결과를 출력하는 함수
def process_and_show_result(new_idx, image_tensor, processed_image):
    print("Current new_idx:", new_idx)
    # new_idx를 사용하여 MAPPER에서 질병 클래스의 인덱스 찾기
    disease_class_index = MAPPER.get(new_idx, new_idx)
    # 해당 인덱스를 이용하여 targets 설정
    targets = [ClassifierOutputTarget(disease_class_index)]
    # GradCAM을 사용하여 히트맵 생성
    actmap = cam(input_tensor=image_tensor, target_layers=model.model.layer4, targets=targets)
    tt = min_max_norm(image_tensor[0].numpy()) * 255.0
    tt = cv2.cvtColor(tt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw, mask = post_process_cam(0.5, actmap[disease_class_index], tt, 0.5) 
    mask = cv2.resize(mask, (draw.shape[1], draw.shape[0]))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_rgb = ((mask_rgb - mask_rgb.min()) * (255 / (mask_rgb.max() - mask_rgb.min()))).astype('uint8')
    draw = draw.astype('uint8')
    combined_image = cv2.addWeighted(draw, 0.7, mask_rgb, 0.3, 0)

    # new_idx 값에 해당하는 질병명 찾기
    disease_name = list(MAPPER.keys())[list(MAPPER.values()).index(disease_class_index)]

    # new_idx 값에 해당하는 질병의 확률 찾기
    probability = processed_image.get(disease_name.lower(), 0.0)

    # 결과 출력
    print(f"질병: {disease_name}, 확률: {probability:.2f}%")
    plt.imshow(combined_image)
    plt.show()

def resize_image(image):
    # 이미지 크기를 760x760으로 조정합니다.
    image_resized = cv2.resize(image, (760, 760))
    return image_resized

# 이미지 표시 및 처리 엔드포인트
@app.route('/show_image', methods=['POST'])
def show_image():
    try:
        # JSON 데이터에서 이미지 URL 추출
        json_data = request.get_json()
        image_url = json_data.get('imageUrl')
        if not image_url:
            return jsonify({'error': 'No image URL provided'}), 400

        # 이미지 다운로드
        if download_image_from_url(image_url):
            # 이미지 로드
            image = cv2.imread('image.jpg')
            # 이미지 크기 조정
            resized_image = cv2.resize(image, (512, 512))
            # 이미지 정규화 및 Torch 텐서로 변환
            normalized_image = xrv.datasets.normalize(resized_image, 255).mean(2)[None, ...].astype(np.float32)
            image_tensor = torch.from_numpy(normalized_image)
            
            # GradCAM을 사용하여 히트맵 생성
            actmap = cam(input_tensor=image_tensor, target_layers=model.model.layer4, targets={"output": new_idx})
            if len(actmap) > 0:
                # 히트맵을 이미지에 적용하여 결합 이미지 생성
                tt = image * 255.0
                tt = cv2.cvtColor(tt.astype(np.uint8), cv2.COLOR_BGR2RGB)
                draw, mask = post_process_cam(0.5, actmap[0], tt, 0.5)
                mask = cv2.resize(mask, (draw.shape[1], draw.shape[0]))
                mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
                mask_rgb = ((mask_rgb - mask_rgb.min()) * (255 / (mask_rgb.max() - mask_rgb.min()))).astype('uint8')
                draw = draw.astype('uint8')
                combined_image = cv2.addWeighted(draw, 0.7, mask_rgb, 0.3, 0)
            else:
                print("The length of actmap is 0.")
                # 필요한 경우, 여기에 에러 처리 코드를 추가하세요.
                # 예를 들어, 길이가 0인 경우 에러를 반환하거나 다른 조치를 취할 수 있습니다.

            # 이미지 처리
            processed_image = process_image(image_tensor, model)
            
            # 결과 저장을 위한 리스트 초기화
            result = []
            label = []
            image_urls = []  # 새로운 이미지 URL 리스트

            # 결과 및 레이블 생성
            for disease, probability in processed_image.items():
                probability_percent = round(probability * 100, 1)
                result.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")  # 결과를 리스트에 추가

                # 확률값이 45%를 넘어가는 경우 label 리스트에 추가
                if probability > 0.45:
                    label.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")
                    
                    # 해당 질병의 이미지를 업로드
                    disease_image_path = f"{disease}_image.jpg"
                    # 이미지 크기를 760x760으로 조정합니다.
                    resized_combined_image = cv2.resize(combined_image, (760, 760))
                    # 업로드할 이미지의 파일 이름을 지정합니다.
                    cv2.imwrite(disease_image_path, resized_combined_image)
                    # S3에 이미지를 업로드하고 해당 이미지의 URL을 가져옵니다.
                    disease_image_url = upload_file_to_s3(disease_image_path, bucket_name, disease_image_path)
                    # image_urls 리스트에 해당 질병의 이미지 URL을 추가합니다.
                    image_urls.append(disease_image_url)
                    # 이미지를 업로드하고 나면 반복문을 종료합니다.
                    break
            
            # 'normal' 추가
            if not label:
                label.append("normal")

            # 결과 출력
            process_and_show_result(new_idx, image_tensor, processed_image)

            # 결과 반환
            return jsonify({'result': result, 'label': label, 'image_urls': image_urls}), 200
        else:
            return jsonify({'error': 'Failed to download the image'}), 400
    except Exception as e:
        return jsonify({'error': f'Error occurred: {e}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5002)
