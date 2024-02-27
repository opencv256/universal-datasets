import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import boto3
from flask import Flask, jsonify, request
import requests
import numpy as np
import torch
import cv2
import json
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchxrayvision as xrv
from univdt.components.nih import MAPPER

# 사용자가 업로드한 사진이 저장된 버킷 이름
bucket_name = 'medit-static-files'

# AWS 계정의 액세스 키 ID 및 비밀 액세스 키
AWS_ACCESS_KEY_ID = 'AKIA5GP7S5DFKJJHEPAF'
AWS_SECRET_ACCESS_KEY = '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'

# AWS 서비스 및 지역(region) 지정
aws_region = 'ap-northeast-2'

# boto3 모듈을 이 지점에서 임포트하여 사용
s3 = boto3.client('s3', 
                  region_name=aws_region,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)


def download_image_from_url(url):
    # 이미지 다운로드
    response = requests.get(url)
    if response.status_code == 200:
        image_data = response.content
        # 이미지 파일로 저장
        with open('image.jpg', 'wb') as f:
            f.write(image_data)
        print(f"Downloaded image URL: {url}")
        return True
    else:
        return False

app = Flask(__name__)

@app.before_first_request
def download_and_display_image():
    # 특정 이미지 URL
    photo_url = "https://medit-static-files.s3.ap-northeast-2.amazonaws.com/0e47070c-3ec0-4167-9ffb-baa2db867943MIMIC-CXR-Chest-X-Ray-00_0.jpeg"
    
    # 이미지 파일 다운로드
    if download_image_from_url(photo_url):
        # 이미지 파일 열기
        os.system('start image.jpg')  # 윈도우의 경우 이미지를 기본 뷰어로 열기


def resize_image(image):
    # OpenCV를 사용하여 이미지 크기 조정
    image_resized = cv2.resize(image, (512, 512))
    return image_resized

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

def process_image(image_resized):
    # XRV 패키지의 ResNet 모델을 로드하고 입력 이미지를 처리하여 결과 반환
    model = xrv.models.ResNet(weights="resnet50-res512-all")
    outputs = model(image_resized[None, ...])

    # 병변과 확률을 매핑하기 위한 사전 생성
    pathology_probabilities = dict(zip(model.pathologies, outputs[0].detach().numpy()))

    # MAPPER에 있는 질병들만을 포함하는 결과를 반환
    result = {}
    for disease, probability in pathology_probabilities.items():
        if disease.lower() in MAPPER:
            result[disease.lower()] = probability

    return result

def upload_file_to_s3(image_path, bucket_name, s3_file_name):
    s3.upload_file(image_path, bucket_name, s3_file_name)
    print(f"Uploaded image to S3: {s3_file_name}")

    # 파일이 업로드된 S3 URL 반환
    file_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_name}"
    return file_url

# 이미지 URL을 받아와 처리된 이미지를 반환하는 엔드포인트
@app.route('/process_image', methods=['POST'])
def process_image_endpoint():
    data = request.get_json()
    image_url = data['image_url']

    processed_image = process_image(image_url)

    if processed_image is not None:
        # 처리된 이미지를 파일로 저장하여 S3에 업로드
        cv2.imwrite('processed_image.jpg', processed_image)
        s3_file_name = 'processed_image.jpg'
        s3.upload_file('processed_image.jpg', bucket_name, s3_file_name)
        uploaded_image_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{s3_file_name}"

        return jsonify({'uploaded_image_url': uploaded_image_url})
    else:
        return jsonify({'error': 'Failed to process image'}), 400

@app.route('/show_image')
def show_image():
    # 이미지 로드
    image = cv2.imread('image.jpg')
    
    # 이미지 크기 조정
    resized_image = resize_image(image)
    
    # 이미지 정규화
    normalized_image = normalize_image(resized_image)
    
    # 이미지 처리
    processed_image = process_image(normalized_image)

    # 결과를 저장할 리스트
    result = []
    label = []
    
    # 출력 결과와 해당 질병의 확률을 표시
    for disease, probability in processed_image.items():
        probability_percent = round(probability * 100, 1)
        result.append(f"질병: {disease}, 확률: {probability_percent:.2f}%")  # 결과를 리스트에 추가

        # 확률값이 45%를 넘어가는 경우 label 리스트에 추가
        if probability > 0.45:
            probability_per = round(probability * 100, 1)
            label.append(f"질병: {disease}, 확률: {probability_per:.2f}%")

    # 확률이 임계값을 넘지 못한 경우 'normal'을 추가
    if not label:
        label.append("normal")

     # 이미지 파일을 S3에 업로드
    s3_file_name = 'image.jpg'  # 이미지 파일 이름은 고정
    uploaded_image_url = upload_file_to_s3('image.jpg', bucket_name, s3_file_name)

    # 처리된 이미지 값과 결과를 함께 반환
    return json.dumps({'result': result, 'label': label, 'uploaded_image_url': uploaded_image_url}, ensure_ascii=False)

if __name__ == '__main__':
    app.run(debug=True)
