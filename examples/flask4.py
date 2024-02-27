import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import boto3
from flask import Flask, jsonify, request, abort
import requests
import cv2
import matplotlib.pyplot as plt

# 사용자가 업로드한 사진이 저장된 버킷 이름
bucket_name = 'medit-static-files'

# AWS 계정의 액세스 키 ID 및 비밀 액세스 키
AWS_ACCESS_KEY_ID = 'AKIA5GP7S5DFKJJHEPAF'
AWS_SECRET_ACCESS_KEY = '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'

# AWS 서비스 및 지역(region) 지정
aws_region = 'ap-northeast-2'

# S3 클라이언트 설정
s3 = boto3.client('s3', region_name=aws_region,
                  aws_access_key_id=AWS_ACCESS_KEY_ID,
                  aws_secret_access_key=AWS_SECRET_ACCESS_KEY)

app = Flask(__name__)

def download_image_from_url(image_url, file_name='image.jpg'):
    try:
        response = requests.get(image_url)
        response.raise_for_status()  # 오류 발생 시 예외 발생
        with open(file_name, 'wb') as f:
            f.write(response.content)
        print(f"Downloaded image URL: {image_url}")
        return True
    except Exception as e:
        print(f"Failed to download the image: {e}")
        return False

@app.route('/upload', methods=['POST'])
def receive_json():
    try:
        json_data = request.get_json()
        print(json_data)

        # JSON 데이터에서 이미지 URL 추출
        image_url = json_data.get('imageUrl')
        if not image_url:
            abort(400, 'No image URL provided')

        # 이미지 URL 다운로드
        if download_image_from_url(image_url):
            image = cv2.imread('image.jpg')
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.axis('off')
            plt.show()
            return jsonify({'message': 'Image received and displayed successfully.'}), 200
        else:
            abort(400, 'Failed to download the image')
    except Exception as e:
        abort(400, f'Error occurred: {e}')

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=5002)
