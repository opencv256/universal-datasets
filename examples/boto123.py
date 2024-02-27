import json
import os
import boto3
import requests

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

def get_photo_urls(bucket_name):
    # 버킷 내의 객체(파일) 목록 가져오기
    response = s3.list_objects_v2(Bucket=bucket_name)

    # 사진 파일의 주소값을 담을 리스트
    photo_urls = []

    # 객체 목록을 순회하며 주소값 가져오기
    for obj in response.get('Contents', []):
        # 사진 파일의 주소는 버킷 URL과 파일의 키(key)로 구성됨
        photo_url = f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{obj['Key']}"
        photo_urls.append(photo_url)

    # 이미지 파일이 있는 디렉토리 경로만 필터링하여 가져오기
    image_directories = [url for url in photo_urls if url.endswith('/')]

    return image_directories

def download_and_display_image(url):
    # 이미지 다운로드
    response = requests.get(url)
    if response.status_code == 200:
        image_data = response.content
        # 이미지 파일로 저장
        with open('image.jpg', 'wb') as f:
            f.write(image_data)
        # 이미지 파일의 경로 출력
        print(f"Downloaded image URL: {url}")
        # 이미지 파일 열기
        os.system('start image.jpg')  # 윈도우의 경우 이미지를 기본 뷰어로 열기
    else:
        print(f"Failed to download image from {url}")

# 주소값을 가져오는 함수 호출
photo_urls = get_photo_urls(bucket_name)

# 이미지 파일이 있는 디렉토리의 이미지를 다운로드하여 출력
for directory_url in photo_urls:
    # 디렉토리 내의 객체(파일) 목록 가져오기
    prefix = directory_url.replace(f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/", "")
    response = s3.list_objects_v2(Bucket=bucket_name, Prefix=prefix)
    try:
        image_files = [f"https://{bucket_name}.s3.{aws_region}.amazonaws.com/{obj['Key']}" for obj in response.get('Contents', []) if obj['Key'].endswith(('.png', '.jpg', '.jpeg'))]
        for image_file in image_files:
            # 이미지 파일 다운로드 및 출력
            download_and_display_image(image_file)
    except json.JSONDecodeError:
        print(f"Failed to decode JSON response for directory {directory_url}")
        continue
