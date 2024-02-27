import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

from univdt.components.nih import NIH
from functools import partial
import torchxrayvision as xrv
from xrcv import generate_random_index, generate_random_index_with_path, resize_image, normalize_image, process_image, plot_image
from matplt import visualize_results 

# 무작위 인덱스 생성
root_dir = 'C:\\nih\\nih'  # 루트 디렉토리 설정
random_index = generate_random_index(25596)

# 이미지와 라벨 로드
nih = partial(NIH, root_dir=root_dir, transform=None)
nih_test = nih(split='test')
assert len(nih_test) == 25596

random_index, image_path = generate_random_index_with_path(len(nih_test), nih_test)
print("Random Index:", random_index)
print("Image Path:", image_path)

data = nih_test.load_data(random_index)
image = data['image']
label = data['label']

# 이미지 전처리
image_resized = resize_image(image)
image_resized = normalize_image(image_resized)

# 이미지 처리
outputs = process_image(image_resized)

# 이미지 출력
plot_image(image, label, outputs)

# 모델 객체 생성
model = xrv.models.ResNet(weights="resnet50-res512-all")  

# 결과 시각화
result_table = visualize_results(outputs, model)
print(result_table)


