from matplotlib import pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import cv2
import copy
import numpy as np
from univdt.components import xrv

from univdt.components.nih import MAPPER

model = xrv.models.ResNet(weights="resnet50-res512-all")  

cam = GradCAM(model=model, target_layers=model.model.layer4)

# 새로운 배열 정의
new_pathologies = {'normal': 0, 'Atelectasis': 3, 'Consolidation': 5, 'Infiltration': 13, 'Pneumothorax': 10, 
                   'Edema': 4, 'Emphysema': 2, 'Fibrosis': 12, 'Effusion': 1, 'Pneumonia': 11, 
                   'Pleural_Thickening': 6, 'Nodule': 9, 'Mass': 8, 'Hernia': 7}

# 새로운 인덱스 생성
new_idx_mapping = {}
for pathology, idx in new_pathologies.items():
    if pathology in MAPPER:
        new_idx_mapping[pathology] = MAPPER[pathology]
    else:
        new_idx_mapping[pathology] = idx

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
def process_and_show_result(new_idx):
    print("Current new_idx:", new_idx)
    pathology = list(new_pathologies.keys())[list(new_pathologies.values()).index(new_idx)] # 수정된 부분
    print("Pathology from new_pathologies:", pathology)
    actmap = cam(input_tensor=image[None,...], targets=[ClassifierOutputTarget(new_idx)])
    tt = min_max_norm(image[0].numpy()) * 255.0
    tt = cv2.cvtColor(tt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
    draw, mask = post_process_cam(0.5, actmap[0], tt, 0.5)
    mask = cv2.resize(mask, (draw.shape[1], draw.shape[0]))
    mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
    mask_rgb = ((mask_rgb - mask_rgb.min()) * (255 / (mask_rgb.max() - mask_rgb.min()))).astype('uint8')
    draw = draw.astype('uint8')
    combined_image = cv2.addWeighted(draw, 0.7, mask_rgb, 0.3, 0)
    plt.imshow(combined_image)
    plt.show()

# new_idx를 변경하면서 결과를 출력
new_idx = 10
pathology = list(new_pathologies.keys())[list(new_pathologies.values()).index(new_idx)] # 수정된 부분
print(pathology)
process_and_show_result(new_idx)
