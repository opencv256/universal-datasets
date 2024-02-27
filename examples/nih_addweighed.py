from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import copy
import cv2
import torchxrayvision as xrv

model = xrv.models.ResNet(weights="resnet50-res512-all")

cam = GradCAM(model=model, target_layers=model.model.layer4)

idx = 11
print(model.pathologies[idx])
actmap = cam(input_tensor=image[None,...], targets=[ClassifierOutputTarget(idx)])

def min_max_norm(image: np.ndarray) -> np.ndarray:
    image = image.astype(np.float32)
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    return image

def post_process_cam(prob, actmap: np.ndarray, image: np.ndarray, threshold: 0.5, color_contour=(255,0,255)) -> np.ndarray:
    if prob < threshold:
        draw = copy.deepcopy(image)
        mask = np.zeros_like(image)
        return draw, mask
    actmap = min_max_norm(actmap)
    new_cam = copy.deepcopy(actmap)

    # for calc contours
    new_cam2 = np.where(new_cam > threshold, 1, 0).astype(np.uint8)
    contours, _ = cv2.findContours(new_cam2, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    new_cam = np.where(new_cam > threshold, new_cam, 0)

    filtered_contours = []
    for contour in contours:
        # 컨투어 내부의 최대 값 찾기
        max_value = np.max(new_cam * (cv2.drawContours(np.zeros_like(new_cam),
                                                       [contour], 0, 1,
                                                       thickness=cv2.FILLED)))
        # 최대 값이 0.8 이상인 경우만 남기기
        if max_value >= 0.8:
            filtered_contours.append(contour)
    # 원본에 컨투어 그리기
    if image.shape[-1] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    draw = cv2.drawContours(copy.deepcopy(image),
                            filtered_contours, -1, color_contour, 1)
    mask = cv2.drawContours(np.zeros_like(image),
                            filtered_contours, -1, 1, -1)
    mask = cv2.cvtColor(actmap, cv2.COLOR_GRAY2RGB) * mask
    mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
    return draw, mask

tt = min_max_norm(image[0].numpy())*255.0
# print(tt.min(), tt.max())
tt = cv2.cvtColor(tt.astype(np.uint8), cv2.COLOR_GRAY2BGR)
draw, mask = post_process_cam(0.5, actmap[0], tt, 0.5)

plt.figure(figsize=(15,5))
plt.subplot(1,2,1)
plt.imshow(draw)
plt.subplot(1,2,2)
plt.imshow(mask)