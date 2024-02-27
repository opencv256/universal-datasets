import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
import torchxrayvision as xrv
import albumentations as A
from functools import partial
from univdt.components import NIH
from univdt.transforms.builder import AVAILABLE_TRANSFORMS

def main():
    root_dir = 'C:\\nih\\nih' # TODO: set root dir
    nih = partial(NIH, root_dir=root_dir, transform=None)

    nih_val = nih(split='val')
    print(len(nih_val))
    assert len(nih_val) == 9925

    data = nih_val.load_data(0)
    image = data['image']
    label = data['label']
    print(image.shape, label) 

    plt.figure(figsize=(15,5))
    plt.subplot(1,3,1)
    plt.imshow(image, cmap='gray')
    plt.show()

    DEFAULT_HEIGHT = 768
    DEFAULT_WIDTH = 768
    DEFAULT_TRAIN_TRANSFORMS = A.Compose([AVAILABLE_TRANSFORMS['random_resize'](height=DEFAULT_HEIGHT,
                                                                                width=DEFAULT_WIDTH,
                                                                                pad_val=0, p=1.0),
                                          A.HorizontalFlip(p=0.5),
                                          AVAILABLE_TRANSFORMS['random_blur'](magnitude=0.2, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_brightness'](magnitude=0.2, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_contrast'](magnitude=0.2, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_gamma'](magnitude=0.2, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_noise'](magnitude=0.2, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_windowing'](magnitude=0.5, p=0.5),
                                          AVAILABLE_TRANSFORMS['random_zoom'](scale=0.2, pad_val=0, p=0.5),
                                          A.Affine(rotate=(-45, 45), p=0.5),
                                          A.Affine(translate_percent=(0.01, 0.1), p=0.5),
                                          ])

    nih_val = NIH(root_dir=root_dir, transform=DEFAULT_TRAIN_TRANSFORMS, split='val')

    data = nih_val.load_data(0)

    batch = nih_val[0]

    print(batch.keys())
    print(batch['image'].shape, batch['label'])

    plt.figure(figsize=(15,5))
    original_image = cv2.imread(batch['path'])
    plt.subplot(1,3,1)
    plt.imshow(original_image)
    plt.show()

    transformed = batch['image']
    plt.subplot(1,3,2)
    transformed = transformed.numpy().transpose(1,2,0)
    plt.imshow(transformed, cmap='gray')
    plt.show()

    data = nih_val.load_data(0)
    image = data['image']
    label = data['label']
    print(image.shape, label) 

    image = cv2.resize(image, (512,512))
    image = np.expand_dims(image, axis=2)
    print(image.shape, label)

    image = xrv.datasets.normalize(image, 255) 
    image:np.ndarray = image.mean(2)[None, ...] # Make single color channel
    image = image.astype(dtype=np.float32)
    print(image.shape, label)
    image = torch.from_numpy(image)
    print(image.shape, label, image.dtype)

    model = xrv.models.ResNet(weights="resnet50-res512-all")
    outputs = model(image[None,...]) # or model.features(img[None,...]) 

    from pprint import pprint 
    pprint(dict(zip(model.pathologies,outputs[0].detach().numpy())))

def show_image(image):
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    main()
