import numpy as np
import cv2


def add_gaussian_noise(image_in, noise_sigma):
    temp_image = np.float64(np.copy(image_in))
    h, w = temp_image.shape[0:2]
    # 标准正态分布*noise_sigma
    noise = np.random.randn(h, w) * noise_sigma
    # print(noise)
    noisy_image = np.zeros(temp_image.shape, np.float64)
    if len(temp_image.shape) == 2:  # 单通道 or三通道
        noisy_image = temp_image + noise
    else:
        noisy_image[:, :, 0] = temp_image[:, :, 0] + noise
        noisy_image[:, :, 1] = temp_image[:, :, 1] + noise
        noisy_image[:, :, 2] = temp_image[:, :, 2] + noise

    return np.uint8(noisy_image)
