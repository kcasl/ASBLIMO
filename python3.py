import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import convolve2d
from scipy.ndimage import gaussian_filter

#무작위 이미지 생성
image = np.random.rand(100, 100)

#가우시안 필터 생성
sigma = 2
gaussian_filter_kernel = gaussian_filter(np.zeros((21, 21)), sigma)

#이미지 필터링
filtered_image = convolve2d(image, gaussian_filter_kernel, mode='same', boundary='wrap')

#결과
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray', origin='upper')
plt.title('Original Image')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.imshow(filtered_image, cmap='gray', origin='upper')
plt.title(f'Filtered Image (Gaussian Filter, Sigma = {sigma})')
plt.axis('off')

plt.tight_layout()
plt.show()