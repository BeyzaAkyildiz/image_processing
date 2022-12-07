#!/usr/bin/env python
# coding: utf-8

# In[17]:


import cv2
import matplotlib.pyplot as plt
import numpy as np

img = cv2.imread("C:/Users/Beyza/Desktop/elma.jpeg",0)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
print(img)
plt.show()
plt.hist(img.ravel(), 256, [0, 256])
plt.show()
img = cv2.equalizeHist(img)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_GRAY2RGB))
plt.show()
h = img.ravel()
plt.hist(img.ravel(), 256, [0, 256])
plt.show()

k_size = 3
flt = np.ones((k_size, k_size)) / (k_size ** 2)
img2 = np.zeros((img.shape[0], img.shape[1]), np.uint8)
flt_x_diff = int(flt.shape[0] / 2)
flt_y_diff = int(flt.shape[1] / 2)
for i in range(flt_x_diff, img.shape[0] - flt_x_diff):
    for j in range(flt_y_diff, img.shape[1] - flt_y_diff):
        img2[i][j] = np.sum(np.multiply(
            img[i - flt_x_diff:i + flt_x_diff + 1, j - flt_y_diff:j + flt_y_diff + 1], flt
        ))

plt.imshow(cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB))
plt.show()

plt.hist(img2.ravel(), 256, [0, 256])
plt.show()

flt = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]
])
img3 = cv2.filter2D(src=img, ddepth=-1, kernel=flt)
plt.imshow(cv2.cvtColor(img3, cv2.COLOR_GRAY2RGB))
plt.show()

img4 = cv2.medianBlur(img, 3)
plt.imshow(cv2.cvtColor(img4, cv2.COLOR_GRAY2RGB))
plt.show()


# In[ ]:




