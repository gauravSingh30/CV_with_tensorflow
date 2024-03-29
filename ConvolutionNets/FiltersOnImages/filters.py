import numpy as np
from scipy import misc
i = misc.ascent()

import matplotlib.pyplot as plt
plt.grid(False)
plt.gray()
plt.axis('off')
plt.imshow(i)
plt.show()

i_transformed = np.copy(i)
size_x = i_transformed.shape[0]
size_y = i_transformed.shape[1]

# filters = [ [-1, -2, -1], [0, 0, 0], [1, 2, 1]]
filters = [ [-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
weight = 1

for x in range(1,size_x-1):
      for y in range(1,size_y-1):
            convolution = 0.0
            convolution = convolution + (i[x - 1, y-1] * filters[0][0])
            convolution = convolution + (i[x, y-1] * filters[0][1])
            convolution = convolution + (i[x + 1, y-1] * filters[0][2])
            convolution = convolution + (i[x-1, y] * filters[1][0])
            convolution = convolution + (i[x, y] * filters[1][1])
            convolution = convolution + (i[x+1, y] * filters[1][2])
            convolution = convolution + (i[x-1, y+1] * filters[2][0])
            convolution = convolution + (i[x, y+1] * filters[2][1])
            convolution = convolution + (i[x+1, y+1] * filters[2][2])
            convolution = convolution * weight
            if(convolution<0):
                convolution=0
            if(convolution>255):
                convolution=255
            i_transformed[x, y] = convolution


plt.gray()
plt.grid(False)
plt.imshow(i_transformed)
#plt.axis('off')
plt.show()

#POOLING

new_x = int(size_x/2)
new_y = int(size_y/2)
newImage = np.zeros((new_x, new_y))
for x in range(0, size_x, 2):
    for y in range(0, size_y, 2):
        pixels = []
        pixels.append(i_transformed[x, y])
        pixels.append(i_transformed[x+1, y])
        pixels.append(i_transformed[x, y+1])
        pixels.append(i_transformed[x+1, y+1])
        newImage[int(x/2),int(y/2)] = max(pixels)

# Plot the image. Note the size of the axes -- now 256 pixels instead of 512
plt.gray()
plt.grid(False)
plt.imshow(newImage)
#plt.axis('off')
plt.show()

