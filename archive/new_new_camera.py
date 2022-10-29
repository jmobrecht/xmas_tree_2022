import imageio as iio
import matplotlib.pyplot as plt
import time

camera = iio.get_reader('<video0>')

t0 = time.time()

img = camera.get_data(0)
#img = iio.v3.imread('<video0>')

t1 = time.time()
print('Time: {:1.2f} ms'.format((t1 - t0) * 1E3))
camera.close()

plt.imshow(img)
plt.show()
