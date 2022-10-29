import imageio as iio
import matplotlib.pyplot as plt
import time

#camera = iio.get_reader('<video0>')
#meta = camera.get_meta_data()
#delay = 1 / meta['fps']
#for frame_counter in range(15):
#    print(frame_counter)
#    frame = camera.get_next_data()
#    time.sleep(delay)
#camera.close()
#plt.imshow(frame)
#plt.show()

camera = iio.get_reader('<video0>')
meta = camera.get_meta_data()
delay = 1 / meta['fps']
for frame_counter in range(15):
    
    t0 = time.time()

    frame = camera.get_next_data()
    
    t1 = time.time()
    print('Time 1: {:1.2f} ms'.format((t1 - t0) * 1E3))

    t0 = time.time()
    
    time.sleep(delay)

    t1 = time.time()
    print('Time 2: {:1.2f} ms'.format((t1 - t0) * 1E3))
    
camera.close()

import numpy as np
import matplotlib.pyplot as plt

plt.imshow(frame)
plt.show()