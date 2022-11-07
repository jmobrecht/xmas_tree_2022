import board
import neopixel

def run_first_x_pixels(x, rgb):
    pixels = neopixel.NeoPixel(board.D12, 350, brightness=1, pixel_order='RGB')
    pixels.fill((0, 0, 0))
    sleep(2)
    pixels.fill(rgb)
    pixels.show()

def run_pixel_x(x, rgb):
    pixels = neopixel.NeoPixel(board.D12, 350, brightness=1, pixel_order='RGB')
    pixels.fill((0, 0, 0))
    sleep(2)
    pixels[x-1] = rgb
    pixels.show()

def all_off():
    pixels = neopixel.NeoPixel(board.D12, 350, brightness=0, pixel_order='RGB')
    pixels.fill((0, 0, 0))

run_first_x_pixels(350, (255, 0, 0))
#run_pixel_x(10, (255, 255, 255))
#all_off()
