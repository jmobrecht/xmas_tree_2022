import board
import neopixel

def all_off():
    pixels = neopixel.NeoPixel(board.D12, 650, brightness=0, pixel_order='RGB')
    pixels.fill((0, 0, 0))

all_off()
