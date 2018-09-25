# coding=utf-8
import os

from PIL import Image

from tools.load_configs import Config


def generate_rbg_pixels(file_path, window_size=1):
    assert file_path.endswith(".gif")
    frame = Image.open(file_path)
    n_frames = 0

    square = window_size ** 2.
    while frame:
        frame_rgb = frame.convert("RGB")

        pixels = []
        for _y in range(0, (frame_rgb.height // window_size) * window_size, window_size):
            for _x in range(0, (frame_rgb.width // window_size) * window_size, window_size):
                window = tuple(frame_rgb.getpixel((_x + __x, _y + __y)) for __y in range(window_size) for __x in range(window_size))
                _r, _g, _b = zip(*window)
                each_pixel = round(sum(_r) / square), round(sum(_g) / square), round(sum(_b) / square)
                pixels.append(each_pixel)
        yield tuple(pixels)

        n_frames += 1
        try:
            frame.seek(n_frames)
        except EOFError:
            n_frames = 0


def generate_grayscale_pixels(file_path, window_size=1):
    for each_frame in generate_rbg_pixels(file_path, window_size=window_size):
        yield tuple(round(sum(_p) / 3) for _p in each_frame)


def generate_pixel_examples(pixel_generator):
    last_pixels = next(pixel_generator)
    while True:
        pixels = next(pixel_generator)
        yield zip(last_pixels, pixels)
        last_pixels = pixels


def write_image(greyscale_pixels, width, height, file_path):
    assert len(greyscale_pixels) == width * height
    frame = Image.new("RGB", (width, height))
    grid = frame.load()

    _i = 0
    for _y in range(height):
        for _x in range(width):
            value = greyscale_pixels[_i]
            grid[_x, _y] = value, value, value
            _i += 1
    """            
    for _i, each_pixel in enumerate(greyscale_pixels):
        _x = _i % width
        _y = _i // height
        grid[_x, _y] = each_pixel, each_pixel, each_pixel
    """
    frame.save(file_path, os.path.splitext(file_path)[-1][1:])


if __name__ == "__main__":
    config = Config("../configs/config.json")
    size = 5
    original = Image.open(config["data_dir"] + "gifs/tenor.gif")
    for no_frame, values in enumerate(generate_grayscale_pixels(config["data_dir"] + "gifs/tenor.gif", window_size=size)):
        print(f"processing frame {no_frame}...")
        write_image(values, original.width // size, original.height // size, "{:03d}.png".format(no_frame))
