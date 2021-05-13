import os
import sys
import tifffile as tif
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
sys.path.append('..')
from lib.util import ImageMeta, find_matching_files


def fix_one_image(
    fname,
    compression='DEFLATE',
    open_kwargs=dict(),
    write_kwargs=dict()
):
    image = ImageMeta(str(fname), **open_kwargs)

    with tif.TiffWriter(fname) as tfw:
        tfw.write(image.asarray(), compression=compression)

    image.close()

    return True


def main():
    if len(sys.argv) < 2:
        print(f'usage: {__file__} directory')
        print('recursively reshapes TIFF files using '
              'ImageMeta to guess the correct shape')
