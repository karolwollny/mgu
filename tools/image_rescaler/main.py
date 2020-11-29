# TODO: Add arg parser

import os
import glob
from PIL import Image

SIZE = (50, 50)
PHOTOS_DIR = '../../data/'


def resize_photo(photo_path: str, size: tuple) -> bool:
    """Function which resize "photo_path" photo to "size" and save it"""
    try:
        img = Image.open(photo_path)
        img_resized = img.resize(size, Image.ANTIALIAS)
        img_resized.save(photo_path)
        return True
    except:
        print(f"Could not resize file {photo_path}!")
        return False


if __name__ == '__main__':
    photos_paths = glob.glob(os.path.join(PHOTOS_DIR, '**/*.jpg'), recursive=True)
    i = 0

    for photo_path in photos_paths:
        resize_photo(photo_path, SIZE)
        i += 1
        print(f'Resized {i} of {len(photos_paths)} photos')

    print(f'Finished resizing {len(photos_paths)} photos')
