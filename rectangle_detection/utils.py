import os
import cv2
from shutil import rmtree


def ensure_dir(path, delete_if_exists=False):
    if delete_if_exists:
        rmtree(path, ignore_errors=True)
    if not os.path.exists(path):
        os.makedirs(path)


def get_files_from_dir(dir_path, include_paths=True):
    print('Discovering: {}...'.format(dir_path), end='\r')
    files = os.listdir(dir_path)
    files = list(filter(lambda x: os.path.isfile(os.path.join(dir_path, x)), files))
    print('Found {} files.'.format(len(files)))

    if include_paths:
        file_paths = [os.path.join(dir_path, file) for file in files]
        return files, file_paths

    return files


def named_window(img, name, size, wait_key=True):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(name, size[0], size[1])
    cv2.imshow(name, img)
    if wait_key:
        cv2.waitKey(0)
