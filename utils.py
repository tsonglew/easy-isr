import os
import re

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from ISR.models import RDN

def resize_pics(dir, shape, fp):
    for root, dirs, files in os.walk(dir):
        for f in files:
            file_path = os.path.join(root, f)
            img = cv2.imread(file_path)
            if img.shape[:2] == shape:
                print(file_path, " shape: ", img.shape, " skipped.")
                continue
            resized_img = im_resize(img, shape, fp)
            print(file_path, " shape: ", img.shape, " => ", shape)
            cv2.imwrite(file_path, resized_img)

def im_resize(img, shape, fixed_proportion):
    if fixed_proportion is not None:
        # opencv-python resize: (image, (width, height))
        return cv2.resize(img, (shape[1], shape[1] * img.shape[0] // img.shape[1]))
    return cv2.resize(img, (shape[1], shape[0]))

def resize_np_array(np_array, width, height):
    np.resize(np_array, (width, height))
    return np_array

def load_model(model_path):
    fbn = os.path.basename(model_path)
    arch_params={
        'C': int(re.search(r'C\d+', fbn).group()[1:]),
        'D': int(re.search(r'D\d+', fbn).group()[1:]),
        'G': int(re.search(r'G\d+', fbn).group()[1:]),
        'G0': int(re.search(r'G0\d+', fbn).group()[2:]),
        'x': int(re.search(r'x\d+', fbn).group()[1:])
    }
    print("Load RDN model arch params: ", arch_params)
    rdn = RDN(arch_params=arch_params)
    rdn.model.load_weights(model_path)
    return rdn

def get_all_handled_imgs(tgt_path):
    return set(os.listdir(tgt_path))

def handle_dir(src_dir, tgt_dir, rdn, width, height, rs, fp, ps, bpos):
    filenames = os.listdir(src_dir)
    handled_imgs = get_all_handled_imgs(tgt_dir)
    error_imgs = []

    pbar = tqdm(filenames)
    for f in pbar:
        if f in handled_imgs:
            continue
        try:
            pbar.set_description("Processing %s" % f)
            handle_file(os.path.join(src_dir, f), os.path.join(tgt_dir, f), rdn, width, height, rs, fp, ps, bpos)
        except Exception:
            error_imgs.append(f)
        break

    print("Error files: ", error_imgs)


def handle_file(src_path, tgt_path, rdn, width, height, rs, fp, ps, bpos):
    img = cv2.imread(src_path)
    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img, padding_size=ps, by_patch_of_size=bpos)
    while sr_img.shape[1] < width:
        sr_img = rdn.predict(sr_img, padding_size=ps, by_patch_of_size=bpos)
    if rs:
        resized_img = im_resize(sr_img, (height, width), fp)
    resized_img = sr_img
    cv2.imwrite(tgt_path, resized_img)
