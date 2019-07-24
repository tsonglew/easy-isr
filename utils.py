import os
import re

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image
from ISR.models import RDN


def im_resize(img, shape, fixed_proportion):
    if fixed_proportion:
        return cv2.resize(img, (shape[1], shape[1] * img.shape[0] // img.shape[1]))
    return cv2.resize(img, (shape[1], shape[1]))

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

def handle_dir(src_dir, tgt_dir, rdn, width, height, fp):
    filenames = os.listdir(src_dir)

    for i in filenames:
        handle_file(os.path.join(src_dir, i), os.path.join(tgt_dir, i), rdn, width, height, fp)

def handle_file(src_path, tgt_path, rdn, width, height, fp):
    print('Handle file: ', src_path)
    img = cv2.imread(src_path)
    lr_img = np.array(img)
    sr_img = rdn.predict(lr_img)
    resized_img = im_resize(sr_img, (height, width), fp)
    cv2.imwrite(tgt_path, resized_img)
    print('Saved to: ', tgt_path)
