import os
import cv2
import numpy as np

label_list = ['l_eye', 'r_eye', 'u_lip', 'l_lip']

folder_base = 'CelebAMask-HQ-mask-anno'
folder_save = 'CelebAMaskHQ-mask'

img_num = 30000

os.chdir(os.path.join(os.getcwd()))

print(os.listdir())


def make_folder(path):
    if not os.path.exists(os.path.join(path)):
        os.makedirs(os.path.join(path))


make_folder(folder_save)

for k in range(img_num):
    folder_num = k // 2000
    im_base = np.zeros((512, 512))
    is_empty = True
    for idx, label in enumerate(label_list):
        filename = os.path.join(folder_base, str(folder_num), str(
            k).rjust(5, '0') + '_' + label + '.png')
        if os.path.exists(filename.strip(' ')):
            is_empty = False
            im = cv2.imread(filename)
            im = im[:, :, 0]
            im_base[im != 0] = 255
    if not is_empty:
        filename_save = os.path.join(folder_save, str(k) + '.png')
        print(filename_save)
        cv2.imwrite(filename_save, im_base)
