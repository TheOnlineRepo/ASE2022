from PIL import Image
import PIL
import numpy as np
import os

def path_fig(path, adv_method, adv_eps, RGB=True):
    valid_file = os.path.join(path, adv_method, str(adv_eps), 'valid.npy.npz')
    if RGB:
        valid_x = np.load(valid_file)['x'][0]*255
    else:
        valid_x = np.load(valid_file)['x'][0, :, :, 0]*255
    test_image = Image.fromarray(valid_x.astype('uint8'))
    save_path = os.path.join(path, adv_method + '_' + str(adv_eps) + '.png')
    if RGB:
        test_image.convert('RGB').save(save_path)
    else:
        test_image.convert('L').save(save_path)

base_dir = '/media/data0/DeepSuite/adv_dataset/'
dataset = 'mnist'
model = 'leNet_1'
save_dir = os.path.join(base_dir, dataset, model)
if dataset == 'mnist':
    RGB = False
else:
    RGB = True
# path_fig(save_dir, 'FGSM', 0.3, RGB)
# path_fig(save_dir, 'FGSM', 0.5, RGB)
# path_fig(save_dir, 'FGSM', 0.8, RGB)
path_fig(save_dir, 'DF', 1e-5, RGB)
path_fig(save_dir, 'DF', 1e-6, RGB)
path_fig(save_dir, 'CW_Linf', 1, RGB)
# path_fig(save_dir, 'CW_Linf', 0.3, RGB)
