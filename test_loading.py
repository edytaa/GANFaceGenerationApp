import numpy as np
import PIL
import os
import pickle
import sys

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib


generation = 0
participant_id = 1905


def main():
    basePth = r'/home/edytak/Documents/GAN_project/code/'
    participant_dir = basePth + f'stimuli/{participant_id}/'
    pickle_file = participant_dir + f'allZ_gen_{generation}.pkl'

    save_path = participant_dir + r"regenerated/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
    G, D, Gs = pretrained_networks.load_networks(network_pkl)
    truncation_psi = 0.5
    Gs_kwargs = dnnlib.EasyDict()
    Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    Gs_kwargs.randomize_noise = False
    if truncation_psi is not None:
        Gs_kwargs.truncation_psi = truncation_psi

    noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]
    seed = 3004
    rnd = np.random.RandomState(seed)

    allZ = pickle.load(open( pickle_file, "rb"))
    gen_images(allZ, generation, save_path, rnd, Gs, Gs_kwargs, noise_vars)


def gen_images(allZ_, gen_, save_path_, rnd, Gs, Gs_kwargs, noise_vars):
    for tt in range(allZ_.shape[0]):
        thsTrlPth = save_path_ + 'trl_' + str(gen_) + "_" + str(tt) + '.png'
        print(f'Generating image  {tt} for generation {gen_} ...')
        z = allZ_[np.newaxis, tt, :]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(thsTrlPth)


if __name__ == "__main__":
    main()
