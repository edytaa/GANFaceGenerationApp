import numpy as np
import zmq
import PIL
import sys

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

destinDir = basePth + 'stimuli/images/'  # path to folder for storing images

seed = 3004
rnd = np.random.RandomState(seed)
nTrl = 5  # number of trials
nGen = 2  # number of generations
trial = -1
gen = 0

network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
G, D, Gs = pretrained_networks.load_networks(network_pkl)
print("Network was loaded")
noise_vars = [var for name, var in Gs.components.synthesis.vars.items() if name.startswith('noise')]

truncation_psi = 0.5
Gs_kwargs = dnnlib.EasyDict()
Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
Gs_kwargs.randomize_noise = False
if truncation_psi is not None:
    Gs_kwargs.truncation_psi = truncation_psi

print("Server initialised - waiting for a client")

def gen_images(allZ_, gen_):
    for tt in range(allZ_.shape[0]):
        thsTrlPth = destinDir + 'trl_' + str(gen_) + "_" + str(tt) + '.png'
        print(f'Generating image  {tt} for generation {gen_} ...')
        z = allZ[np.newaxis, tt, :]
        tflib.set_vars({var: rnd.randn(*var.shape.as_list()) for var in noise_vars})  # [height, width]
        images = Gs.run(z, None, **Gs_kwargs)  # [minibatch, height, width, channel]
        PIL.Image.fromarray(images[0], 'RGB').save(thsTrlPth)

while True:
    #  Wait for next request from client
    request = socket.recv()
    print("Received request: %s" % request)  # print request in terminal
    message = request.decode("utf-8")  # request as a string
    state = message.split(' ')[1]  # get state of the app (True or False)
    rate = message.split(' ')[3]  # get rate from the previous trial (None or 1-9)

    # start a new experiment
    if state == "True":
        print("Starting a new experiment...")
        allZ = np.random.randn(nTrl, 512)
        trial = 0
        gen = 0
        gen_images(allZ, gen)

    #  if
    elif trial == nTrl-1:
        trial = 0
        gen += 1
        allZ = np.random.randn(nTrl, 512)  # TODO delete later
        gen_images(allZ, gen)

    else:
        #  Send reply back to client
        trial += 1

    print(f'Trial: {trial}')
    socket.send_multipart([bytes([trial]), bytes([gen])])
