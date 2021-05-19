import numpy as np
import zmq
import PIL
import sys
import csv
import os
import pickle

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib
from random import randint

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")

participant_id = randint(0, 1000)
participant_dir = basePth + r'stimuli/'  # f'stimuli/{participant_id}/'
destinDir = participant_dir + 'images/'  # path to folder for storing images
if not os.path.exists(destinDir):
    os.makedirs(destinDir)

seed = 3004
rnd = np.random.RandomState(seed)
nTrl = 5  # number of trials
nGen = 2  # number of generations
trial = -1
gen = 0
nSurv = 1 # number of best samples that survives the grading process (and prob goes to next generation)
nRnd = 1

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


#def evaluateOneGeneration(nTrl_, nSurv_, nRnd_, gen_, wFitterParent_,
#                          basePth_, allZ_, ratings_, mutAmp=.4, mutP=.3):
def evaluateOneGeneration(allZ_, ratings_, gen_, nTrl_, nSurv_, nRnd_, basePth_=None, wFitterParent_=0.75, mutAmp=.4, mutP=.3):
    """

    :param nTrl_: number of trials in generation
    :param nSurv_: number of "best" pictures which are used to generate new set
    :param nRnd_: number of (fully) new random faces
    :param gen_: current generation
    :param wFitterParent_: "power" of the higher graded parent picture
    :param basePth_: not used now :p
    :param allZ_: last generation feature vector
    :param ratings_: last generation grades (list)
    :param mutAmp:  ?
    :param mutP: ?
    :return: updated allZ_
    """
    nInPool = nTrl_ - nSurv_ - nRnd_
    thsResp = ratings_

    # save current latents and responses
    #thsLatentPth = basePth_ + 'stimuli/latent/generation_' + str(gen_) + '.mat'
    #savemat(thsLatentPth, {"allZ_": allZ_, "thsResp": thsResp})

    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
    # transform responses to probabilities (needed for sampling parents)
    thsFitness = softmax(thsResp)

    # take latent vectors of nSurvival highest responses
    thsIndices = np.argsort(thsFitness)
    thsSurv = allZ_[thsIndices[-nSurv_:], :]

    # generate recombinations from 2 parent latent vectors of current gen
    # w. fitness proportional to probability of being parent
    # parent with higher fitness contributes more
    thsPool = np.zeros([nInPool, allZ_.shape[1]])

    for rr in range(nInPool):
        thsParents = np.random.choice(nTrl_, 2, False, thsFitness)

        if thsFitness[thsParents[0]] > thsFitness[thsParents[1]]:
            contrib0 = wFitterParent_
            contrib1 = 1 - wFitterParent_
        elif thsFitness[thsParents[0]] < thsFitness[thsParents[1]]:
            contrib0 = 1 - wFitterParent_
            contrib1 = wFitterParent_
        elif thsFitness[thsParents[0]] == thsFitness[thsParents[1]]:
            contrib0 = .5
            contrib1 = .5

        thsPool[rr, :] = allZ_[thsParents[0], :] * contrib0 + allZ_[thsParents[1], :] * contrib1

    # each latent dimension of children in recombined pool has some probability of mutation
    toEdit = np.random.choice([0, 1], (nInPool, thsPool.shape[1]), True, [1 - mutP, mutP])  # mutP global
    thsEdits = np.random.randn(np.sum(toEdit)) * mutAmp  # mutAmp global
    thsPool[np.nonzero(toEdit)] = thsPool[np.nonzero(toEdit)] + thsEdits

    # add some random faces to the mix
    thsRnd = np.random.randn(nRnd_, 512)

    # combine direct survivals and recombined / mutated pool
    allZ_ = np.concatenate((thsSurv, thsPool, thsRnd), axis=0)
    # shuffle order of trials
    np.random.shuffle(allZ_)
    return allZ_

def save_generation_info(gen, allZ_, grades, participant_path):
    with open(participant_path+r'grades.csv', 'a') as fd:
        write = csv.writer(fd)
        write.writerow(grades)
    with open(participant_path+f'allZ_gen_{gen}.pkl', 'wb') as fd:
        pickle.dump(allZ_, fd)


gen_rates = []

while True:
    #  Wait for next request from client
    request = socket.recv()
    print("Received request: %s" % request)  # print request in terminal
    message = request.decode("utf-8")  # request as a string
    state = message.split(' ')[1]  # get state of the app (True or False)
    rate = message.split(' ')[3]  # get rate from the previous trial (None or 1-9)
    print(f'grade: {rate}, {type(rate)}')
    if rate != "None":
        rate = int(rate)
        print(f'grade: {rate}, {bytes(rate)}, {type(rate)}')
        #rate = int.from_bytes(rate, "little")
        gen_rates.append(rate)
    print(f'current grades: {gen_rates}')

    # start a new experiment
    if state == "True":
        print("Starting a new experiment...")
        allZ = np.random.randn(nTrl, 512)
        trial = 0
        gen = 0
        gen_images(allZ, gen)
        gen_rates = []

    #  change generation
    elif trial == nTrl-1:
        trial = 0
        save_generation_info(gen, allZ, gen_rates, participant_dir)
        allZ = evaluateOneGeneration(allZ, gen_rates, gen, nTrl_=nTrl, nSurv_=nSurv, nRnd_=nRnd)
        gen += 1
        gen_images(allZ, gen)
        gen_rates = []

    else:
        #  Send reply back to client
        trial += 1

    print(f'Trial: {trial}')
    socket.send_multipart([bytes([trial]), bytes([gen])])
