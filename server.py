import numpy as np
import zmq
import PIL
import sys
import csv
import os
import pickle
from os import walk

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib

TESTING = True

context = zmq.Context()
socket = context.socket(zmq.REP)
socket.bind("tcp://*:5555")


class Server:
    def __init__(self):
        if TESTING:
            seed = 3004
            self.rnd = np.random.RandomState(seed)
            self.nTrl = 6  # number of trials
            self.nGen = 10  # number of generations
            self.nSurv = 2  # number of best samples that survive the grading process (prob goes to next generation)
            self.nRnd = 2
        else:
            self.rnd = np.random.RandomState()
            self.nTrl = 50  # number of trials
            self.nGen = 100  # number of generations
            self.nSurv = 10  # number of best samples that survive the grading process (prob goes to next generation)
            self.nRnd = 10

        self.trial = -1  # current trial
        self.gen = 0  # current generation
        self.noise_vars = []
        self.Gs = None
        self.Gs_kwargs = None
        self.allZ = np.random.randn(self.nTrl, 512)
        self.initialise_network()
        self.nInPool = self.nTrl - self.nSurv - self.nRnd
        self.parents_all = None

    def initialise_network(self):
        network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
        G, D, self.Gs = pretrained_networks.load_networks(network_pkl)
        print("Network was loaded")
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

        truncation_psi = 0.5
        self.Gs_kwargs = dnnlib.EasyDict()
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            self.Gs_kwargs.truncation_psi = truncation_psi

        print("Server initialised - waiting for a client")

    def gen_images(self):
        for tt in range(self.allZ.shape[0]):
            thsTrlPth = participant_dir + 'trl_' + str(self.gen) + "_" + str(tt) + '.png'  # FIXME: remove all globals (e.g. participant_dir)
            print(f'Generating image  {tt} for generation {self.gen} ...')
            z = self.allZ[np.newaxis, tt, :]
            tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
            images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(thsTrlPth)

    def evaluateOneGeneration(self, ratings_, wFitterParent_=0.75, mutAmp=.4, mutP=.3):
        """
        :param wFitterParent_: "power" of the higher graded parent picture
        :param ratings_: last generation grades (list)
        :param mutAmp:  ?
        :param mutP: ?
        """
        thsResp = ratings_

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # transform responses to probabilities (needed for sampling parents)
        thsFitness = softmax(thsResp)

        # take latent vectors of nSurvival highest responses
        thsIndices = np.argsort(thsFitness)
        thsSurv = self.allZ[thsIndices[-self.nSurv:], :]
        parents_survival = [(p, -1, '>') for p in thsIndices[-self.nSurv:]]
        print(f'grades: {thsResp} \n thsFitness: {thsFitness} \n thsIndices {thsIndices}, \n parents_survival: {parents_survival} \n\n')

        # generate recombinations from 2 parent latent vectors of current gen
        # w. fitness proportional to probability of being parent
        # parent with higher fitness contributes more
        thsPool = np.zeros([self.nInPool, self.allZ.shape[1]])

        parents_mixed = []
        for rr in range(self.nInPool):
            thsParents = np.random.choice(self.nTrl, 2, False, thsFitness)

            if thsFitness[thsParents[0]] > thsFitness[thsParents[1]]:
                contrib0 = wFitterParent_
                contrib1 = 1 - wFitterParent_
                contribution_ = '>'
            elif thsFitness[thsParents[0]] < thsFitness[thsParents[1]]:
                contrib0 = 1 - wFitterParent_
                contrib1 = wFitterParent_
                contribution_ = '<'
            elif thsFitness[thsParents[0]] == thsFitness[thsParents[1]]:
                contrib0 = .5
                contrib1 = .5
                contribution_ = '='

            thsPool[rr, :] = self.allZ[thsParents[0], :] * contrib0 + self.allZ[thsParents[1], :] * contrib1
            parents_mixed.append((int(thsParents[0]), int(thsParents[1]), contribution_))

        # each latent dimension of children in recombined pool has some probability of mutation
        toEdit = np.random.choice([0, 1], (self.nInPool, thsPool.shape[1]), True, [1 - mutP, mutP])
        thsEdits = np.random.randn(np.sum(toEdit)) * mutAmp
        thsPool[np.nonzero(toEdit)] = thsPool[np.nonzero(toEdit)] + thsEdits

        # add some random faces to the mix
        thsRnd = np.random.randn(self.nRnd, 512)
        parents_random = [(-1, -1, '=') for _ in range(self.nRnd)]

        # combine direct survivals and recombined / mutated pool
        self.allZ_backup = self.allZ.copy()
        self.allZ = np.concatenate((thsSurv, thsPool, thsRnd), axis=0)
        self.parents_all = np.array(parents_survival + parents_mixed + parents_random)
        # shuffle order of trials
        shuffle_idx = list(range(self.allZ.shape[0]))
        np.random.shuffle(shuffle_idx)
        #self.allZ = self.allZ[shuffle_idx, :]
        #self.parents_all = self.parents_all[shuffle_idx]

    def save_generation_info(self, grades, participant_path):
        with open(participant_path+r'grades.csv', 'a') as fd:
            write = csv.writer(fd)
            write.writerow(grades)
        with open(participant_path+f'allZ_gen_{self.gen}.pkl', 'wb') as fd:
            pickle.dump(self.allZ, fd)

    def save_creation_info(self, participant_path):
        with open(participant_path+r'parents.csv', 'a') as fd:
            write = csv.writer(fd)
            write.writerow(self.parents_all)

    def decode_msg(self, request) -> tuple:
        message = request.decode("utf-8")  # request as a string
        state = message.split(' ')[1]  # get state of the app (True or False)
        rate = message.split(' ')[3]  # get rate from the previous trial (None or 1-9)
        participant_id = message.split(' ')[5]  # get rate from the previous trial (None or 1-9)

        state = state == "True"
        if rate == "None":
            rate = None
        else:
            rate = int(rate)

        return (state, rate, participant_id)  # parsed message

    def check_if_participant_exists(self) -> bool:
        return False

    def reset_session(self):
        pass

    def resume_session(self):
        pass

    def handle_new_generation(self):
        pass

    def handle_next_trial(self):
        pass


gen_rates = []
session = Server()  # initialise class object

while True:
    #  Wait for next request from client
    request = socket.recv()
    print("Received request: %s" % request)  # print request in terminal
    state, rate, participant_id = session.decode_msg(request)
    participant_dir = basePth + f'stimuli/{participant_id}/'
    resume_session = False
    if not os.path.exists(participant_dir):
        os.makedirs(participant_dir)
    else:
        # This should be moved to function
        _, _, filenames = next(walk(participant_dir))
        past_generations = [int(f[9:-4]) for f in filenames if '.pkl' in f]  # 'allZ_gen_290.pkl' -> 9 letters is 'allZ_gen_', -4 is '.pkl'
        if len(past_generations):
            last_full_generation = max(past_generations) if len(past_generations) else -1
            pickle_with_allZ = participant_dir + f'allZ_gen_{last_full_generation}.pkl'
            session.allZ = pickle.load(open(pickle_with_allZ, "rb"))
            resume_session = True
    print(f'resume_session: {resume_session} (load old allZ)')

    print(f'userID: {participant_id}')

    if rate is not None:
        gen_rates.append(rate)

    print(f'current grades: {gen_rates}')

    # start a new experiment
    if state:
        print("Starting a new experiment...")
        if not resume_session:
            session.gen = 0
        else:
            session.gen = last_full_generation + 1
        session.trial = 0
        session.gen_images()
        gen_rates = []

    #  change generation
    elif session.trial == session.nTrl-1:
        session.trial = 0
        session.save_generation_info(gen_rates, participant_dir)
        session.evaluateOneGeneration(gen_rates)
        session.save_creation_info(participant_dir)
        session.gen += 1
        session.gen_images()
        gen_rates = []

    else:
        #  Send reply back to client
        session.trial += 1

    print(f'Trial: {session.trial}')
    socket.send_multipart([bytes([session.trial]), bytes([session.gen])])
