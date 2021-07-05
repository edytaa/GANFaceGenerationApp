import numpy as np
import zmq
import PIL
import sys
import csv
import os
import pickle
from os import walk
import threading
from random import randint
from datetime import datetime
from collections import defaultdict

basePth = r'/home/edytak/Documents/GAN_project/code/'
path_stylegan = basePth + r'stylegan2'
sys.path.append(path_stylegan)

import pretrained_networks
import dnnlib
import dnnlib.tflib as tflib
import tensorflow as tf
from tensorflow.python.keras.backend import set_session

TESTING = True
NON_FUNCTIONAL_GRADE = 99
active_thread = False

context = zmq.Context()
context = zmq.Context()
socket = context.socket(zmq.REP)
server_id = sys.argv[1]
address = r'tcp://*:' + str(server_id)
socket.bind(address)


class SessionInfo:
    def __init__(self, gen=0, trial=-1, allZ=None):
        self.current_generation = gen
        self.current_trial = trial
        self.last_update = datetime.now()
        self.rates = []
        self.allZ = allZ
        self.pictures_generated = False
        self.pictures_generated_info_send = False
        self.n_generated_samples = 0

    def get_age_in_min(self):
        return (datetime.now() - self.last_update).total_seconds() / 60

class Server:
    def __init__(self):
        if TESTING:
            print('TESTING MODE \n')
            seed = 3004
            self.rnd = np.random.RandomState(seed)
            self.nTrl = 6  # number of trials per generation
            self.nGen = 10  # number of generations
            self.nSurv = 2  # number of best samples that survive the grading process (prob goes to next generation)
            self.nRnd = 2
        else:
            self.rnd = np.random.RandomState()
            self.nTrl = 55  # number of trials
            self.nGen = 100  # number of generations
            self.nSurv = 7  # number of best samples that survive the grading process (prob goes to next generation)
            self.nRnd = 7

        self.noise_vars = []
        self.Gs = None
        self.Gs_kwargs = None
        self.session = tflib.init_tf()
        self.graph = tf.get_default_graph()
        with self.graph.as_default():
            set_session(self.session)
            print('init', tf.get_default_session())
            self.initialise_network()
        self.nInPool = self.nTrl - self.nSurv - self.nRnd
        self.parents_info = None

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

    def gen_images(self, participant_id, participant_info: SessionInfo):
        gen = participant_info.current_generation
        allZ = participant_info.allZ
        print('gen_images', tf.get_default_session())
        participant_dir = self.get_participant_path(participant_id)
        with open(participant_dir+f'allZ_gen_{gen}.pkl', 'wb') as fd:
            pickle.dump(allZ, fd)
        for tt in range(allZ.shape[0]):
            refresher()
            thsTrlPth = participant_dir + 'trl_' + str(gen) + "_" + str(tt) + '.png'
            # check if image exists; if not: generate image
            if not os.path.isfile(thsTrlPth):
                print(f'Generating image  {tt} for generation {gen} ...')
                z = allZ[np.newaxis, tt, :]
                tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
                images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
                PIL.Image.fromarray(images[0], 'RGB').save(thsTrlPth)
                participant_info.n_generated_samples += 1
            else:
                pass
        participant_info.pictures_generated = True
        refresher()
        global active_thread
        active_thread = False


    def evaluateOneGeneration(self, participant_info: SessionInfo, wFitterParent_=0.75, mutAmp=.4, mutP=.3):
        """
        :param wFitterParent_: "power" of the higher graded parent picture
        :param ratings_: last generation grades (list)
        :param mutAmp:  ?
        :param mutP: ?
        """
        thsResp = participant_info.rates
        allZ = participant_info.allZ

        def softmax(x):
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        # transform responses to probabilities (needed for sampling parents)
        thsFitness = softmax(thsResp)

        # take latent vectors of nSurvival highest responses
        thsIndices = np.argsort(thsFitness)
        thsSurv = allZ[thsIndices[-self.nSurv:], :]
        parents_survival = [(p, -1, '>') for p in thsIndices[-self.nSurv:]]
        print(f'grades: {thsResp} \n thsFitness: {thsFitness} \n thsIndices {thsIndices}, \n parents_survival: {parents_survival}, nTrl: {self.nTrl} \n\n')

        # generate recombination from 2 parent latent vectors of current gen
        # w. fitness proportional to probability of being parent
        # parent with higher fitness contributes more
        thsPool = np.zeros([self.nInPool, allZ.shape[1]])

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

            thsPool[rr, :] = allZ[thsParents[0], :] * contrib0 + allZ[thsParents[1], :] * contrib1
            parents_mixed.append((int(thsParents[0]), int(thsParents[1]), contribution_))

        # each latent dimension of children in recombined pool has some probability of mutation
        toEdit = np.random.choice([0, 1], (self.nInPool, thsPool.shape[1]), True, [1 - mutP, mutP])
        thsEdits = np.random.randn(np.sum(toEdit)) * mutAmp
        thsPool[np.nonzero(toEdit)] = thsPool[np.nonzero(toEdit)] + thsEdits

        # add some random faces to the mix
        thsRnd = np.random.randn(self.nRnd, 512)
        # mix random samples with survivors (parents)
        for v in range(len(thsRnd)):
            parent = np.random.choice(self.nTrl, 1, False, thsFitness)
            contrib_parent = .25  # contribution of parent sample
            contrib_random = .75  # contribution of random sample
            thsRnd[v, :] = allZ[parent[0], :] * contrib_parent + thsRnd[v, :] * contrib_random
        parents_random = [(-1, -1, '=') for _ in range(self.nRnd)]

        # combine direct survivals and recombined / mutated pool
        allZ = np.concatenate((thsSurv, thsPool, thsRnd), axis=0)
        #self.parents_info = np.array(parents_survival + parents_mixed + parents_random)
        # shuffle order of trials
        shuffle_idx = list(range(allZ.shape[0]))
        np.random.shuffle(shuffle_idx)
        allZ = allZ[shuffle_idx, :]
        #self.parents_info = self.parents_info[shuffle_idx]
        participant_info.allZ = allZ

    @staticmethod
    def save_generation_info(participant_id, participant_info: SessionInfo):
        participant_path = Server.get_participant_path(participant_id)
        with open(participant_path+r'grades.csv', 'a') as fd:
            write = csv.writer(fd)
            write.writerow(participant_info.rates)

    def save_creation_info(self, participant_id):
        participant_path = Server.get_participant_path(participant_id)
        with open(participant_path+r'parents.csv', 'a') as fd:
            write = csv.writer(fd)
            write.writerow(self.parents_info)

    @staticmethod
    def decode_msg(request) -> tuple:
        message = request.decode("utf-8")  # request as a string
        rate = message.split(' ')[1]  # get rate from the previous trial (None or 1-9)
        participant_id = message.split(' ')[3]  # get rate from the previous trial (None or 1-9)

        if rate == "None":
            rate = None
        else:
            rate = int(rate)

        return (rate, participant_id)  # parsed message

    @staticmethod
    def get_participant_path(participant_id):
        global basePth
        return basePth + f'stimuli/{participant_id}/'

    @staticmethod
    def check_if_participant_exists(participant_id) -> bool:
        participant_dir = Server.get_participant_path(participant_id)
        return os.path.exists(participant_dir)

    def new_participant(self, participant_id, generate_folder=True):
        if generate_folder:
            participant_dir = Server.get_participant_path(participant_id)
            os.makedirs(participant_dir)
        gen = 0
        trial = 0
        allZ = np.random.randn(self.nTrl, 512)
        return gen, trial, allZ

    def resume_session(self, participant_id):
        participant_dir = Server.get_participant_path(participant_id)
        _, _, filenames = next(walk(participant_dir))
        past_generations = [int(f[9:-4]) for f in filenames if
                            '.pkl' in f]  # 'allZ_gen_290.pkl' -> 9 first letters is 'allZ_gen_', -4 is '.pkl'
        if len(past_generations):
            last_full_generation = max(past_generations)
            pickle_with_allZ = participant_dir + f'allZ_gen_{last_full_generation}.pkl'
            allZ = pickle.load(open(pickle_with_allZ, "rb"))
            trial = 0
            gen = last_full_generation
            print("Resume session -> found old generation")
        else:  # there is a folder but empty
            gen, trial, allZ = self.new_participant(participant_id, generate_folder=False)
        return gen, trial, allZ

    def switch_generations(self, participant_id, participant_info: SessionInfo, session, graph):
        print(f'switch_generations: id {participant_id}, n_rates: {len(participant_info.rates)}')
        participant_info.n_generated_samples = 0
        Server.save_generation_info(participant_id, participant_info)
        self.evaluateOneGeneration(participant_info)
        participant_info.current_trial = 0
        participant_info.current_generation += 1
        participant_info.rates = []
        self.parents_info = [] # FIXME - remove all parents info
        with graph.as_default():
            with session.as_default():
                self.gen_images(participant_id, participant_info)

    @staticmethod
    def switch_trials(participant_info: SessionInfo):
        participant_info.current_trial += 1


def refresher():
    # https://stackoverflow.com/questions/62718133/how-to-make-streamlit-reloads-every-5-seconds
    mainDir = os.path.dirname(__file__)
    filePath = os.path.join(mainDir, 'dummy.py')
    with open(filePath, 'w') as f:
        f.write(f'# {randint(0, 100000)}')



def main():
    session = Server()  # initialise class object
    active_sessions = defaultdict(SessionInfo)
    thread_queue = []  # list of threads waiting to be initialised
    global active_thread
    while True:
        #  Wait for next request from client
        request = socket.recv()
        rate, participant_id_ = session.decode_msg(request)
        if rate != NON_FUNCTIONAL_GRADE:
            print(f'\n###########\nreceived request with: rate {rate}, participant_id {participant_id_}')

        # new / different participant
        if participant_id_ != '0' and participant_id_ not in active_sessions:
            print("Activate new session")
            if not session.check_if_participant_exists(participant_id_):
                gen_l, trial_l, allZ_l = session.new_participant(participant_id_)  # initialise a new participant
            else:
                gen_l, trial_l, allZ_l = session.resume_session(participant_id_)  # resume paused session
            active_sessions[participant_id_] = SessionInfo(gen_l, trial_l, allZ_l)

            session.gen_images(participant_id_, active_sessions[participant_id_])
            active_sessions[participant_id_].rates = []
            trial_response = active_sessions[participant_id_].current_trial
            gen_response = active_sessions[participant_id_].current_generation
            active_sessions[participant_id_].pictures_generated_info_send = False

        elif participant_id_ != '0':
            print(f"Continue session (id: {participant_id_}, gen: {active_sessions[participant_id_].current_generation}, "
                  f"trail: {active_sessions[participant_id_].current_trial})")
            if active_sessions[participant_id_].pictures_generated:
                # Add rating
                if active_sessions[participant_id_].pictures_generated_info_send and rate != NON_FUNCTIONAL_GRADE:
                    active_sessions[participant_id_].rates.append(rate)

                # switch between image generator,
                if active_sessions[participant_id_].current_trial == session.nTrl - 1 and\
                        len(active_sessions[participant_id_].rates) == session.nTrl:
                    active_sessions[participant_id_].pictures_generated = False  # new samples not ready (part of threading)
                    active_sessions[participant_id_].pictures_generated_info_send = False
                    thread = threading.Thread(target=session.switch_generations,
                                              args=(participant_id_, active_sessions[participant_id_],
                                                    session.session, session.graph))
                    if not active_thread:
                        active_thread = True
                        thread.start()
                    else:
                        print(f"Add new task to waiting list; id {participant_id_}, n_rates: {len(active_sessions[participant_id_].rates)}")
                        thread_queue.append(thread)
                elif active_sessions[participant_id_].pictures_generated_info_send and rate != NON_FUNCTIONAL_GRADE:  # TODO: check if NON_FUNC can go to higher level
                    session.switch_trials(active_sessions[participant_id_])
                    active_sessions[participant_id_].pictures_generated_info_send = True
                else:
                    active_sessions[participant_id_].pictures_generated_info_send = True
                trial_response = active_sessions[participant_id_].current_trial
            else:
                trial_response = active_sessions[participant_id_].n_generated_samples

            if not active_thread and len(thread_queue):
                print(f'number of waiting threads: {len(thread_queue)}')
                thred_loc = thread_queue.pop(0)
                active_thread = True
                thred_loc.start()

            gen_response = active_sessions[participant_id_].current_generation
            n_generated_samples = active_sessions[participant_id_].n_generated_samples
            samples_ready = active_sessions[participant_id_].pictures_generated
            samples_ready_information_sent = active_sessions[participant_id_].pictures_generated_info_send
        else:
            trial_response, gen_response, n_generated_samples = 0, 0, 0
            samples_ready, samples_ready_information_sent = False, False

        if rate != NON_FUNCTIONAL_GRADE:
            print(f'info send: trial: {trial_response}, gen: {gen_response}, new generation ready?: {samples_ready}, info_send: {samples_ready_information_sent}')
        socket.send_multipart([bytes([trial_response]), bytes([gen_response]),
                               bytes(str(samples_ready), 'utf-8'), bytes([n_generated_samples])])


if __name__ == "__main__":
    main()
