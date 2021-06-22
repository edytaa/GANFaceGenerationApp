import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys
import datetime
import os
from os import path
import csv
import ast
import math
import pickle
import pandas as pd


basePth = r'/home/edytak/Documents/GAN_project/code/'

TESTING = True  # if true, use only one set of parameters
SAVE_RESULTS = True  # if true, render and safe images

if SAVE_RESULTS:
    path_stylegan = basePth + r'stylegan2'
    sys.path.append(path_stylegan)
    import pretrained_networks
    import dnnlib
    import dnnlib.tflib as tflib


class GeneticAlgorithm:
    def __init__(self, allTrl_=10000, nTrl_=50, nSurv_=10, nRnd_=10, set_number_ = 1):
        self.allTrl = allTrl_
        self.nTrl = nTrl_    # number of trials in each generations
        self.nGen = math.floor(allTrl_ / nTrl_)   # number of (full) generations
        self.nSurv = nSurv_  # number of samples surviving from one generations to the other
        self.nRnd = nRnd_    # number of random samples added to each generation
        self.set_number = set_number_  # number of sets of parameters for simulation
        self.completed_trials = None   # number of completed trials
        self.dim = 512       # dimension of a vector ("number of features")
        self.trim_value = 3  # threshold for trimming
        self.ratings = []
        self.distances = []
        self.target_vector = np.random.randn(1, self.dim)
        self.target_vector_norm = np.clip(self.target_vector, -self.trim_value, self.trim_value)
        self.target_vector_norm[0] = self.normalise_vectors(self.target_vector_norm[0])  # normalise values to be in 0-10 range
        self.samples = np.random.randn(self.nTrl, self.dim)
        self.fitness = None
        self.average_distance_for_not_random = []
        self.noise_vars = []
        self.rnd = np.random.RandomState()
        self.Gs = None
        self.Gs_kwargs = None
        self.initialise_network()  # initialise network for picture rendering
        self.today_simulation_path = None
        self.simulation_path = None
        self.get_simulation_path()
        self.distances_plotting = []
        self.average_distance_plotting = []

    # restrict values between 0 and 10
    def normalise_vectors(self, sample):
        norm_vector = sample.copy()
        range_ = 2 * self.trim_value
        norm_vector = (norm_vector + self.trim_value) / range_
        return norm_vector

    # calculate ratings for a given generation
    def rate_one_generation(self):
        self.ratings = []
        distances = []
        for sample in self.samples:
            trimmed_sample = np.clip(sample, -3, 3)
            scaled_sample = self.normalise_vectors(trimmed_sample)
            dist = np.linalg.norm(self.target_vector_norm - scaled_sample)  # similarity between vectors (Euclidean dist.)
            distances.append(dist)
            self.distances_plotting.append(dist)  # collect distance for each sample
        self.distances = np.array(distances)
        n_highest = np.partition(self.distances, -self.nRnd - 1)[-self.nRnd-1:]  # choose only the best vectors
        # convert distances to rates
        rates = distances - min(distances)
        rates /= min(n_highest) - min(distances)
        rates *= -10
        rates += 10
        rates = np.clip(rates, 0, 10)
        self.average_distance_for_not_random.append(np.average(np.sort(self.distances)[:self.nTrl - self.nRnd]))
        self.ratings = rates.copy()
        self.softmax()

    # transform ratings to probabilities (needed for sampling parents)
    def softmax(self):
        e_x = np.exp(self.ratings - np.max(self.ratings))
        self.fitness = e_x / e_x.sum()

    # main part of genetic algorithm (selection, crossover, mutation)
    def evaluate_one_generation(self, wFitterParent=0.75, mutAmp=.4, mutP=.3):
        # take latent vectors of nSurvival highest responses
        thsIndices = np.argsort(self.fitness)
        thsSurv = self.samples[thsIndices[-self.nSurv:], :]

        nInPool = self.nTrl - self.nSurv - self.nRnd  # number of samples that will result from mutation
        thsPool = np.zeros([nInPool, self.samples.shape[1]])

        # generate recombination from 2 parent latent vectors of current gen w. fitness proportional
        # to probability of being parent (parent with higher fitness contributes more)
        for rr in range(nInPool):
            thsParents = np.random.choice(self.nTrl, 2, False, self.fitness)

            if self.fitness[thsParents[0]] > self.fitness[thsParents[1]]:
                contrib0 = wFitterParent
                contrib1 = 1 - wFitterParent
            elif self.fitness[thsParents[0]] < self.fitness[thsParents[1]]:
                contrib0 = 1 - wFitterParent
                contrib1 = wFitterParent
            elif self.fitness[thsParents[0]] == self.fitness[thsParents[1]]:
                contrib0 = .5
                contrib1 = .5

            thsPool[rr, :] = self.samples[thsParents[0], :] * contrib0 + self.samples[thsParents[1], :] * contrib1

        # each latent dimension of children in recombined pool has some probability of mutation
        toEdit = np.random.choice([0, 1], (nInPool, thsPool.shape[1]), True, [1 - mutP, mutP])
        thsEdits = np.random.randn(np.sum(toEdit)) * mutAmp
        thsPool[np.nonzero(toEdit)] = thsPool[np.nonzero(toEdit)] + thsEdits

        # add some random vectors
        thsRnd = np.random.randn(self.nRnd, self.dim)

        # combine direct survivals and recombined / mutated pool
        self.samples = np.concatenate((thsSurv, thsPool, thsRnd), axis=0)
        #np.random.shuffle(self.samples)  # shuffle order of trials

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

    def get_simulation_path(self):
        date = str(datetime.date.today())  # use date for naming dir
        params_list = [self.nTrl, self.nSurv, self.nRnd]
        self.today_simulation_path = basePth + r'simulation/' + date + r'/'
        self.simulation_path = self.today_simulation_path + str(params_list) + r'/'

    def render_pictures(self, gen):
        print("Rendering images...")
        # create folder and save target face
        if gen == 0:
            if not path.isdir(self.simulation_path):
                os.makedirs(self.simulation_path)
            for tt in range(self.target_vector.shape[0]):
                target_path = self.simulation_path + 'target.png'
                z = self.target_vector[np.newaxis, tt, :]
                tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
                images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
                PIL.Image.fromarray(images[0], 'RGB').save(target_path)

        # save samples (only survivors )
        for tt in range(self.nSurv):
            samples_path = self.simulation_path + str(self.completed_trials) + "_" + str(tt) + '.png'
            z = self.samples[np.newaxis, tt, :]
            tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
            images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(samples_path)

    # create and save plots (distances and rates)
    def plot_distances(self):
        fig, ax = plt.subplots()
        ax.hist(self.ratings, bins=10, color='tab:blue')
        ax.hist(self.distances, bins=10, color='tab:orange')
        ax.set_xlabel('Rate (blue), Distance (orange)')
        ax.set_title(f'Trials {self.completed_trials}, nTrial: {self.nTrl}, nRand: {self.nRnd}, nSur: {self.nSurv}')
        plot_path = self.simulation_path + 'plot_' + str(self.completed_trials)
        fig.savefig(plot_path + '.png')
        plt.close(fig)

    def plot_average_distance(self, mean_error):
        fig, ax = plt.subplots()
        x_values = np.linspace(0, self.allTrl, self.nGen)  # show trial number on x-axis
        ax.plot(x_values, self.average_distance_for_not_random)
        ax.grid()
        ax.set_xlabel('Trials')
        ax.set_title(f'average distance \n nTrial: {self.nTrl}, nRand: {self.nRnd},'
                  f' nSur: {self.nSurv}, mean_abs_error: {mean_error:.2f}')
        fig.savefig(self.simulation_path + 'average_distance.png')
        plt.close(fig)

    def moving_average(self, w=100):
        return np.convolve(self.distances_plotting, np.ones(w), 'valid') / w

    # save information about each set of tested parameters
    def save_distances(self, gen):
        distances_file = self.today_simulation_path + r'distances_' + str(self.set_number) + r'.p'
        average_distance_plotting = self.moving_average(600)
        info_for_saving_pickle = {"nTrl": self.nTrl, "nSurv": self.nSurv, "nRnd": self.nRnd,
                                  "gen": gen, "dist": average_distance_plotting}
        pickle.dump(info_for_saving_pickle, open(distances_file, "wb"))

    # plot average distances over trials (for all sets of parameters)
    def plot_multiple_distances(self, number_of_sets):
        fig, ax = plt.subplots()
        for set in range(number_of_sets):
            distances_file = self.today_simulation_path + r'distances_' + str(set) + r'.p'
            info = pickle.load(open(distances_file, "rb"))
            distances = info.get('dist')
            labels = str(info.get('nTrl')) + '-' + str(info.get('nSurv')) + '-' + str(info.get('nRnd'))
            ax.plot(distances,  label=labels)
            ax.grid()
            ax.legend()
            ax.set_xlabel('Trials')
            ax.set_ylabel('Average distance')
            ax.set_title('Average distance over trials - comparison')
            fig.savefig(self.today_simulation_path + 'comparison.png')

        ax.set_xlabel('Trials')
        ax.set_ylabel('Average distance')
        ax.set_title('Average distance over trials - comparison')
        fig.savefig(self.today_simulation_path + 'comparison.png')
        plt.close(fig)

def main():
    if TESTING:
        set_to_test = [{"nTrl_": 40, "nSurv_": 6, "nRnd_": 6, "set_number_": 0},
                       {"nTrl_": 50, "nSurv_": 7, "nRnd_": 7, "set_number_": 1},
                       {"nTrl_": 60, "nSurv_": 9, "nRnd_": 9, "set_number_": 2}
                       ]
    else:
        set_to_test = [{"nTrl_": 50, "nSurv_": 1, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 5, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 10, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 15, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 20, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 25, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 30, "nRnd_": 10},
                       {"nTrl_": 50, "nSurv_": 1, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 5, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 10, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 15, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 20, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 25, "nRnd_": 1},
                       {"nTrl_": 50, "nSurv_": 30, "nRnd_": 1},
                       {"nTrl_": 100, "nSurv_": 1, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 5, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 10, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 15, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 20, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 25, "nRnd_": 10},
                       {"nTrl_": 100, "nSurv_": 30, "nRnd_": 10},
                       ]
    all_mean_errors = []
    for params in set_to_test:
        simulation = GeneticAlgorithm(**params)
        print(f'Number of generations: {simulation.nGen}')
        for gen in range(simulation.nGen):
            simulation.rate_one_generation()
            simulation.evaluate_one_generation()
            simulation.completed_trials = simulation.nTrl * (gen + 1)  # number of completed trials
            if gen % 10 == 0:
                print(f'\n Trials: {simulation.completed_trials} \n Generation: {gen} \n')
                if SAVE_RESULTS:
                    simulation.render_pictures(gen)
                    simulation.plot_distances()
        simulation.save_distances(gen)  # save info for a finished simulation
        mean_error = sum(simulation.average_distance_for_not_random) / simulation.nGen
        if SAVE_RESULTS:
            simulation.plot_average_distance(mean_error)
        all_mean_errors.append(mean_error)
       # print(mean_error, params)
    simulation.plot_multiple_distances(len(set_to_test))  # plot average distances over trials (all sets of parameters)


if __name__ == "__main__":
    main()
