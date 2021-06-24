import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys
import datetime
import os
from os import path
import math
import pandas as pd
import seaborn as sns

basePth = r'/home/edytak/Documents/GAN_project/code/'

SAVE_RESULTS = True  # if true, render and safe images

if SAVE_RESULTS:
    path_stylegan = basePth + r'stylegan2'
    sys.path.append(path_stylegan)
    import pretrained_networks
    import dnnlib
    import dnnlib.tflib as tflib


class GeneticAlgorithm:
    def __init__(self, allTrl_=5000, nTrl_=50, nSurv_=10, nRnd_=10):
        self.allTrl = allTrl_
        self.nTrl = nTrl_    # number of trials in each generations
        self.nGen = math.floor(allTrl_ / nTrl_)   # number of (full) generations
        self.nSurv = nSurv_  # number of samples surviving from one generations to the other
        self.nRnd = nRnd_    # number of random samples added to each generation
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
        self.get_simulation_path()

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
        # np.random.shuffle(self.samples)  # shuffle order of trials

    # initialise network for picture rendering 
    def initialise_network(self):
        network_pkl = 'gdrive:networks/stylegan2-ffhq-config-f.pkl'
        G, D, self.Gs = pretrained_networks.load_networks(network_pkl)
        self.noise_vars = [var for name, var in self.Gs.components.synthesis.vars.items() if name.startswith('noise')]

        truncation_psi = 0.5
        self.Gs_kwargs = dnnlib.EasyDict()
        self.Gs_kwargs.output_transform = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        self.Gs_kwargs.randomize_noise = False
        if truncation_psi is not None:
            self.Gs_kwargs.truncation_psi = truncation_psi

    # get path to a folder for current simulation
    def get_simulation_path(self):
        date = str(datetime.date.today())  # use date for naming dir
        self.today_simulation_path = basePth + r'simulation/' + date + r'/'

    # plot distribution of distances for tested sets of parameters
    def plot_best_samples(self, data_frame):
        if not path.isdir(self.today_simulation_path):  # create a folder for plots (if it is not already there)
            os.makedirs(self.today_simulation_path)
        fig, ax = plt.subplots()
        ax = sns.swarmplot(x='set', y='distance', data=data_frame)
        plt.grid()
        ax.set_xlabel('Sets')
        ax.set_ylabel('Average distance')
        ax.set_title('Distribution of best distances for different sets of parameters')
        fig.savefig(self.today_simulation_path + 'best_distances.png')
        plt.close(fig)


def main():
    set_to_test = [{"nTrl_": 50, "nSurv_": 10, "nRnd_": 10},
                   {"nTrl_": 40, "nSurv_": 5, "nRnd_": 5}
                   ]
    n_simulations = 10  # number of simulation run with a given set of parameters
    best_distance = []
    x_labels = []
    df_distances = pd.DataFrame({'distance': best_distance, 'set': x_labels})
    for params in set_to_test:
        for sim in range(n_simulations):
            simulation = GeneticAlgorithm(**params)
            for gen in range(simulation.nGen):
                simulation.rate_one_generation()
                simulation.evaluate_one_generation()
            print(f'Simulation number: {sim} \n')
            best_distance = min(simulation.distances)  # smallest distance -> best sample
            x_labels = str(list(params.values()))  # testes values as labels on x-axis
            new_row = {'distance': best_distance, 'set': x_labels}  # new entry to data frame
            df_distances = df_distances.append(new_row, ignore_index=True)

    simulation.plot_best_samples(df_distances)  # create a plot


if __name__ == "__main__":
    main()
