import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys
import datetime
import os
from os import path
import csv
import ast

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
    def __init__(self, nGen_=200, nTrl_=50, nSurv_=10, nRnd_=10):
        self.nTrl = nTrl_    # number of trials in each generations
        self.nGen = nGen_    # number of generations
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
        self.average_distance = None
        self.average_distance_for_not_random = []
        self.noise_vars = []
        self.rnd = np.random.RandomState()
        self.Gs = None
        self.Gs_kwargs = None
        self.initialise_network()  # initialise network for picture rendering
        self.today_simulation_path = None
        self.simulation_path = None
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
            #dist /= self.dim ** 0.5
            #dist = 10 - dist  # lower distance means better rate
            distances.append(dist)
        self.distances = np.array(distances)
        n_highest = np.partition(self.distances, -self.nRnd - 1)[-self.nRnd-1:]
        rates = distances - min(distances)
        rates /= min(n_highest) - min(distances)
        rates *= -10
        rates += 10
        rates = np.clip(rates, 0, 10)
        self.average_distance = np.average(np.sort(self.distances)[:self.nTrl - self.nRnd])
        self.average_distance_for_not_random.append(self.average_distance)
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
        print("Rendering...")
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
            samples_path = self.simulation_path + str(gen) + "_" + str(tt) + '.png'
            z = self.samples[np.newaxis, tt, :]
            tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
            images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(samples_path)

    # create and save plots (distances and rates)
    def plot_distances(self, gen):
        trials = self.nTrl*(gen+1)  # number of already run trials
        fig, ax = plt.subplots()
        ax.hist(self.ratings, bins=10, color='tab:blue')
        ax.hist(self.distances, bins=10, color='tab:orange')
        ax.set_xlabel('Rate (blue), Distance (orange)')
        ax.set_title(f'Trials {trials}, nTrial: {self.nTrl}, nRand: {self.nRnd}, nSur: {self.nSurv}')
        plot_path = self.simulation_path + 'plot_' + str(trials)
        fig.savefig(plot_path + '.png')

    def plot_average_distance(self, mean_error):
        fig, ax = plt.subplots()
        x_values = list(range(0, self.nTrl*self.nGen, self.nTrl)) # show trial number on x-axis
        ax.plot(x_values, self.average_distance_for_not_random)
        ax.set_xlabel('Trials')
        ax.set_title(f'average distance \n nTrial: {self.nTrl}, nRand: {self.nRnd},'
                  f' nSur: {self.nSurv}, mean_abs_error: {mean_error:.2f}')
        fig.savefig(self.simulation_path + 'average_distance.png')

    # save information about each set of tested parameters
    def save_distances(self, gen):
        distances_file = self.today_simulation_path + r'distances.csv'
        info_for_saving = self.nTrl, self.nSurv, self.nRnd, gen, self.average_distance_for_not_random
        with open(distances_file, 'a') as file:
            write = csv.writer(file)
            write.writerow(info_for_saving)

    def plot_multiple_distances(self):
        fig, ax = plt.subplots()
        x_values = list(range(0, self.nTrl * self.nGen, self.nTrl))  # show trial number on x-axis

        with open(self.today_simulation_path + r'distances.csv', 'r') as csvfile:
            plots = csv.reader(csvfile, delimiter=',')

            for row in plots:
                labels = [int(i) for i in row[:3]]
                distances = row[4]  # get distances
                distances_list = ast.literal_eval(distances)  # string to list
                ax.plot(x_values, distances_list, label=labels)

        ax.legend()
        ax.set_xlabel('Trials')
        ax.set_ylabel('Average distance')
        ax.set_title('Average distance over trials - comparison')
        fig.savefig(self.today_simulation_path + 'comparison.png')


def main():
    if TESTING:
        set_to_test = [{"nTrl_": 30, "nSurv_": 5, "nRnd_": 5},
                       {"nTrl_": 40, "nSurv_": 5, "nRnd_": 5},
                       {"nTrl_": 50, "nSurv_": 5, "nRnd_": 5},
                       {"nTrl_": 60, "nSurv_": 5, "nRnd_": 5}]
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
        for gen in range(simulation.nGen):
            simulation.rate_one_generation()
            simulation.evaluate_one_generation()
            if gen % 10 == 0:
                print(f'\n Trials: {simulation.nTrl*(gen+1)} \n Generation: {gen} \n')
                if SAVE_RESULTS:
                    simulation.render_pictures(gen)
                    simulation.plot_distances(gen)
        simulation.save_distances(gen)
        mean_error = sum(simulation.average_distance_for_not_random) / simulation.nGen
        if SAVE_RESULTS:
            simulation.plot_average_distance(mean_error)

        all_mean_errors.append(mean_error)
       # print(mean_error, params)
    #    print(f'\n Distances: {simulation.distances}, \n\n Rates: {simulation.ratings}')
    simulation.plot_multiple_distances()

if __name__ == "__main__":
    main()
