import numpy as np
import matplotlib.pyplot as plt
import PIL
import sys

basePth = r'/home/edytak/Documents/GAN_project/code/'

TESTING = True
SAVE_RESULTS = True

if SAVE_RESULTS:
    path_stylegan = basePth + r'stylegan2'
    sys.path.append(path_stylegan)
    import pretrained_networks
    import dnnlib
    import dnnlib.tflib as tflib


class GeneticAlgorithm:
    def __init__(self, nGen_=50, nTrl_=50, nSurv_=10, nRnd_=10):
        self.nTrl = nTrl_    # number of trials in each generations
        self.nGen = nGen_    # number of generations
        self.nSurv = nSurv_  # number of samples surviving from one generations to the other
        self.nRnd = nRnd_    # number of random samples added to each generation
        self.dim = 512       # dimension of a vector ("number of features")
        self.trim_value = 3  # threshold for trimming
        self.ratings = []
        self.distances = []
        self.target_vector = np.random.randn(1, self.dim)
        self.target_vector = np.clip(self.target_vector, -self.trim_value, self.trim_value)
        self.target_vector[0] = self.normalise_vectors(self.target_vector[0])  # normalise values to be in 0-10 range
        self.samples = np.random.randn(self.nTrl, self.dim)
        self.fitness = None
        self.average_distance_for_not_random = []
        self.noise_vars = []
        self.rnd = np.random.RandomState()
        self.Gs = None
        self.Gs_kwargs = None
        self.initialise_network()  # initialise network for picture rendering

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
            dist = np.linalg.norm(self.target_vector - scaled_sample)  # similarity between vectors (Euclidean dist.)
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
        self.average_distance_for_not_random.append(np.average(np.sort(self.distances)[:self.nTrl - self.nRnd]))
        self.ratings = rates.copy()

        self.softmax()
     #   print(f'samples: {self.samples}, \n\n rates: {self.ratings}, \nfit: {self.fitness}')

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

    def render_picture(self, gen):
        print("Rendering...")
        simulation_path = basePth + r'simulation/'
        # save target face
        if gen == 0:
            for tt in range(self.target_vector.shape[0]):
                target_path = simulation_path + 'target.png'
                z = self.target_vector[np.newaxis, tt, :]
                tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
                images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
                PIL.Image.fromarray(images[0], 'RGB').save(target_path)
        # save samples
        for tt in range(self.samples.shape[0]):
            samples_path = simulation_path + str(gen) + "_" + str(tt) + '.png'
            z = self.samples[np.newaxis, tt, :]
            tflib.set_vars({var: self.rnd.randn(*var.shape.as_list()) for var in self.noise_vars})  # [height, width]
            images = self.Gs.run(z, None, **self.Gs_kwargs)  # [minibatch, height, width, channel]
            PIL.Image.fromarray(images[0], 'RGB').save(samples_path)


def main():
    if TESTING:
        set_to_test = [{"nTrl_": 5, "nSurv_": 1, "nRnd_": 2}]
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
                print(f'\n Generation: {gen} \n')
                plt.hist(simulation.ratings, bins=10)
                plt.hist(simulation.distances, bins=100)
                plt.title(f"gen {gen} nTrial {simulation.nTrl}, nRand {simulation.nRnd}, nSur {simulation.nSurv}")
                plt.show()
                if SAVE_RESULTS:
                    simulation.render_picture(gen)
        plt.plot(simulation.average_distance_for_not_random)
        mean_error = sum(simulation.average_distance_for_not_random) / simulation.nGen
        plt.title(f'averagedistance, nTrial {simulation.nTrl}, nRand {simulation.nRnd}, nSur {simulation.nSurv}, mean_abs_error {mean_error:.2f}')
        plt.show()

        all_mean_errors.append(mean_error)
        print(mean_error, params)


if __name__ == "__main__":
    main()
