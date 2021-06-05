import numpy as np
import matplotlib.pyplot as plt


class GeneticAlgorithm:
    def __init__(self, nGen_=250, nTrl_=50, nSurv_=10, nRnd_=10):
        self.nTrl = nTrl_  # number of trials in each generations
        self.nGen = nGen_  # number of generations
        self.nSurv = nSurv_  # number of samples surviving from one generations to the other
        self.nRnd = nRnd_   # number of random samples added to each generation
        self.dim = 512    # dimension of a vector ("number of features")
        self.trim_value = 3
        self.ratings = []
        self.distances = []
        self.target_vector = np.random.randn(1, self.dim)
        self.target_vector = np.clip(self.target_vector, -self.trim_value, self.trim_value)
        self.target_vector[0] = self.normalise_vectors(self.target_vector[0])
        self.samples = np.random.randn(self.nTrl, self.dim)
        self.fitness = None
        self.average_distance_for_non_ranom = []

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
     #       print(f'min: {min(scaled_sample)}, max: {max(scaled_sample)}')
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
        self.average_distance_for_non_ranom.append(np.average(np.sort(self.distances)[:self.nTrl - self.nRnd]))
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
        # shuffle order of trials
        #np.random.shuffle(self.samples)


def main():
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
        experiment = GeneticAlgorithm(**params)
        for gen in range(experiment.nGen):
            experiment.rate_one_generation()
            experiment.evaluate_one_generation()
            if gen % 100 == 0 and False:
                print(f'\n Generation: {gen} \n')
                plt.hist(experiment.ratings, bins=10)
                plt.hist(experiment.distances, bins=100)
                plt.title(f"gen {gen} nTrial {experiment.nTrl}, nRand {experiment.nRnd}, nSur {experiment.nSurv}")
                plt.show()
        #print(f'\n target vector: {experiment.target_vector}')
        plt.plot(experiment.average_distance_for_non_ranom)
        mean_error = sum(experiment.average_distance_for_non_ranom) / experiment.nGen
        plt.title(f'averagedistance, nTrial {experiment.nTrl}, nRand {experiment.nRnd}, nSur {experiment.nSurv}, mean_abs_error {mean_error:.2f}')
        plt.show()

        all_mean_errors.append(mean_error)
        print(mean_error, params)




if __name__ == "__main__":
    main()