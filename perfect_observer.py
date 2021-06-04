import numpy as np


class GeneticAlgorithm:
    def __init__(self):
        self.nTrl = 10
        self.nGen = 10
        self.nSurv = 2
        self.nRnd = 2
        self.dim = 3
        self.ratings = []
        self.target_vector = np.random.randn(1, self.dim)
        self.samples = np.random.randn(self.nTrl, self.dim)
        self.fitness = None

    def rate_one_generation(self):
        self.ratings = []
        for i in range(len(self.samples)):
            dist = np.linalg.norm(self.target_vector - self.samples[i])
            self.ratings.append(dist)
        self.softmax()
        print(f'samples: {self.samples}, \n\n rates: {self.ratings}, \nfit: {self.fitness}')

    def softmax(self):
        e_x = np.exp(self.ratings - np.max(self.ratings))
        self.fitness = e_x / e_x.sum()

    def evaluate_one_generation(self, wFitterParent=0.75, mutAmp=.4, mutP=.3):
        # take latent vectors of nSurvival highest responses
        thsIndices = np.argsort(self.fitness)
        thsSurv = self.samples[thsIndices[-self.nSurv:], :]

        nInPool = self.nTrl - self.nSurv - self.nRnd
        thsPool = np.zeros([nInPool, self.samples.shape[1]])

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

        # add some random faces to the mix
        thsRnd = np.random.randn(self.nRnd, self.dim)

        # combine direct survivals and recombined / mutated pool
        self.samples = np.concatenate((thsSurv, thsPool, thsRnd), axis=0)
        # shuffle order of trials
        #np.random.shuffle(self.samples)


def main():
    experiment = GeneticAlgorithm()
    print(f'target vector: {experiment.target_vector}')
    for i in range(experiment.nGen):
        print(f'\n Generation: {i} \n')
        experiment.rate_one_generation()
        experiment.evaluate_one_generation()

if __name__ == "__main__":
    main()