from model import LogisticRegression
import os
import numpy as np
import pickle as pkl


class GA(object):
    def __init__(self,
                 X_train,
                 Y_train,
                 X_test,
                 Y_test,
                 populationSize = 100,
                 crossoverProb = 0.2,
                 mutationProb = 0.1,
                 iteration = 100):
        '''
        genetic algorithm (GA)

        :param X_train:
        :param Y_train:
        :param X_test:
        :param Y_test:
        :param populationSize:
        :param crossoverProb:
        :param mutationProb:
        :param iteration:
        '''

        self.X_train = X_train
        self.Y_train = Y_train
        self.X_test = X_test
        self.Y_test = Y_test

        self.best_probsList = []
        self.best_factorsList = []
        self.best_probs = None
        self.best_factors= None

        self.num = X_train.shape[1]
        self.populationSize = populationSize
        self.crossoverProb = crossoverProb
        self.mutationProb = mutationProb
        self.iteraton = iteration

        self.up = self.num - self.populationSize // 2
        self.down = self.populationSize // 2

    def newPopulation(self):
        '''
        initailize population

        :return:
        '''
        print("  Begin newPopulation")
        population = self._generate_random()

        prob, _, _ = self.Fitness(population)
        self.best_probsList.append(np.max(prob))
        self.best_factorsList.append(population[np.argmax(prob), :])
        return population

    def _generate_random(self):
        num = np.random.randint(self.down, self.up, self.populationSize)
        chromosomes = None
        for i in num:
            chromosome = np.concatenate((np.ones(i), np.zeros(self.num - i)))
            permutation = np.random.permutation(self.num)
            chromosome = chromosome[permutation]
            if chromosomes is None:
                chromosomes = chromosome.reshape(1, -1)
            else:
                chromosomes = np.concatenate((chromosomes, chromosome.reshape(1, -1)))
        return chromosomes

    def CheckChromosome(self, chromosome):
        '''
        check the chromosome whether satisfies condition

        :param chromosome:
        :return:
        '''
        if self.down <= np.sum(chromosome) <= self.up:
            return True
        else:
            return False


    def Selection(self, init_population, cum_fitness):
        '''
        selection

        :param init_population:
        :param cum_fitness:
        :return:
        '''
        print("  Begin Selection")
        sel_population = np.zeros_like(init_population)
        for i, randomnum in enumerate(np.random.rand(self.populationSize)):
            print("    Selection", i)
            index = np.where(cum_fitness >= randomnum)
            sel_population[i, :] = init_population[index[0][0], :]
        return sel_population


    def Crossover(self, init_population, population):
        '''
        crossover

        :param init_population:
        :param population:
        :return:
        '''
        print("  Begin Crossover")
        is_crossover = 1 - np.random.binomial(1, self.crossoverProb, self.populationSize // 2)
        for i in np.where(is_crossover)[0]:
            print("    Crossover", i)
            while True:
                kind_crossover = np.random.randint(0, 3)
                if kind_crossover == 0:
                    chromosome1 = np.copy(population[2*i, :])
                    chromosome2 = np.copy(population[2*i+1, :])
                    where_crossover = np.random.randint(0, self.num, 1)
                    chromosome1[where_crossover] = np.copy(init_population[2*i+1, where_crossover])
                    chromosome2[where_crossover] = np.copy(init_population[2*i, where_crossover])
                elif kind_crossover == 1:
                    chromosome1 = np.copy(population[2*i, :])
                    chromosome2 = np.copy(population[2*i+1, :])
                    where_crossover = np.random.randint(1, self.num // 2, 1)[0]
                    chromosome1[:where_crossover] = np.copy(init_population[2*i+1, :where_crossover])
                    chromosome2[:where_crossover] = np.copy(init_population[2*i, :where_crossover])
                else:
                    chromosome1 = np.copy(population[2*i, :])
                    chromosome2 = np.copy(population[2*i+1, :])
                    num_crossover = np.random.randint(0, self.num // 2)
                    where_crossover = np.random.randint(0, self.num, num_crossover)
                    chromosome1[where_crossover] = np.copy(init_population[2*i+1, where_crossover])
                    chromosome2[where_crossover] = np.copy(init_population[2*i, where_crossover])

                if self.CheckChromosome(chromosome1) and self.CheckChromosome(chromosome2):
                    population[[2*i, 2*i+1], :] = np.array([chromosome1, chromosome2])
                    break
        return population

    def Mutation(self, population):
        '''


        :param population:
        :return:
        '''
        print("  Begin Mutation")
        is_mutate = 1 - np.random.binomial(1, self.mutationProb, self.populationSize)
        for i in np.where(is_mutate)[0]:
            print("    Mutation", i)
            while True:
                chromosome = np.copy(population[i, :])
                where_mutate = np.random.randint(0, self.num, 1)
                chromosome[where_mutate] =  1 - chromosome[where_mutate]
                if self.CheckChromosome(chromosome):
                    population[i,:] = chromosome
                    break
        return population

    def Fitness(self, population):
        '''


        :param population:
        :return:
        '''
        print("  Begin Fitness")
        fitness = np.zeros(self.populationSize)
        for i in range(self.populationSize):
            print("    Fitness", i)
            X_train_mark = self.X_train[:, population[i,:] == 1]
            X_test_mark = self.X_test[:, population[i,:] == 1]
            LR = LogisticRegression(X_train_mark,
                                    self.Y_train,
                                    X_test_mark,
                                    self.Y_test)
            fitness[i] = LR.evalution(is_GA=True)

        prob = fitness / np.sum(fitness)
        cum_prob = np.cumsum(prob)
        return prob, cum_prob, fitness


    def Start(self):
        '''
        start GA

        :return:
        '''
        init_population = self.newPopulation()
        _, cum_fitness, _ = self.Fitness(init_population)

        for itera in range(self.iteraton):
            print("Iteration of Genestic Algorithm: %dth" % itera)
            sel_population = self.Selection(init_population, cum_fitness)
            cro_population = self.Crossover(init_population, sel_population)
            mut_population = self.Mutation(cro_population)

            prob, cum_fitness, fitness = self.Fitness(mut_population)
            init_population = mut_population

            self.best_probsList.append(np.max(prob))
            ans = mut_population[np.argmax(prob), :]
            self.best_factorsList.append(ans)
            print("\n", sum(ans))
            print(ans, "\n")

        self.best_probs = np.max(self.best_probsList)
        self.best_factors = self.best_factorsList[np.argmax(self.best_probsList)]


if __name__ == '__main__':
    fpath = "F:\\DeepLearning\\Data"

    fpath_insample = os.path.join(fpath, "insample")
    fpath_outsample = os.path.join(fpath, "outsample")
    X_train = np.load(os.path.join(fpath_insample, "X.npy"))
    Y_train = np.load(os.path.join(fpath_insample, "Y.npy"))
    X_test = np.load(os.path.join(fpath_outsample, "X.npy"))
    Y_test = np.load(os.path.join(fpath_outsample, "Y.npy"))

    ga = GA(X_train, Y_train, X_test, Y_test, populationSize=50, iteration = 10)
    ga.Start()

    result = {
        "best_factors": ga.best_factors,
        "best_factorsList": ga.best_factorsList,
        "best_probs": ga.best_probs,
        "best_probsList": ga.best_probsList
    }
    print(ga.best_probs, "\n", sum(ga.best_factors))

    with open("F:\\DeepLearning\\Model\\result_GA.pkl", 'wb+') as file:
        pkl.dump(result, file)