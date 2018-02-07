import numpy as np
import random
import copy
from player_object import FlappyBirdPlayer
from neural_network import NeuralNetwork


class GeneticAlgorithm(object):
    def __init__(self, x_over_crosses=1, percent_win=.4, x_over_rate=.7, mut_num=10, mut_rate=1, mut_nudge=0.5, mut_decline_epochs=0, fit_converge_epochs=30):
        # HYPER PARAMETERS
        #-----------------------------

        # N uniform crossover points
        self.NUM_OF_CROSS_OVER_POINTS = x_over_crosses
        # Percent of population considered winners (cross over only happens with winners)
        # IE the last place solution will not cross with the second last
        self.PERCENT_OF_WINNERS = percent_win
        # Crossover Rate
        self.CROSS_OVER_RATE = x_over_rate

        # N unifrom mutation points
        self.NUM_OF_MUTATIONS = mut_num
        # Rate of Mutation
        self.RATE_OF_MUTATION = mut_rate
        # Max nudge of muation
        self.MUTATION_NUDGE = mut_nudge

        # Simulated Annealing: # of epochs until mutation nudge is nearly zero ( < 0 for never changing)
        self.EPOCHS_UNTIL_MUTATION_BECOMES_ZERO = mut_decline_epochs

        # epocs until only fitness is used (-1 for always fitness) opposed to fitness and diversity
        self.EPOCHS_UNTIL_FITNESS_CONVERGENCE = fit_converge_epochs

        #-----------------------------
        # END HYPER PARAMETERS


    def partition_by_fitness(self, players, begin, end):
        pivot = begin
        for i in range(begin + 1, end + 1):
            if (players[i].get_fitness() >= players[begin].get_fitness()):
                pivot += 1
                players[i], players[pivot] = players[pivot], players[i]
        players[pivot], players[begin] = players[begin], players[pivot]
        return pivot


    def quicksort_by_fitness(self, players, begin=0, end=None):
        if (end is None):
            end = len(players) - 1

        def _quicksort_by_fitness( players, begin, end):
            if (begin >= end):
                return players
            pivot = self.partition_by_fitness(players, begin, end)
            _quicksort_by_fitness(players, begin, pivot - 1)
            _quicksort_by_fitness(players, pivot + 1, end)
        return _quicksort_by_fitness(players, begin, end)


    def partition_by_diversity(self, players, begin, end):
        pivot = begin
        for i in range(begin + 1, end + 1):
            if (players[i].diversity >= players[begin].diversity):
                pivot += 1
                players[i], players[pivot] = players[pivot], players[i]
        players[pivot], players[begin] = players[begin], players[pivot]
        return pivot


    def quicksort_by_diversity(self, players, begin=0, end=None):
        if (end is None):
            end = len(players) - 1

        def _quicksort_by_diversity(players, begin, end):
            if (begin >= end):
                return players
            pivot = self.partition_by_diversity(players, begin, end)
            _quicksort_by_diversity(players, begin, pivot - 1)
            _quicksort_by_diversity(players, pivot + 1, end)
        return _quicksort_by_diversity(players, begin, end)


    def mutate(self, weights, biases, epoch_counter):

        for i in range(self.NUM_OF_MUTATIONS):
            # determine if there will be a mutation
            if (random.uniform(0, 1) > self.RATE_OF_MUTATION):
                continue

            mutation_reduction = 1
            if (self.EPOCHS_UNTIL_MUTATION_BECOMES_ZERO > 0):
                mutation_reduction = 1 - \
                    (epoch_counter / self.EPOCHS_UNTIL_MUTATION_BECOMES_ZERO)

            # determine nudge
            nudge = 0
            if (random.randint(0, 1) == 0):
                # nudge will be positive
                #nudge = random.uniform(0,MUTATION_NUDGE)
                nudge = self.MUTATION_NUDGE * mutation_reduction
            else:
                # nudge will be negative
                #nudge = -1 * random.uniform(0,MUTATION_NUDGE)
                nudge = -1 * self.MUTATION_NUDGE * mutation_reduction

            # determine if the mutation will be in biases or weights
            if (random.randint(0, 1) == 0):
                # selected weights
                weight_selected = random.randint(0, len(weights) - 1)
                weights[weight_selected] += nudge
            else:
                # selected biases
                bias_selected = random.randint(0, len(biases) - 1)
                biases[bias_selected] += nudge

        return (weights, biases)


    def create_child(self, network_A, network_B, epoch_counter):
        # Network A and Network B are the parents. (Note: must be of same NN structure)
        weights_A = network_A.get_weights_as_list()
        weights_B = network_B.get_weights_as_list()
        biases_A = network_A.get_biases_as_list()
        biases_B = network_B.get_biases_as_list()

        child_A = list()
        child_B = list()

        weights_cross_over_site = list()
        biases_cross_over_site = list()
        # Select crossover points
        for i in range(self.NUM_OF_CROSS_OVER_POINTS):
            site = -1
            while (site not in weights_cross_over_site):
                site = random.randint(2, len(weights_A) - 2)
                weights_cross_over_site.append(site)
            site = -1
            while (site not in biases_cross_over_site):
                site = random.randint(1, len(biases_A) - 1)
                biases_cross_over_site.append(site)

        weights_cross_over_site.sort(key=int)
        biases_cross_over_site.sort(key=int)

        # cross over weights
        for i in range(len(weights_cross_over_site)):
            if (i + 1 >= len(weights_cross_over_site) and i % 2 == 0):
                weights_A[weights_cross_over_site[i]:] = weights_B[weights_cross_over_site[i]:]
                weights_B[weights_cross_over_site[i]:] = weights_A[weights_cross_over_site[i]:]
                break
            if (i + 1 >= len(weights_cross_over_site)):
                break

            if (i % 2 == 0):
                # child 1
                weights_A[weights_cross_over_site[i]: weights_cross_over_site[i + 1]
                        ] = weights_B[weights_cross_over_site[i]: weights_cross_over_site[i + 1]]
                # child 2
                weights_B[weights_cross_over_site[i]: weights_cross_over_site[i + 1]
                        ] = weights_A[weights_cross_over_site[i]: weights_cross_over_site[i + 1]]

        # Cross over with biases now
        for i in range(len(biases_cross_over_site)):
            if (i + 1 >= len(biases_cross_over_site) and i % 2 == 0):
                biases_A[biases_cross_over_site[i]:] = biases_B[biases_cross_over_site[i]:]
                biases_B[biases_cross_over_site[i]:] = biases_A[biases_cross_over_site[i]:]
                break
            if (i + 1 >= len(biases_cross_over_site)):
                break

            if (i % 2 == 0):
                # child 1
                biases_A[biases_cross_over_site[i]: biases_cross_over_site[i + 1]
                        ] = biases_B[biases_cross_over_site[i]: biases_cross_over_site[i + 1]]
                # child 2
                biases_B[biases_cross_over_site[i]: biases_cross_over_site[i + 1]
                        ] = biases_A[biases_cross_over_site[i]: biases_cross_over_site[i + 1]]
            if (i + 1 >= len(biases_cross_over_site)):
                break

        # Select 1 child to move on at random
        chosen_child_weights = weights_A
        chosen_child_biases = biases_A

        if (random.randint(0, 1) == 1):
            chosen_child_weights = weights_B
            chosen_child_biases = biases_B

        mutated_weights_biases = self.mutate(
            chosen_child_weights, chosen_child_biases, epoch_counter)

        # Note: in this case the learning rate and decay should be set to zero
        child_network = NeuralNetwork(
            network_A.layers_neurons, network_A.learning_rate, network_A.decay)

        # just changes the weights and biases to the proper ones selected above
        #(a newly created network randomizes them)
        child_network.list_to_weights(mutated_weights_biases[0])
        child_network.list_to_biases(mutated_weights_biases[1])

        return child_network


    def find_diversity(self, player, player_array):
        player_weights = np.asmatrix(player.neural_network.get_weights_as_list()).T
        player_biases = np.asmatrix(player.neural_network.get_biases_as_list()).T
        sum_of_diversity = 0
        for players in player_array:
            players_weights = np.asmatrix(
                players.neural_network.get_weights_as_list()).T
            players_biases = np.asmatrix(
                players.neural_network.get_biases_as_list()).T
            dist_weights = np.linalg.norm(player_weights - players_weights)
            dist_biases = np.linalg.norm(player_biases - players_biases)
            sum_of_diversity += ((dist_weights + dist_biases) / 2)

        sum_of_diversity = sum_of_diversity / len(player_array)
        player.diversity = sum_of_diversity
        return sum_of_diversity


    def cross_over(self, players_array, epoch_counter):
        num_of_winners = int(len(players_array) * self.PERCENT_OF_WINNERS)

        mod_list = list()

        for i in range(len(players_array)):
            if (i == 0):
                mod_list.append(players_array[i].neural_network)
                continue
            # determine if there is a cross
            if (random.uniform(0, 1) <= self.CROSS_OVER_RATE):
                # cross with random winner
                random_winner = random.randint(0, num_of_winners)
                child_net = self.create_child(players_array[i].neural_network,
                                        players_array[random_winner].neural_network, epoch_counter)
                mod_list.append(child_net)
            else:
                mod_list.append(players_array[i].neural_network)

        for i in range(len(players_array)):
            players_array[i].neural_network = mod_list[i]

        return players_array


    def choose_next_generation(self, previous_generation, epoch_counter):
        next_generation = list()
        remaining = previous_generation

        previous_generation_by_fitness = previous_generation

        self.quicksort_by_fitness(previous_generation_by_fitness)

        counter = 0
        for player in previous_generation_by_fitness:
            player.fitness_rank = counter
            counter += 1

        # Elitism method (pick best fit) add them to next generation
        next_generation.append(previous_generation_by_fitness[0])
        remaining.remove(previous_generation_by_fitness[0])

        while (len(remaining) > 0):
            if (epoch_counter < self.EPOCHS_UNTIL_FITNESS_CONVERGENCE):
                for player in remaining:
                    self.find_diversity(player, next_generation)
                

                self.quicksort_by_diversity(remaining)

                player_to_add_next = 0
                best_diversity_fitness_rank = 1000

                for player in remaining:
                    index_for_diversity = remaining.index(player)
                    index_for_fitness = player.fitness_rank
                    diversity_fitness_rank = index_for_diversity + index_for_fitness
                    if (diversity_fitness_rank < best_diversity_fitness_rank):
                        player_to_add_next = player
                        best_diversity_fitness_rank = diversity_fitness_rank

            else:
                self.quicksort_by_fitness(remaining)
                player_to_add_next = remaining[0]

            next_generation.append(player_to_add_next)
            remaining.remove(player_to_add_next)

        next_generation = self.cross_over(next_generation, epoch_counter)

        return next_generation
