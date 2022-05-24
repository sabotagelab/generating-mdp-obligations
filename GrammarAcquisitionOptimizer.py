import logging
from copy import copy
from typing import Tuple

from boss.code.optimizers.GrammarGeneticAlgorithmAcquisitionOptimizer import GrammarGeneticProgrammingOptimizer, unparse

from emukit.core import ParameterSpace
from emukit.core.acquisition import Acquisition
from emukit.core.initial_designs import RandomDesign
from emukit.core.loop import OuterLoop
from emukit.core.loop.user_function_result import UserFunctionResult

import numpy as np
from emukit.core.optimization import ContextManager

from tqdm import tqdm, trange

from model_check import Automaton, Obligation, checkObligation

_log = logging.getLogger(__name__)


class GrammarGeneticValidityOptimizer(GrammarGeneticProgrammingOptimizer):
    """
    Optimizes the acquisition function using Genetic programming over CFG parameters
    """

    def __init__(self, space: ParameterSpace, automaton: Automaton, dynamic: bool = False, num_evolutions: int = 10,
                 population_size: int = 5, tournament_prob: float = 0.5,
                 p_crossover: float = 0.8, p_mutation: float = 0.05,
                 ) -> None:
        """
        :param space: The parameter space spanning the search problem (has to consist of a single CFGParameter).
        :param num_steps: Maximum number of evolutions.
        :param dynamic: allow early stopping to choose number of steps (chooses between 10 and 100 evolutions)
        :param num_init_points: Population size.
        :param tournament_prob: proportion of population randomly chosen from which to choose a tree to evolve
                                (larger gives faster convergence but smaller gives better diversity in the population)
        :p_crossover: probability of crossover evolution (if not corssover then just keep the same (reproducton))
        :p_mutation: probability of randomly mutatiaon

        """
        super().__init__(space, dynamic, num_evolutions, population_size, tournament_prob, p_crossover, p_mutation)
        self.validities = {}
        self.automaton = automaton
        self.outer_loop = None

    def set_outer_loop(self, outer_loop: OuterLoop):
        self.outer_loop = outer_loop

    def get_best_valid(self, acquisition: Acquisition, n: int = 10, sample_attempts: int = 3, opt_mode: bool = False):
        """
        Optimize an acquisition function subject to validity in the automaton using a genetic algorithm

        :param acquisition: acquisition function to be maximized in pair with validity.
        :param automaton: automaton with respect to which formulas must be valid.
        :param n: the number of valid, maximizing formulas to return.
        :param sample_attempts: the number of times to resample the population if none of the samples are valid
        :return: a list of strings that maximize the acquisition function s.t. being valid in the automaton
        """
        # initialize validities
        # self.validities = {}
        # initialize population of tree
        random_design = RandomDesign(self.space)
        # in case the starting population is entirely invalid, resample
        for i in range(sample_attempts):
            population = random_design.get_samples(self.population_size)
            # calc fitness for current population
            fitness_pop = self._fitness_st_validity(population, acquisition, self.automaton, 0)
            if sum(fitness_pop) == 0 and (i + 1) >= sample_attempts:
                raise ValueError('No valid samples could be found; try enumerative search instead.')
            elif sum(fitness_pop) != 0:
                break
        standardized_fitness_pop = -1 * (fitness_pop / sum(fitness_pop))
        # initialize best location and score so far
        best_fit, best_x = get_top_n(population, fitness_pop, n)
        X_max = np.zeros((1, 1), dtype=object)
        X_max[0] = unparse(population[np.argmax(fitness_pop)])
        acq_max = np.max(fitness_pop).reshape(-1, 1)
        iteration_bests = []
        i = 0
        pbar = tqdm(total=self.num_evolutions + 90)
        _log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
        for step in range(self.num_evolutions):
            _log.info("Performing evolution step {}".format(step))
            # recalc fitness
            standardized_fitness_pop, acq_max, X_max, best_x, best_fit, iteration_bests \
                = self._update_pop(sample_attempts, population, standardized_fitness_pop, acquisition, self.automaton,
                                   step, acq_max, X_max, best_x, iteration_bests)
            pbar.update()
        # if dynamic then keep running (stop when no improvement over most recent 10 populations)
        if self.dynamic:
            stop = False
        else:
            stop = True
        j = 10
        while not stop:
            standardized_fitness_pop, acq_max, X_max, best_x, best_fit, iteration_bests \
                = self._update_pop(sample_attempts, population, standardized_fitness_pop, acquisition, self.automaton,
                                   i, acq_max, X_max, best_x, iteration_bests)
            # if acq_max[0][0] == max(iteration_bests[:-10]):
            #     stop = True
            # also stop if ran for 100 evolutions in total
            if j == 100:
                stop = True
            i += 1
            j += 1
            pbar.update()

        pbar.close()
        # correct the fitness from negation for maximization
        best_fit = best_fit * -1
        # return best n solutions from the whole optimization
        return best_x, best_fit

    def _recalc_fitness(self, sample_attempts, init_pop, std_fit_pop, acq, auto, v_imp):
        # recalc fitness
        for i in range(sample_attempts):
            population = self._evolve(init_pop, std_fit_pop)
            # calc fitness for current population
            fitness_pop = self._fitness_st_validity(population, acq, auto, v_imp)
            if sum(fitness_pop) == 0 and (i + 1) >= sample_attempts:
                raise ValueError('No valid samples could be found; try enumerative search instead.')
            elif sum(fitness_pop) != 0:
                standardized_fitness_pop = -1 * (fitness_pop / sum(fitness_pop))
                return population, fitness_pop, standardized_fitness_pop

    def _update_pop(self, sample_attempts, population, standardized_fitness_pop, acquisition, automaton, step, acq_max,
                    X_max, best_x, iteration_bests):
        population, fitness_pop, standardized_fitness_pop = self._recalc_fitness(sample_attempts, population,
                                                                                 standardized_fitness_pop,
                                                                                 acquisition, automaton, step)
        # update best location and score (if found better solution)
        acq_pop_max = np.max(fitness_pop)
        iteration_bests.append(acq_pop_max)
        _log.info("best acqusition score in the new population".format(acq_pop_max))
        acq_max[0][0] = self._fitness_st_validity(X_max, acquisition, automaton, step, False)
        if acq_pop_max > acq_max[0][0]:
            acq_max[0][0] = acq_pop_max
            X_max[0] = unparse(population[np.argmax(fitness_pop)])
        best_fit = self._fitness_st_validity(best_x, acquisition, automaton, step, False)
        best_x, best_fit = compare_best(best_x, best_fit, unparse(population), fitness_pop)
        return standardized_fitness_pop, acq_max, X_max, best_x, best_fit, iteration_bests

    def _fitness_st_validity(self, population, acq, automaton, validity_importance, parse=True):
        if parse:
            formulas = unparse(population)
        else:
            formulas = np.array(population)
        fitness_pop = acq.model.predict(formulas)[0]
        validity_pop = []
        for phi in formulas:
            if phi[0] in self.validities:
                validity_pop.append(self.validities[phi[0]])
            else:
                obligation = Obligation.fromPCTL(reformat(phi[0]))
                validity = checkObligation(automaton, obligation)
                self.validities[phi[0]] = validity
                validity_pop.append(validity)
        validity_pop = np.array(validity_pop).reshape(-1, 1)
        # fitness is negated because the model minimizes toward 0, but this optimizer maximizes
        # a formula's fitness is multiplied by a factor corresponding to its validity and the round of optimization
        # i.e. an invalid formula will be more negative (after negation) than a valid formula
        # and the size of this factor increases as the rounds of optimization increase, eventually ensuring that all
        # optimal formulas are valid formulas.
        return -1 * fitness_pop * (validity_importance * (1-validity_pop) + 1)

    def _optimize(self, acquisition: Acquisition, context_manager: ContextManager) -> Tuple[np.ndarray, np.ndarray]:
        """
        See AcquisitionOptimizerBase._optimizer for parameter descriptions.

        Optimize an acquisition function using a GA that respects validity
        """
        valid_champ = False
        pass_through = False
        k = 0
        while not valid_champ:
            _log.info("Performing genetic optimization round {}".format(k))
            # if pass_through:
                # self.outer_loop._update_models()
            # initialize population of tree
            random_design = RandomDesign(self.space)
            population = random_design.get_samples(self.population_size)
            # clac fitness for current population
            fitness_pop = acquisition.evaluate(unparse(population))
            standardized_fitness_pop = fitness_pop / sum(fitness_pop)
            # initialize best location and score so far
            X_max = np.zeros((1, 1), dtype=object)
            X_max[0] = unparse(population[np.argmax(fitness_pop)])
            acq_max = np.max(fitness_pop).reshape(-1, 1)
            iteration_bests = []
            _log.info("Starting local optimization of acquisition function {}".format(type(acquisition)))
            for step in range(self.num_evolutions):
                _log.info("Performing evolution step {}".format(step))
                # evolve populations
                population = self._evolve(population, standardized_fitness_pop)
                # recalc fitness
                fitness_pop = acquisition.evaluate(unparse(population))
                standardized_fitness_pop = fitness_pop / sum(fitness_pop)
                # update best location and score (if found better solution)
                acq_pop_max = np.max(fitness_pop)
                iteration_bests.append(acq_pop_max)
                _log.info("best acqusition score in the new population".format(acq_pop_max))
                if acq_pop_max > acq_max[0][0]:
                    acq_max[0][0] = acq_pop_max
                    X_max[0] = unparse(population[np.argmax(fitness_pop)])
            # if dynamic then keep running (stop when no improvement over most recent 10 populations)
            if self.dynamic:
                stop = False
            else:
                stop = True
            i = 10
            while not stop:
                _log.info("Performing evolution step {}".format(step))
                # evolve populations
                population = self._evolve(population, standardized_fitness_pop)
                # recalc fitness
                fitness_pop = acquisition.evaluate(unparse(population))
                standardized_fitness_pop = fitness_pop / sum(fitness_pop)
                # update best location and score (if found better solution)
                acq_pop_max = np.max(fitness_pop)
                iteration_bests.append(acq_pop_max)
                _log.info("best acqusition score in the new population".format(acq_pop_max))
                if acq_pop_max > acq_max[0][0]:
                    acq_max[0][0] = acq_pop_max
                    X_max[0] = unparse(population[np.argmax(fitness_pop)])
                if acq_max[0][0] == max(iteration_bests[:-10]):
                    stop = True
                # also stop if ran for 100 evolutions in total
                if i == 100:
                    stop = True
                i += 1
            # check the best population for validity, starting from the top, and returning once found
            results = []
            for _ in trange(len(population)):
                max_index = np.argmax(fitness_pop)
                phi = unparse(population[max_index])
                seen_phi = False
                if phi[0] in self.validities:
                    valid = self.validities[phi[0]]
                    seen_phi = True
                else:
                    obligation = Obligation.fromPCTL(reformat(phi[0]))
                    valid = checkObligation(self.automaton, obligation)
                    self.validities[phi[0]] = valid
                if valid:
                    X_max[0] = phi[0]
                    acq_max[0][0] = fitness_pop[max_index]
                    valid_champ = True
                    break
                elif not seen_phi:
                    fitness_pop[max_index] = 0
                    # update the model telling it this formula is invalid and therefore bad (in the model's scoring)
                    results += [UserFunctionResult(phi, np.array([1]))]
                else:
                    fitness_pop[max_index] = 0
            # TODO: only update model if we restart optimization?
            # self.outer_loop.loop_state.results += results
            pass_through = True

            k += 1
        # return best solution from the whole optimization
        return X_max, acq_max


def get_top_n(population, fitness_pop, n):
    temp_fit = fitness_pop
    maximizers = []
    maxes = []
    for i in range(n):
        max_index = np.argmax(temp_fit)
        x_max = unparse(population[max_index])
        maximizers.append(x_max)
        maxes.append(copy(temp_fit[max_index]))
        temp_fit[max_index] = -np.inf
    return maxes, maximizers


def compare_best(pop1, fit1, pop2, fit2):
    while max(fit2) > min(fit1):
        min_ind1 = np.argmin(fit1)
        max_ind2 = np.argmax(fit2)
        if pop2[max_ind2] not in pop1:
            pop1[min_ind1] = pop2[max_ind2]
            fit1[min_ind1] = fit2[max_ind2]
        else:
            fit2[max_ind2] = -np.inf
    return pop1, fit1


def reformat(x):
    x = x.replace("lb", "(")
    x = x.replace("rb", ")")
    return x
