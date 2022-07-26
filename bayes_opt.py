"""
Created December 2021

@author: Colin

Perform symbolic regression of a formula to meet an objective that is latent in a user.
Interact with the user to receive feedback on proposed formulas.
Model the user's objective over formulas as a Gaussian process with a string similarity kernel.
Find the string most likely to improve the performance of the model using an evolutionary model.
Once preferences have been suitably learned, search only for formulas that are satisfied by a given model.

Based on, and using code from "BOSS: Bayesian Optimization over String Spaces"
by Henry B. Moss, Daniel Beck, Javier Gonzalez, David S. Leslie, Paul Rayson
"""

import pickle
import timeit
import chime
import numpy as np

from emukit.core import ParameterSpace
from emukit.core.initial_designs import RandomDesign
from emukit.bayesian_optimization.loops import BayesianOptimizationLoop
from emukit.bayesian_optimization.acquisitions import ExpectedImprovement
from emukit.core.loop import FixedIterationsStoppingCondition

from nltk import ChartParser
from tqdm import tqdm

# import BOSS code
from boss.code.CFG.CFG import Grammar
from boss.code.parameters.cfg_parameter import CFGParameter
from boss.code.parameters.cfg_parameter import unparse
from boss.code.emukit_models.emukit_ssk_model import SSK_model

# import our code
from GrammarAcquisitionOptimizer import GrammarGeneticValidityOptimizer
from model_check import Obligation, checkObligation


def explore_formulas(auto, propositions=None, init_query_size=10, online_query_size=60):
    if not propositions:
        propositions = ['p', 'q', 'a', 'b']
    prop_str = " | ".join(['"' + proposition + '"' for proposition in propositions])

    # a simplified, restricted PCTL grammar; for speed and for eliminating start-state formulas
    grammar_str = """
                    N -> PROB COMP PCT OB Q CB | NOT PROB COMP PCT OB Q CB
                    Q -> NOT LB Q RB | Q AND R | Q OR R | Q IMPLIES LB R RB | LB S RB
                    R -> Q
                    COMP -> LEQ | GEQ
                    PCT -> ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | NINE | TEN
                    S -> GLB O | NXT O | FTR O | LB O UNTIL M RB
                    M -> O
                    O -> LB L RB | S | Q | T
                    L -> NOT LB K RB | J AND K | J OR K | J IMPLIES LB K RB | LB K RB
                    K -> L | S | T
                    J -> K
                    LEQ -> "<="
                    LT -> "<"
                    GEQ -> ">="
                    GT -> ">"
                    PROB -> "P"
                    ZERO -> "0"
                    ONE -> "0.1"
                    TWO -> "0.2"
                    THREE -> "0.3"
                    FOUR -> "0.4"
                    FIVE -> "0.5"
                    SIX -> "0.6"
                    SEVEN -> "0.7"
                    EIGHT -> "0.8"
                    NINE -> "0.9"
                    TEN -> "1.0"
                    NOT -> "!"
                    AND -> "&"
                    OR -> "|"
                    IMPLIES -> "=>"
                    UNTIL -> "U"
                    GLB -> "G"
                    NXT  -> "X"
                    FTR -> "F"
                    LB -> "lb"
                    RB -> "rb"
                    OB -> "["
                    CB -> "]"
                    T -> """

    grammar_str = grammar_str + prop_str

    ctl_grammar = Grammar.fromstring(grammar_str)
    ctl_parser = ChartParser(ctl_grammar)
    max_length = 24
    space = ParameterSpace([CFGParameter("grammar", ctl_grammar, max_length=max_length, min_length=0)])
    random_design = RandomDesign(space)
    # We recommend setting a breakpoint below, and calling random_design.get_samples(10) in pdb, then interrupting the
    # execution of the function. This should allow the get_samples calls made in this program to execute more quickly.
    X_init = random_design.get_samples(50)
    # get initial formulas and interests
    # The following line can be uncommented to initialize the bayesian optimization with hand-picked formulas instead
    # of random formulas. In which case, comment out the calls to random_valid_formulas and objective(xs)
    # xs, ys = init_query(init_query_size, ctl_parser, max_length)
    xs = random_valid_formulas(space, auto, 10)
    ys = objective(xs)

    # for testing, set up dummy inputs and outputs
    # xs = ["EG name=7", "EF AX name=7", "AG name=7", "AF EX name=7", "EX name=0", "EX name=1", "EF AG name=11",
    #       "name=0", "A[ ! name=11 U name=0 ]", "A[ name=5 U name=0 ]"]
    # xs = ["! P >= 0.4 [ ( F G name=7 ) ]", "P >= 0 [ ( F name=7 ) ]", "P <= 0.7 [ ( F name=11 ) ]",
    #       "! P <= 0.7 [ ( F name=10 ) => ( F name=11 ) ]", "P >= 1.0 [ ( F name=0 ) ]", "P <= 0 [ ! ( X name=1 ) ]",
    #       "P >= 0.7 [ ( X name=4 ) ]", "P <= 0 [ ( F name=11 ) & ( F name=7 ) ]",
    #       "P <= 1.0 [ ( ! ( name=11 ) U name=0 ) ]", "P <= 1.0 [ ( name=5 U name=0 ) ]"]
    # xs = ["! P >= 0.2 [ ( F name=0 ) ]", "P >= 0.5 [ ( F name=14 ) ]",
    #       "P <= 0.5 [ ( F name=0 ) ]", "! P <= 0.8 [ ( F name=14 ) ]",
    #       "! P <= 0.7 [ ( X X X X X X name=14 ) ]"]
    # xs = [deformat(x) for x in xs]
    # ys = [1 - 78 / 100, 1 - 70 / 100, 1 - 90 / 100, 1 - 74 / 100, 1 - 0, 1 - 0, 1 - 58 / 100, 1 - 0, 1 - 0, 1 - 0]
    # ys = [1 - 95/100, 1 - 95/100, 1 - 85/100, 1 - 89/100, 1 - 87/100]

    xs = np.array(xs).reshape(-1, 1)
    ys = np.array(ys).reshape(-1, 1)
    # build BO loop with SSK model
    model = SSK_model(space, xs, ys, max_subsequence_length=8, n_restarts=5, observation_noise=True)
    # with open("bayesopt_loop_SSK.model-cliffworld.pkl", 'rb') as f:
    #     model = pickle.load(f)
    # load acquisition function
    expected_improvement = ExpectedImprovement(model, jitter=0.4)
    # use GA to optimize acquisition function
    optimizer = GrammarGeneticValidityOptimizer(space, auto,
                                                dynamic=True,
                                                population_size=100,
                                                tournament_prob=0.5,
                                                p_crossover=0.8,
                                                p_mutation=0.1)
    # define BO loop
    bayesopt_loop_SSK = BayesianOptimizationLoop(model=model,
                                                 space=space,
                                                 acquisition=expected_improvement,
                                                 acquisition_optimizer=optimizer)
    # tell the optimizer which loop it's working with
    optimizer.set_outer_loop(bayesopt_loop_SSK)
    # add loop summary
    bayesopt_loop_SSK.iteration_end_event.append(summary)
    # run BO loop
    stop_crit = FixedIterationsStoppingCondition(i_max=online_query_size)
    bayesopt_loop_SSK.run_loop(objective, stop_crit)
    print("Generating Best Valid Formulas")
    best_x, best_fit = optimizer.get_best_valid(expected_improvement, n=10)
    best_x = np.array(best_x)
    rand_x = random_valid_formulas(space, auto, n=10)
    xs, ys, zs = collect_eval(best_x, best_fit, rand_x, bayesopt_loop_SSK.model)
    with open("valid_experiment_cliffworld_6.pkl", "wb") as f:
        pickle.dump((xs, ys, zs), f)
    print("Done")
    return best_x, best_fit


def summary(loop, loop_state):
    print("Performing BO step {}".format(loop.loop_state.iteration))


def reformat(x):
    x = x.replace("lb", "(")
    x = x.replace("rb", ")")
    return x


def deformat(x):
    x = x.replace("(", "lb")
    x = x.replace(")", "rb")
    return x


def query(x):
    valid_value = False
    while not valid_value:
        print("How interesting is this formula [0-100]: ")
        print(reformat(x))
        y_out = float(input())
        if 0 <= y_out <= 100:
            valid_value = True
        else:
            print("Invalid Value: please input a number between 0 and 100.")
    # return 1-interest so we're minimizing
    return 1 - y_out / 101


def objective(X):
    # must take an n by 1 ndarray
    Y = np.zeros((X.shape[0], 1))
    for i in range(X.shape[0]):
        x = X[i][0]
        chime.info()
        Y[i] = query(x)
    return Y


def init_query(init_size, parser, max_length):
    xs = []
    ys = []
    # initialize dataset
    print("Please provide " + str(init_size) + " formulas and your interest in them...")
    for i in range(init_size):
        valid_ctl = False
        while not valid_ctl:
            print("Formula " + str(i + 1) + ": ")
            x_out = input()
            x_out = deformat(x_out)
            try:
                parse_trees = parser.parse(x_out.split())
            except ValueError as err:
                print("Grammatical Error: {0}".format(err))
                print("Please try again,\n")
                continue
            # check to see if the given formula is a valid CTL formula
            for _ in parse_trees:
                valid_ctl = True
                break
            if not valid_ctl:
                print("Grammatical Error: invalid formula.")
                print("Please try again,\n")
            if len(x_out.split(" ")) > max_length:
                print("Grammatical Error: formula too long.")
                print("Please try again, or restart with larger kernel space,\n")
                valid_ctl = False

        xs.append(x_out)

        ys.append(query(x_out))
        print("\n")
        print("###################################################")
    return xs, ys


def random_valid_formulas(space, auto, n):
    valid = []
    invalid = []
    random_design = RandomDesign(space)
    pbar = tqdm(total=n)
    while len(valid) < n:
        population = random_design.get_samples(n)
        formulas = unparse(population)
        for phi in formulas:
            if phi[0] not in invalid:
                obligation = Obligation.fromPCTL(reformat(phi[0]))
                validity = checkObligation(auto, obligation)
                if validity:
                    valid.append(phi[0])
                    pbar.update()
                    if len(valid) >= n:
                        break
                else:
                    invalid.append(phi[0])
    pbar.close()
    return np.array(valid).reshape(-1, 1)


def collect_eval(best_x, best_fit, rand_x, model):
    # the formulas
    xs = np.vstack((best_x, rand_x))
    # the evaluator's score for each formula
    ys = np.zeros(xs.shape)
    # the model's score for each formula
    zs = np.vstack((best_fit, np.zeros(best_fit.shape)))
    n = len(best_x) + len(rand_x)
    choices = np.arange(n)
    np.random.shuffle(choices)
    for i in choices:
        ys[i][0] = query(xs[i][0])
        if i >= len(best_x):
            zs[i][0] = model.predict(np.array([xs[i]]))[0]
    return xs, ys, zs


if __name__ == "__main__":
    test_str = """
                N -> PROB COMP PCT OB Q CB
                Q -> N | NOT LB Q RB | Q AND R | Q OR R | Q IMPLIES R | LB S RB | T
                R -> Q
                COMP -> LEQ | LT | GEQ | GT
                PCT -> ZERO | ONE | TWO | THREE | FOUR | FIVE | SIX | SEVEN | EIGHT | NINE | TEN
                S -> GLB O | NXT O | FTR O | O UNTIL P
                O -> S | Q
                P -> S | Q
                LEQ -> "<="
                LT -> "<"
                GEQ -> ">="
                GT -> ">"
                PROB -> "P"
                ZERO -> "0"
                ONE -> "0.1"
                TWO -> "0.2"
                THREE -> "0.3"
                FOUR -> "0.4"
                FIVE -> "0.5"
                SIX -> "0.6"
                SEVEN -> "0.7"
                EIGHT -> "0.8"
                NINE -> "0.9"
                TEN -> "1.0"
                NOT -> "!"
                AND -> "&"
                OR -> "|"
                IMPLIES -> "=>"
                UNTIL -> "U"
                GLB -> "G"
                NXT  -> "X"
                FTR -> "F"
                LB -> "lb"
                RB -> "rb"
                OB -> "["
                CB -> "]"
                T -> "state=7" | "state=11" | "state=3" | "state=6" | "state=10"
                """
    test_grammar = Grammar.fromstring(test_str)
    test_space = ParameterSpace([CFGParameter("test_grammar", test_grammar, max_length=15, min_length=0)])
    test_design = RandomDesign(test_space)
    print("Testing...")
    print(timeit.timeit('test_design.get_samples(10)', globals=globals(), number=10))
    samples = test_design.get_samples(10)
    for sample in samples:
        print(reformat(unparse(sample)[0]))
