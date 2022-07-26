"""
Created July 2020

@author: Colin
"""

from model_check import *
from bayes_opt import explore_formulas
import numpy as np


def setupGridworld():
    return Automaton.as_gridworld(4, 3, cells=[("goal", [(3, 2)], 10, True, True),
                                               ("pit", [(3, 1)], -50, True, True),
                                               ("wall", [(1, 1)], -1, True, False)])


def setupGridworldSmaller():
    return Automaton.as_gridworld(3, 3, cells=[("goal", [(2, 2)], 10, True, True),
                                               ("pit", [(2, 1)], -50, True, True),
                                               ("wall", [(1, 1)], -1, True, False)])


def setupGridworldTwo():
    return Automaton.as_gridworld(5, 5, cells=[("goal", [(4, 2)], 10, True, True),
                                               ("fire", [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4), (1, 3), (2, 3), (3, 3),
                                                         (1, 2), (2, 2), (3, 2)], -5, False, True)])


def setupCliffworld(cliff_reward=-10):
    gw = Automaton.as_gridworld(5, 4, start=(0, 2),
                                cells=[("goal", [(4, 2)], 10, True, True),
                                       ("cliff", [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)], cliff_reward, True, True),
                                       ("wall", [(1, 2), (3, 2)], -1, True, False)])
    # combine cliff states
    cliffs = gw.graph.vs.select(type="cliff")
    first_cliff = gw.graph.vs.find(pos=(0, 0))
    e_to_cliff = gw.graph.es.select(_target_in=cliffs)
    for edge in e_to_cliff:
        source = edge.source
        attributes = edge.attributes()
        gw.graph.add_edges([(source, first_cliff)], attributes)
    gw.graph.delete_edges(e_to_cliff)
    bad_cliffs = cliffs.select(pos_ne=(0, 0))
    # gw.graph.delete_vertices(bad_cliffs)
    # make start state inaccessible
    start = gw.graph.vs.find(pos=(0, 2))
    e_to_start = gw.graph.es.select(_target=start, _source_ne=start.index)
    for edge in e_to_start:
        source = edge.source
        prob = edge["prob"]
        act = edge["action"]
        other_edge = gw.graph.es.find(_source=source, _target=source, action=act)
        other_edge["prob"] += prob
    gw.graph.delete_edges(e_to_start)
    return gw


def test_gridworld():
    gw = setupGridworld()
    # plot(gw.graph)
    frags = generate_fragments(gw, gw, gw.q0, "EX (name = 0 | name = 1)", t=2)
    # assert len(frags) == 12
    print(len(frags))
    for frag in frags:
        # plot(frag.graph, layout=frag.graph.layout_fruchterman_reingold())
        pass
    tree = build_fragment_tree(frags, gw)
    # plot(tree.graph, layout=tree.graph.layout_fruchterman_reingold())
    actions = ["up", "down", "left", "right"]
    for v in tree.graph.vs:
        for action in actions:
            act_es = tree.graph.es.select(_source=v, action=action)
            act_prob = sum(act_es["prob"])
            np.testing.assert_allclose(act_prob, 1)
    print(tree.optimal(0.9))
    pass


def explore_gridworld():
    gw = setupGridworld()
    atoms = []
    for s in range(12):
        atoms.append("name=" + str(s))
    explore_formulas(gw, propositions=atoms, online_query_size=10)


def explore_gridworld2():
    gw = setupGridworldTwo()
    atoms = []
    for s in range(5 * 5):
        atoms.append("name=" + str(s))
    explore_formulas(gw, propositions=atoms, online_query_size=60)


def cliffworld_experiment():
    gw = setupCliffworld()
    atoms = []
    state_range = [0] + list(range(5, 20))
    for s in state_range:
        atoms.append("name=" + str(s))
    explore_formulas(gw, propositions=atoms, init_query_size=5, online_query_size=20)


if __name__ == "__main__":
    # test_gridworld()
    # enum_gridworld()
    # explore_gridworld()
    cliffworld_experiment()
    pass
