"""
Created June 2020

@author: Colin
"""

import pickle
import subprocess
import numpy as np
import mdptoolbox.mdp as mdp
from copy import copy, deepcopy
from itertools import product
from igraph import *
from random_weighted_automaton import *


# TODO: refactor checkObligation, checkConditional, and generateFragments as Automaton class functions
# ^ the Obligation class could have validatedBy(Automaton)
# ^ also add a get_optimal_actions function to Automaton that returns the optimal actions
# TODO: asserts on the construction logic so malformed automata aren't constructed
# TODO: constructor that takes a list of "action classes" and associated probabilities
# ^ maybe even a gridworld constructor
# TODO: separate function of 'label', 'name' and atomic propositions
# ^ allow multiple propositions on a given vertex, and be able to check them.
# TODO: refactor lots of the functions just to simplify them
class Automaton(object):
    def __init__(self, graph, initial_state, atomic_propositions=None):
        """
        Create a new Automaton object from an igraph graph object and an initial state.
        The graph should have vertex properties "label" and "name".
        The graph should have the edge properties "action" and "weight.
        The graph may optionally have the edge property "prob".
        The initial_state parameter should the id of a vertex in graph.
        Adds a "delete" property to the edges; if edge["delete"] = 1, then it is marked for deletion later.

        :param graph:
        :param initial_state:
        """
        self.graph = graph
        self.graph.es["delete"] = [0] * self.graph.ecount()
        self.q0 = initial_state

        self.qn = self.q0
        self.t = 0
        self.q_previous = []
        self.t_previous = []
        self.num_clones = 0
        self.counter = False
        self.propositions = atomic_propositions
        self.prob = "prob" in self.graph.es.attribute_names()

    @classmethod
    def with_actions(cls, graph, actions, q0=0, probs=None):
        """
        graph is a directed igraph graph object
        actions is a dictionary that maps edges to actions
        key(action), values(list of edges)
        e.g. {0:[0, 1, 3], 1:[2, 4], 2:[5], ... }
        probs is a list of probabilities such that probs[k] is the
        probability of following edge[k] when the action containing
        edge[k] is taken.

        :param graph:
        :param actions:
        :param q0:
        :param probs:
        """

        state_names = [str(v.index) for v in graph.vs]
        graph.vs["name"] = state_names
        graph.vs["label"] = state_names
        for action in actions:
            for edge in actions[action]:
                edge = graph.es[edge]
                edge["action"] = action

        if probs:
            graph.es["prob"] = probs
        return cls(graph, q0)

    # TODO: make a gridworld cell class?
    @classmethod
    def as_gridworld(cls, x_len, y_len, start=(0, 0), action_success=0.7, cells=None, default_reward=-1):
        """
        Construct an automaton for an x_len-by-y_len gridworld, starting from the 'start' position, with actions up,
        down, left, and right.
        Each action, when taken, has a 'action_success' chance of effecting. Otherwise, another action is effected with
        probability (1-action_success)/3. That is, by default, the 'up' action has a probability of 0.7 to transition
        the automaton from (0,0) to (0,1). Taking the 'up' action has a probability of 0.1 to move the automaton down.
        Because (0,0) is in a corner, however, moving down leaves the automaton in its same state.

        The cells parameter is a list of tuples [(type, positions, reward, absorbing, accessible), ...].
        Each tuple represents one class of cells in the gridworld, relays the positions of those cells, the reward
        received for entering those cells, whether or not those cells can be exited, and whether or not those cells can
        be entered.
        The 'type' entry in the tuple is a string that denotes the class of cell; e.g. "goal", "pit", or "wall".
        The 'positions' entry is a list of 2-tuples (x, y) that denotes the locations of the cells of the given type in
        this gridworld. E.g. [(0,0), (2,2), (1,3)].
        The 'reward' entry is a real value that denotes the reward for entering a cell of the given type; e.g. 10.7.
        The 'absorbing' entry is a boolean value that denotes if cells of the given type are absorbing states.
        The 'accessible' entry is a boolean value that denotes if cells of the given type can be entered.
        If cells is left as None, then no cells are specified, and all cells are accessible, non-absorbing, and have a
        reward of 'default_reward'.
        An example cells input for a basic 4x3 gridworld is as follows:
        cells=[("goal", [(3, 2)], 10, True, True),
               ("pit", [(2, 2)], -50, True, True),
               ("wall", [(1, 1)], -1, False, False)]
        This places a goal in the upper-right of the grid with reward 10, and is an absorbing state,
        a pit just below the goal with a reward of -50, and is an absorbing state,
        and a wall just north-east of the starting position that is inaccessible. Note that because the wall is
        not accessible, its 'reward' and 'absorbing' entries are irrelevant.

        If a cell is not included among the positions in the 'cells' parameter, its type is "default", it is accessible,
        and not absorbing, and the reward for entering it is 'default_reward'.

        :param x_len:
        :param y_len:
        :param start:
        :param action_success:
        :param default_reward:
        :param cells:
        """
        n = x_len * y_len
        g_new = Graph(directed=True)
        pos_to_type = {}
        type_to_spec = {}
        default_spec = ("default", [], default_reward, False, True)
        # cache information about cell positions and types for future use
        for spec in cells:
            cell_type = spec[0]
            type_to_spec[cell_type] = spec
            cell_poss = spec[1]
            for cell_pos in cell_poss:
                pos_to_type[cell_pos] = cell_type

        positions = product(range(x_len), range(y_len))
        # set up the attribute dictionary so all vertices can be added in one go
        v_attr_dict = {"x": [], "y": [], "pos": [], "label": [], "name": [], "type": [],
                       "absorbing": [], "accessible": [], "reward": []}
        k = 0
        for y in range(y_len):
            for x in range(x_len):
                v_attr_dict["x"].append(x)
                v_attr_dict["y"].append(y)
                v_attr_dict["pos"].append((x, y))
                v_attr_dict["label"].append(str((x, y)))
                v_attr_dict["name"].append(str(k))
                cell_type = pos_to_type.get((x, y), "default")
                cell_spec = type_to_spec.get(cell_type, default_spec)
                v_attr_dict["type"].append(cell_type)
                v_attr_dict["reward"].append(cell_spec[2])
                v_attr_dict["absorbing"].append(cell_spec[3])
                v_attr_dict["accessible"].append(cell_spec[4])
                k += 1
        # add a vertex for every position, and set its attributes
        g_new.add_vertices(n, attributes=v_attr_dict)

        # set the four actions and the effect of actually following that action
        actions = ["up", "down", "left", "right"]
        effects = [(0, 1), (0, -1), (-1, 0), (1, 0)]
        # define the probability that an effect is effected in the case that the effect is not the assumed consequence
        # of a taken action
        off_prob = (1.0 - action_success) / 3.0
        # set up attribute dictionary and edge list so all edges can be added in one go
        edge_tuples = []
        signatures = []
        e_attr_dict = {"action": [], "weight": [], "prob": []}
        # for each vertex...
        for v in g_new.vs:
            effect_targets = []
            # if that vertex is inaccessible
            if not v["accessible"]:
                # just add a self edge for consistency reasons
                effect_targets = [v] * len(effects)
            elif v["absorbing"]:
                # if it's absorbing, set the target of each effect to itself
                effect_targets = [v] * len(effects)
            else:
                # otherwise, for each effect...
                for effect in effects:
                    # get the x and y positions that the automaton would enter under that effect
                    next_x = v["x"] + effect[0]
                    next_y = v["y"] + effect[1]
                    # if those x and y positions are inside the bounds...
                    if 0 <= next_x < x_len and 0 <= next_y < y_len:
                        # get the target of the effect...
                        next_v = g_new.vs.find(pos=(next_x, next_y))
                        # if the target isn't accessible...
                        if not next_v["accessible"]:
                            # then the effect leads to the same state
                            next_v = v
                    else:
                        # the target of the effect is out of bounds, so the effect would lead to the same state
                        next_v = v
                    # record the target of each effect
                    effect_targets.append(next_v)

            # for each action that can be taken...
            for i, action in enumerate(actions):
                # for each effect that action can have...
                for j, effect in enumerate(effects):
                    # get the target of this effect
                    next_v = effect_targets[j]
                    signature = (v.index, next_v.index, action)
                    # in the case that the effect matches the action taken...
                    if i == j:
                        # the probability of this effect is the action_success probability
                        prob = action_success
                    else:
                        # otherwise, the probability of this effect is the off_prob probability
                        prob = off_prob

                    if signature not in signatures:
                        signatures.append(signature)
                        # record an edge from v to next_v
                        edge_tuples.append((v.index, next_v.index))
                        # record the attributes of this edge
                        e_attr_dict["action"].append(action)
                        e_attr_dict["weight"].append(next_v["reward"])
                        e_attr_dict["prob"].append(prob)
                    else:
                        sig_index = signatures.index(signature)
                        e_attr_dict["prob"][sig_index] += prob
        # add an edge for each (vertex*action*effect), with associated attributes
        g_new.add_edges(edge_tuples, e_attr_dict)
        v_0 = g_new.vs.find(pos=start)
        return cls(g_new, v_0.index)

    def k(self, i):
        """
        get all actions available from vertex i.

        :param i:
        :return:
        """
        es = self.graph.es.select(_source=i)
        actions = []
        for edge in es:
            action = edge["action"]
            if action not in actions:
                actions.append(action)
        return actions

    def setCounter(self, var_name='c', start=0, term=1000):
        """
        create a simple counter in this automaton

        :param var_name:
        :param start:
        :param term:
        :return:
        """
        self.counter = (var_name, start, term)

    def forceKn(self, kn, source=0):
        """
        delete all edges from source vertex with edges in kn that are not
        themselves in kn.

        :param kn:
        :param source:
        :return:
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in kn
        selections = candidates.select(action_ne=kn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self

    def forceEn(self, en, source=0):
        """
        delete all edges from source vertex that are not
        themselves in en.

        :param en:
        :param source:
        :return:
        """
        # find all edges with this source.
        # when an index is deleted, the indices change
        # so I need to delete all edges at once
        # I can't query the edge indices here, so I'll give the edges temporary names
        # and grab the ones with names different from en
        candidates = self.graph.es.select(_source=source)
        for edge in candidates:
            if edge.index != en.index:
                edge["delete"] = 1
            else:
                edge["delete"] = 0
        candidates = candidates.select(delete=1)
        self.graph.delete_edges(candidates)
        return self

    def forceQn(self, qn, source=0):
        """
        delete all edges from source vertex with edges that do not lead to given
        vertex.

        :param qn:
        :param source:
        :return:
        """
        # find all edges with this source
        candidates = self.graph.es.select(_source=source)
        # find all candidates not in qn
        selections = candidates.select(_target_ne=qn)
        # remove these edges from the graph
        self.graph.delete_edges(selections)
        return self

    def union(self, g, target=0):
        """
        modify this automaton such that transitions in itself to the target
        state are replaced with transitions to automaton g.

        :param g:
        :param target:
        :return:
        """
        # recall certain properties of the given graphs
        v_mod = self.graph.vcount() + target % g.graph.vcount()

        # find the transitions to the target state not from previous state
        if len(self.q_previous) > 0:
            # es = self.graph.es.select(_target=target, _source_notin=[self.q_previous[-1]])
            es = self.graph.es.select(_target=target)
        else:
            es = None
        # if no
        if not es:
            return self
        else:
            self.num_clones += 1

        labels = self.graph.vs["label"] + [label + "-" + str(self.num_clones)
                                           for label in g.graph.vs["label"]]
        names = self.graph.vs["name"] + g.graph.vs["name"]
        weights = self.graph.es["weight"] + g.graph.es["weight"]
        actions = self.graph.es["action"] + g.graph.es["action"]
        if self.prob:
            probs = self.graph.es["prob"] + g.graph.es["prob"]
        else:
            probs = None
        # take the disjoint union of this graph and the given graph
        self.graph = self.graph.disjoint_union(g.graph)
        # reinstate edge and vertex attributes
        self.graph.vs["label"] = labels
        self.graph.vs["name"] = names
        self.graph.es["weight"] = weights
        self.graph.es["action"] = actions
        if probs:
            self.graph.es["prob"] = probs
        # properties = [(e.source, e["action"], e["weight"]) for e in es]
        # for each edge, make a replacement edge to new graph
        for edge in es:
            new_edge = self.graph.add_edge(edge.source, self.graph.vs[v_mod])
            new_edge["action"] = edge["action"]
            new_edge["weight"] = edge["weight"]
            if probs:
                new_edge["prob"] = edge["prob"]
        # delete the edges
        if len(self.q_previous) > 0:
            # self.graph.delete_edges(_target=target, _source_notin=[self.q_previous[-1]])
            self.graph.delete_edges(_target=target)
            # self.graph.delete_vertices(VertexSeq(self.graph, target))
        else:
            self.graph.delete_edges(_target=target)

        return self

    def optimal(self, discount, best=True, punish=-1000, steps=100):
        mod = 1
        if not best:
            mod = -1
        tr = self.to_mdp(best, punish)
        sol = mdp.ValueIteration(tr[0], tr[1], discount)
        sol.run()
        return sol.V[self.q0] * mod

    def to_mdp(self, best=True, punish=-1000):
        """
        solve graph as MDP for most (or least) optimal strategy and return value

        :param best:
        :param punish:
        :return:
        """
        vcount = self.graph.vcount()
        ecount = self.graph.ecount()
        # t represents the transition probabilities for each "action" from one
        # "state" to another "state, where every action is associated with a
        # transition, and every state is represented by a vertex.
        # The matrix may be considered, along the "action" vertex, as specifying
        # the probability that action has of moving the process from state A
        # to state B. As we are treating each transition as a sure thing,
        # in the case that we are evaluating a DAU automaton, all
        # probabilities are 1. E.g. if edge 2 in the graph points from vertex 3
        # to vertex 1, then the entry t[2, 3, 1] = 1. The t matrix must be row
        # stochastic, so there must be an entry of 1 in each row; i.e. for each
        # source state, there must be some effect on the process if the process
        # takes that edge - even if the edge is not connected to the source.
        # In the given example, t[2, 3, j] = 0 where j != 1, since it is clear
        # that taking edge 2 from 3 to any edge other than 1 is impossible,
        # however t[2, j, j] = 1 where j != 3, since there *must* be an action
        # for each edge-state pair. Due to this discrepancy in representations,
        # the only reasonable choice is to say that trying to take edges that do
        # not begin from the current vertex leaves you in the current vertex.
        # Letting the process "wait" by "taking" an edge not connected to the
        # current state can be problematic when there are negative weights.
        # If there are negative weights, the MDP seems to want to wait unless
        # it gets punished for doing so, since 0 reward is better than -x.
        # To prevent this behavior, rewards for taking actions that correspond
        # with moving on edges not connected to the current vertex are deeply
        # negative. If the range of rewards for the automaton are lower than
        # this value, it will have to be changed, so it's a parameter with
        # default of -1000.
        # The r matrix, itself, has the same shape as the t matrix, and each
        # entry in r provides the reward for taking the transition in t that has
        # the same index. E.g. if moving from vertex 3 to vertex 1 via edge 2
        # has the reward of 5, then r[2, 3, 1] = 5. For "wait" actions, the
        # reward is equal to the punish value, e.g. r[2, j, j] = -1000 where
        # j != 1. NOTE: Currently, all elements of r are set to a value (either
        # a valid weight, or the punish value). Rewards associated with elements
        # of t where the transition probability is 0 may also be set to 0 if we
        # want to switch to sparse matrix representations.
        if self.prob:
            actions = set(self.graph.es.get_attribute_values("action"))
            t = np.zeros((len(actions), vcount, vcount))
            r = np.full((len(actions), vcount, vcount), punish)
        else:
            t = np.zeros((ecount, vcount, vcount))
            r = np.full((ecount, vcount, vcount), punish)
        # mod negates the weights of the system if we're looking for the worst
        # possible execution (except punishment weights, otherwise the system
        # would do nothing at all).
        mod = 1
        if not best:
            mod = -1

        if self.prob:
            # we're dealing with an actual MDP! Construct the mdp differently
            # start by getting a list of all actions
            actions = set(self.graph.es.get_attribute_values("action"))
            # for each action...
            for i, action in enumerate(actions):
                # find the edges that are in that action
                edges = self.graph.es.select(action_eq=action)
                # for each such edge...
                for j, edge in enumerate(edges):
                    tup = edge.tuple
                    # set the transition probability for action i for this edge to its prob
                    t[i, tup[0], tup[1]] = edge["prob"]
                    r[i, tup[0], tup[1]] = edge["weight"] * mod
                # sometimes an action may not be conditionally possible, so check for rows in t that are all zeros
                # in which case create a self-transition with probability 1 and reward = punish
                for k in range(vcount):
                    if sum(t[i, k]) == 0:
                        t[i, k, k] = 1
                        r[i, k, k] = punish

        else:
            # This loop iterates through the edges in the graph so each transition
            # matrix can be provided for every edge.
            # for each edge...
            for i, edge in enumerate(self.graph.es):
                tup = edge.tuple
                # ... for each vertex considered as source...
                for j in range(vcount):
                    # ... if this vertex actually is the source of this edge...
                    if j == tup[0]:
                        # ... the transition probability from source to target is 1
                        t[i, tup[0], tup[1]] = 1
                    else:
                        # ... otherwise, taking this edge is a "wait" action.
                        t[i, j, j] = 1
                # ... change the reward corresponding to actually taking the edge.
                r[i, tup[0], tup[1]] = edge["weight"] * mod
        return (t, r)

    def checkCTL(self, file, x, verbose=False):
        """
        Checks the automaton for a given CTL specification

        :param file:
        :param x:
        :param verbose:
        :return:
        """
        # convert graph to nuXmv model
        self.convertToNuXmv(file, x)
        # nuxmv = "nuXmv"
        # TODO: extract this and make it easier to change
        # nuxmv = "E:\\Programs\\nuXmv-2.0.0-win64\\bin\\nuXmv.exe"
        nuxmv = "/home/colin/Downloads/nuXmv-2.0.0-Linux/bin/nuXmv"

        # with open("cmd.txt", 'w') as f:
        #     f.write("read_model -i " + file + "\n")
        #     f.write("flatten_hierarchy\n")
        #     f.write("encode_variables\n")
        #     f.write("build_model\n")
        #     f.write("check_ctlspec -p \"" + x + "\"")

        # out = subprocess.run([nuxmv, "-source", "cmd.txt", file], shell=True, stdout=subprocess.PIPE)
        # out = subprocess.run([nuxmv, file], shell=True, stdout=subprocess.PIPE)
        out = subprocess.run([nuxmv, file], stdout=subprocess.PIPE)
        check = "true" in str(out.stdout)
        if verbose:
            print(out.stdout)
        return check

    def checkLTL(self, file, x, verbose=False):
        """
        Checks the automaton for a given LTL specification

        :param file:
        :param x:
        :param verbose:
        :return:
        """
        # convert graph to nuXmv model
        self.convertToNuXmv(file, x, lang="LTL")
        # TODO: extract this and make it easier to change
        # nuxmv = "E:\\Programs\\nuXmv-2.0.0-win64\\bin\\nuXmv.exe"
        nuxmv = "/home/colin/Downloads/nuXmv-2.0.0-Linux/bin/nuXmv"
        out = subprocess.run([nuxmv, file], stdout=subprocess.PIPE)
        check = "true" in str(out.stdout)
        if verbose:
            print(out.stdout)
        return check

    def checkPCTL(self, m_file, p_file, x, verbose=False):
        """
        Checks the automaton for a given PCTL specification

        :param m_file:
        :param p_file:
        :param x:
        :param verbose:
        :return:
        """
        # convert graph to PRISM model
        # TODO: switch to Storm to minimize IO operations for (potential) massive speed-up
        self.convertToPRISM(m_file, p_file, x)
        # TODO: extract this and make it easier to change
        prism = "/home/colin/prism-4.7-linux64/bin/prism"
        out = subprocess.run([prism, m_file, p_file], stdout=subprocess.PIPE)
        check = "true" in str(out.stdout)
        if verbose:
            print(out.stdout)
        return check

    # TODO: checkLTL by lang="LTL", tag a spec with CTL or LTL
    def checkToCTL(self, file, x, negate=False, verbose=False):
        """
        Checks an automaton for a CTL specification, given an LTL specification.

        :param file:
        :param x:
        :param negate:
        :param verbose:
        :return:
        """
        if negate:
            return self.checkCTL(file, '!E' + x, verbose=verbose)
        else:
            return self.checkCTL(file, 'A' + x, verbose=verbose)

    def convertToNuXmv(self, file, x, lang="CTL"):
        """
        Produces a NuXmv input file specifying this automaton.
        :param file:
        :param x:
        :return:
        """
        with open(file, 'w') as f:
            f.write("MODULE main\n\n")
            self._writeStatesNuXmv(f)

            self._writeNamesNuXmv(f)

            self._writeVarsNuXmv(f)

            # begin ASSIGN constraint for state and name transitions
            f.write("ASSIGN\n")

            # States:
            self._writeStateTransNuXmv(f)

            # Names:
            self._writeNameTransNuXmv(f)

            # Properties:
            self._writePropTransNuXmv(f)

            # Specification
            f.write(lang.upper() + "SPEC " + x + ";")
            f.write("\n")

    def _writeStatesNuXmv(self, f):
        sep = ', '
        # include each vertex as a state in the model
        states = [str(v.index) for v in self.graph.vs]
        states = sep.join(states)
        f.write("VAR state: {" + states + "};\n\n")

    def _writeNamesNuXmv(self, f):
        sep = ', '
        # since multiple states can be associated with the same state of a
        # smaller original automaton, we want to track what that original
        # state is with a name variable
        names = self.graph.vs["name"]
        # remove duplicates from names
        names = list(set(names))
        # add names variable to model
        names = sep.join(names)
        f.write("VAR name: {" + names + "};\n\n")

    def _writeStateTransNuXmv(self, f):
        sep = ', '
        # set initial state
        f.write(" init(state) := " + str(self.q0) + ";\n")
        # define state transitions
        f.write(" next(state) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get a string representation of all the vertex's successors
            next_v = [str(vx.index) for vx in v.neighbors(mode=OUT)]
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case
            if next_v:
                next_v = sep.join(next_v)
                f.write("   state = " + state + " : {" + next_v + "};\n")

        # default case
        f.write("   TRUE : state;\n")
        f.write("  esac;\n")
        f.write("\n")

    def _writeNameTransNuXmv(self, f):
        # set initial name
        init_name = self.graph.vs["name"][self.q0]
        f.write(" init(name) := " + str(init_name) + ";\n")
        # define name transitions
        f.write(" next(name) :=\n")
        f.write("  case\n")
        # for each vertex...
        for v in self.graph.vs:
            # ... get that vertex's name
            v_name = v["name"]
            # and a string rep of this vertex
            state = str(v.index)
            # and write out the transitions to the case based on next state
            f.write("   next(state) = " + state + " : " + v_name + ";\n")
        # default case
        f.write("   TRUE : name;\n")
        f.write("  esac;\n")
        f.write("\n")

    def _writeVarsNuXmv(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter var
            c_name = str(self.counter[0])
            start = str(self.counter[1])
            end = str(self.counter[2])
            f.write("VAR " + c_name + " : " + start + " .. " + end + ";\n\n")

        # if auto has labels
        if 'label' in self.graph.vs.attributes():
            # ... then write the label var
            labels_size = str(len(self.propositions))
            f.write("VAR label: unsigned word[" + labels_size + "];\n\n")

    def _writePropTransNuXmv(self, f):
        # if auto has a counter
        if self.counter:
            # ... then write the counter transitions
            c = str(self.counter[0])
            t = str(self.counter[2])
            f.write(" init(" + c + ") := " + str(self.counter[1]) + ";\n")
            f.write(" next(" + c + ") := (" + c + "<" + t + ")?(" + c + "+1):(" +
                    c + ");\n\n")

        # if auto has labels
        if 'label' in self.graph.vs.attributes():
            # set up translation between label strings and bit strings.
            # bit strings get used in nuXmv to represent which propositions a state has.
            # e.g. if the language has 4 propositions ['p', 'q', 'r', 's'], and state 0 has propositions "p, s",
            # then the bit string representing the labels of state 0 would be 0ub8_1001.
            word_size = len(self.propositions)
            no_labels = [0] * len(self.propositions)
            prop_dict = {}
            for i, prop in enumerate(self.propositions):
                label_bits = copy(no_labels)
                label_bits[i] = 1
                prop_dict[prop] = label_bits

            # set initial label
            # get the string representation of state labels, e.g. "p, s"
            init_props = self.graph.vs["label"][self.q0].split(', ')
            # get a list of bit string representations for each proposition, e.g. [[1, 0, 0, 0], [0, 0, 0, 1]]
            init_props_bits = [prop_dict[prop] for prop in init_props]
            # combine those bit string representations into a single bit string, e.g. ['1', '0', '0', '1']
            init_label = [str(sum(bits)) for bits in zip(*init_props_bits)]
            # join those bits into a string, e.g. "1001"
            init_label_str = ''.join(init_label)
            f.write(" init(label) := 0ub" + str(word_size) + "_" + str(init_label_str) + ";\n")
            # define label transitions
            f.write(" next(label) :=\n")
            f.write("  case\n")
            # for each vertex...
            for v in self.graph.vs:
                # ... get that vertex's propositions
                v_props = v["label"].split(', ')
                # then a list of bits
                v_props_bits = [prop_dict[prop] for prop in v_props]
                # combine those into a bit string
                v_label = [str(sum(bits)) for bits in zip(*v_props_bits)]
                v_label_str = ''.join(v_label)
                # and get a string rep of this vertex
                state = str(v.index)
                # and write out the transitions to the case based on next state
                f.write("   next(state) = " + state + " : 0ub" + str(word_size) + "_" + v_label_str + ";\n")
            # default case
            f.write("   TRUE : label;\n")
            f.write("  esac;\n")
            f.write("\n")

            # define relationship between bit words and original propositions
            f.write("DEFINE\n")
            for prop in self.propositions:
                # build logical equivalence string
                logical_bits = prop_dict[prop]
                logical_bits = ''.join([str(bit) for bit in logical_bits])
                f.write(" " + str(prop) + " := (0ub" + str(word_size) + "_" + logical_bits + " & label) = 0ub" +
                        str(word_size) + "_" + logical_bits + ";\n")
            f.write("\n")

    def convertToPRISM(self, m_file, p_file, x):
        """
        Produces a PRISM model file specifying this automaton, and a properties file specifying the formula
        :param m_file:
        :param p_file:
        :param x:

        :return:
        """
        with open(m_file, 'w') as f:
            f.write("mdp\n")
            f.write("module main\n")

            states = [v.index for v in self.graph.vs]
            names = self.graph.vs["name"]
            names = list(set(names))
            names = [int(namei) for namei in names]
            # make a state variable that goes from 0 to number of states; init q0.id
            f.write("state : [0.." + str(max(states)) + "] init " + str(self.q0) + ";\n")
            # make a name variable that goes from 0 to maximum name; init q0.name
            f.write("name : [0.." + str(max(names)) + "] init " + self.graph.vs["name"][self.q0] + ";\n")
            # for each vertex...
            for v in self.graph.vs:
                # for each action at vertex...
                for k in self.k(v):
                    # initialize command string
                    command = "    [] state=" + str(v.index) + " -> "
                    plus = ""
                    # get the edges from this vertex that are part of this action
                    esi = self.graph.es.select(_source_eq=v, action_eq=k)
                    # for each edge in that action...
                    for e in esi:
                        prob = str(e["prob"])
                        tgt_id = str(e.target_vertex.index)
                        tgt_nm = str(e.target_vertex["name"])
                        command += plus + prob + " : (state'=" + tgt_id + ")&(name'=" + tgt_nm + ")"
                        if not plus:
                            plus = " + "
                    command += ";\n"
                    f.write(command)
                f.write("\n")

            f.write("endmodule")

        with open(p_file, 'w') as f:
            # Specification
            f.write(x)
            f.write("\n")


class Obligation(object):
    """
    Contains an obligation in Dominance Act Utilitarian deontic logic
    """

    def __init__(self, phi, is_ctls, is_neg, is_pctl=False):
        """
        Creates an Obligation object

        :param phi:
        :param is_ctls:
        :param is_neg:
        """
        self.phi = phi
        self.is_ctls = is_ctls
        self.is_neg = is_neg
        self.is_stit = not is_ctls
        self.is_pctl = is_pctl
        self.phi_neg = False

    @classmethod
    def fromCTL(cls, phi):
        """
        Creates an Obligation object from a CTL string

        :param phi:
        :return:
        """
        return cls(phi, True, False)

    @classmethod
    def fromPCTL(cls, phi):
        """
        Creates an Obligation object from a PCTL string

        :param phi:
        :return:
        """
        return cls(phi, False, False, True)

    def isCTLS(self):
        """
        Checks if obligation is a well-formed CTL* formula

        :return:
        """
        # TODO: use grammar to check this
        return self.is_ctls

    def isPCTL(self):
        """
        Checks if obligation is a well-formed PCTL formula

        :return:
        """
        # TODO: use grammar to check this
        return self.is_pctl

    def isSTIT(self):
        """
        Checks if obligation is a well-formed dstit statement

        :return:
        """
        # TODO: use grammar to check this
        return self.is_stit

    def isNegSTIT(self):
        """
        Checks if obligation is of the form ![alpha dstit: phi]

        :return:
        """
        return self.is_stit and self.is_neg

    def getPhi(self):
        """
        Gets the inner formula of the obligation

        :return:
        """
        return self.phi


def checkObligation(g, a, verbose=False):
    """
    Check an automaton for if it has a given obligation.

    :param g:
    :param a:
    :param verbose:
    :return:
    """
    # return checkConditional with trivial condition params
    return checkConditional(g, a, "TRUE", 0, verbose=verbose)


# TODO: refactor checkConditional into smaller functions so I can use some of the juicy bits elsewhere
def checkConditional(g, a, x, t, verbose=False):
    """
    Check an automaton for if it has a given obligation under a given condition.

    :param g:
    :param a:
    :param x:
    :param t:
    :param verbose:
    :return:
    """
    optimal = get_optimal_automata(g, t, x, verbose)

    for m in optimal:
        truth_n = True
        if verbose:
            print(m[0])
        if a.isCTLS():
            # truth_n = m[1].checkToCTL('temp.smv', a.getPhi(), a.phi_neg,
                                      # verbose=verbose)
            truth_n = m[1].checkCTL('temp.smv', a.getPhi(), a.phi_neg)
        elif a.isPCTL():
            truth_n = m[1].checkPCTL('temp.sm', 'temp.pctl', a.getPhi())
        elif a.isSTIT():
            phi = a.getPhi()
            if not a.isNegSTIT():
                delib = not g.checkToCTL('temp.smv', phi, a.phi_neg,
                                         verbose=verbose)
                guaranteed = m[1].checkToCTL('temp.smv', phi, a.phi_neg,
                                             verbose=verbose)
                if verbose:
                    print("deliberate: ", delib)
                    print("guaranteed: ", guaranteed)
                truth_n = delib and guaranteed
            else:
                not_delib = g.checkToCTL('temp.smv', phi, a.phi_neg,
                                         verbose=verbose)
                guaranteed = m[1].checkToCTL('temp.smv', phi, a.phi_neg,
                                             verbose=verbose)
                if verbose:
                    print("not deliberate: ", not_delib)
                    print("not guaranteed: ", not guaranteed)
                truth_n = not_delib or not guaranteed
        else:
            raise ValueError(
                'The given obligation was not a well formed CTL* formula, ' +
                'nor a well formed deliberative STIT statement.',
                a)
        if not truth_n:
            return False

    return True


# TODO: consider returning a list of dictionaries
# TODO: easier evaluation of automaton value when x="TRUE", so use that case
def get_choice_automata(g, t, x="TRUE", return_fragments=False):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (action, act_automaton, interval); where action is an action of the
    automaton g available at g.q0 (the starting state of g), act_automaton is the
    automaton generated when g can only take the corresponding action from q0, and
    interval is a list containing the highest and lowest scores of histories of length t
    produced from the act_automaton.

    :param g:
    :param t:
    :param x:
    :param return_fragments:
    :return:
    """
    root = g.q0
    choices = g.k(root)
    out = []
    frags = []
    l = len(choices)
    discount = 0.5
    # for each choice available from start...
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn, source=root)
        gnr = deepcopy(gn)
        gnr.q_previous.append(-1)
        gnp = gnr.union(g, target=root)
        if x == "TRUE" and not return_fragments:
            # skip the hard stuff
            q_of_kn = gnp.optimal(discount)
            out.append((kn, gnp, q_of_kn))
            continue
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        if t <= 0:
            t += 1
        gns = generate_fragments(gnp, g, root, x, t)
        lows = []
        highs = []
        if gns:
            # there are condition-satisfying histories, so gnp is in choice/|x|
            if g.prob:
                # we're dealing with a probabilistic automaton, get the conditional automaton
                cond_auto = build_fragment_tree(gns, g)
                q_of_kn = cond_auto.optimal(discount)
                out.append((kn, gnp, q_of_kn))
                for gf in gns:
                    frags.append((kn, gf))
            else:
                for gf in gns:
                    lows.append(gf.optimal(discount, best=False))
                    highs.append(gf.optimal(discount, best=True))
                    frags.append((kn, gf))
                interval = [np.max(lows), np.max(highs)]
                out.append((kn, gnp, interval))
        else:
            raise RuntimeError("No fragments found: maybe the condition is not satisfiable, or transitions are missing")

    if return_fragments:
        return out, frags
    else:
        return out


def get_choice_fragments(g, t, x="TRUE"):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (action, act_fragment); where action is an action of the
    automaton g available at g.q0 (the starting state of g), act_fragment is the
    fragment generated when g can only take the corresponding action from q0 and has only
    one history up to depth t.

    :param g:
    :param t:
    :param x:
    :return:
    """
    root = g.q0
    choices = g.k(root)
    out = []
    l = len(choices)
    # for each choice available from start...
    for n in np.arange(l):
        kn = choices[n]
        gn = deepcopy(g)
        gn = gn.forceKn(kn, source=root)
        gnr = deepcopy(gn)
        gnr.q_previous.append(-1)
        gnp = gnr.union(g, target=root)
        # get a list of automata whose first action is kn, and have one history
        # up to depth t, and that history satisfies X, and after that it behaves
        # like g
        if t <= 0:
            t += 1
        gns = generate_fragments(gnp, g, root, x, t)
        if gns:
            for gf in gns:
                out.append((kn, gf))

    return out


def get_optimal_automata(g, t, x="TRUE", verbose=False):
    """
    given an automaton g, a time horizon t, and a horizon-limited condition x, generate:
    a list of tuples (opt_action, opt_automaton) where opt_action is an action available
    to g at g.q0 (the starting state of g), and opt_automaton is the automaton generated
    when g can only take the corresponding opt_action from q0. opt_action is the
    dominance optimal action available to g.q0.

    :param g:
    :param t:
    :param x:
    :param verbose:
    :return:
    """
    choices = get_choice_automata(g, t, x)
    return choose_optimal_automata(choices, verbose)


def choose_optimal_automata(choices, verbose=False):
    """
    given a list of tuples (action, act_automaton, interval), generate:
    a list of tuples (opt_action, opt_automaton) where opt_action is an action available
    to g at g.q0 (the starting state of g), and opt_automaton is the automaton generated
    when g can only take the corresponding opt_action from q0. opt_action is the
    dominance optimal action available to g.q0.

    :param choices:
    :param verbose:
    :return:
    """
    intervals = [choice[2] for choice in choices]

    optimal = []
    if not intervals:
        if verbose:
            print("No Intervals")
        return False

    if choices[0][1].prob:
        # find all automata whose expected utility is the best
        v_max = np.max(intervals)
        for i, interval in enumerate(intervals):
            if interval >= v_max:
                if verbose:
                    print(choices[i][0], interval)
                optimal.append((choices[i][0], choices[i][1]))
    else:
        # find all un-dominated intervals
        # optimal carries tuples containing an optimal action and an automaton
        # whose first action is that optimal action.
        inf = np.max(np.min(intervals, axis=1))
        for i, interval in enumerate(intervals):
            if interval[1] >= inf:
                if verbose:
                    print(choices[i][0], interval)
                optimal.append((choices[i][0], choices[i][1]))

    return optimal


def generate_fragments(gn, g0, q0, x, t, check_at_end=True):
    """
    Given an Automaton gn, a prototype Automaton g0, a starting state q0,
    a finite horizon condition x, and the length of that horizon t, generate
    a list of all Automata that start from q0 and have only one history up to
    depth t, that history satisfies x, and after t the Automaton behaves like
    g0.

    If check_at_end is true, then generate_fragments will generate all possible
    fragments of length t, and check each for satisfaction of x. Otherwise, it
    checks each as they're built.

    :param gn:
    :param g0:
    :param q0:
    :param x:
    :param t:
    :param check_at_end:
    :return:
    """

    g = deepcopy(gn)
    # set a clock on the automaton so the condition can be horizon limited
    g.setCounter(var_name="fragmentc")
    # set up the condition to be checked in each step
    f = "E [" + x + " U " + "(fragmentc = " + str(t) + ")]"
    # f = 'E' + x
    # make sure we start from the right state
    g.qn = q0
    # initialize the list of systems with the given system
    systems = [g]
    # until we reach the given horizon...
    for i in trange(t):
        new_systems = []
        # ... for every system we have so far...
        for system in systems:
            # ... get each possible transition for that system from its current state...
            possible_edges = system.graph.es.select(_source=system.qn)
            # ... and for each possible transition...
            for edge in possible_edges:
                state = edge.tuple[1]
                # copy the system
                sys_n = deepcopy(system)
                # force the transition
                sys_n = sys_n.forceEn(edge, source=system.qn)
                sys_n_ren = deepcopy(sys_n)
                # update the list of previous states
                sys_n_ren.q_previous.append(sys_n_ren.qn)
                # tack the prototype system onto the end
                sys_n_prime = sys_n_ren.union(g0, state)
                # if this new system satisfies the condition...
                if not check_at_end:
                    if sys_n_prime.checkCTL("temp.smv", f):
                        # set the system's current state to the only possible next state
                        sys_n_prime.qn = sys_n_prime.graph.neighbors(sys_n_prime.qn, mode=OUT)[0]
                        # and add the system to our list of systems.
                        new_systems.append(sys_n_prime)
                else:
                    # set the system's current state to the only possible next state
                    sys_n_prime.qn = sys_n_prime.graph.neighbors(sys_n_prime.qn, mode=OUT)[0]
                    # and add the system to our list of systems.
                    new_systems.append(sys_n_prime)
        # all systems have been stepped through, and the satisfactory systems
        # get to make it to the next round.
        systems = new_systems
    # now that all the systems in our list are deterministic to depth t
    # we cut the "chaff" from each automaton, because we added a lot of superfluous states and transitions
    # so, for each system...
    good_systems = []
    for system in tqdm(systems):
        # find out what the identifier is for the last prototype we tacked on
        clone_no = "-" + str(system.num_clones)
        # get all the old vertices we want to be keeping
        path = system.q_previous
        path.append(system.qn)
        # and set up a list to retain the vertices we want to delete
        del_v_id = []
        # then for each vertex in our graph...
        for v in system.graph.vs:
            # if that vertex is not one we want to keep and it's not in our last prototype tacked on...
            if v.index not in path and clone_no not in v["label"]:
                # add it to our delete list
                del_v_id.append(v.index)
        # make our delete list a proper vertex sequence so it can be deleted
        del_vs = VertexSeq(system.graph, del_v_id)
        # and delete those vertices! (And any associated edges)
        system.graph.delete_vertices(del_vs)
        # set the q0 of this pared-down system to what it should be
        system.q0 = system.graph.vs.find(name=str(q0)).index
        system.t = t
        if check_at_end:
            if system.checkCTL("temp.smv", f):
                good_systems.append(system)

    if not check_at_end:
        good_systems = systems
    return good_systems


def build_fragment_tree(fragments, g0):
    """
    Build a tree from a given set of fragments such that each leaf of the tree is the end of one of the
    fragments, and each leaf leads into g0 - the original automaton.
    Return an Automaton based on the tree.

    When the fragments are generated with a condition, then the tree may be considered to be a conditional
    automaton.

    :param fragments:
    :param g0:
    :return:
    """
    q0 = fragments[0].q0
    t = fragments[0].t
    g_new = Graph(directed=True)
    v_new = g_new.add_vertex(name=fragments[0].graph.vs[q0]["name"])
    # propagate other attributes of v0
    for attr in fragments[0].graph.vs[q0].attribute_names():
        v_new[attr] = fragments[0].graph.vs[q0][attr]
    # track the fragments on each front of the tree
    frag_partitions = [[fragments, q0]]
    # track the front of the tree
    front_vs = [q0]
    visited = [q0]
    prob = False
    # for each level of the tree....
    for _ in range(t):
        # for each branch (starting from none)...
        new_front = []
        new_partitions = []
        partition_assignment = {}
        partition_idx = 0
        for frag_partition, qn in frag_partitions:
            # get each edge from this front of the tree...
            actions = []
            unique_edges = []
            signatures = []
            # for each history in this branch...
            for fragment in frag_partition:
                edge = fragment.graph.es.find(_source=fragment.q0)
                # add that edge's action to the list of actions
                actions.append(edge["action"])
                # get that edge's action
                target = edge.target_vertex
                # determine the proper signature of the edge
                if "prob" in edge.attribute_names():
                    prob = True
                    edge_sig = (qn, edge["action"], edge["weight"], edge["prob"], target["name"])
                else:
                    edge_sig = (qn, edge["action"], edge["weight"], target["name"])
                if edge_sig not in signatures:
                    # if we haven't seen this signature before, record it
                    signatures.append(edge_sig)
                    unique_edges.append(edge)
                    # remember where we put fragments with this signature
                    partition_assignment[edge_sig] = partition_idx
                    # put this fragment in its partition
                    new_partitions.append([[fragment], None])
                    # remember where to put the next fragment with a new signature
                    partition_idx += 1
                else:
                    # we've seen this signature before, so we know where to put its fragment
                    temp_part_idx = partition_assignment[edge_sig]
                    new_partitions[temp_part_idx][0].append(fragment)
                # now set the fragment's q0 to its next state
                fragment.q0 = target.index
            # get the set of actions among those edges...
            actions = set(actions)
            # for each action...
            for action in actions:
                # get the edges in that action
                act_es = [edge for edge in unique_edges if edge["action"] == action]
                act_probs = []
                if prob:
                    act_probs = [edge["prob"] for edge in act_es]
                    if np.sum(act_probs) == 0:
                        # there are no edges in this action, so

                        pass
                for edge in act_es:
                    # find target label
                    target = edge.target_vertex
                    # determine the signature of the edge
                    if prob:
                        edge_sig = (qn, edge["action"], edge["weight"], edge["prob"], target["name"])
                    else:
                        edge_sig = (qn, edge["action"], edge["weight"], target["name"])
                    temp_part_idx = partition_assignment[edge_sig]
                    # add it to the new graph, starting with the target
                    v_new = g_new.add_vertex()
                    visited.append(v_new.index)
                    new_front.append(v_new.index)
                    new_partitions[temp_part_idx][1] = v_new.index
                    # copy the attributes
                    for attr in target.attribute_names():
                        v_new[attr] = target[attr]
                    # now add the edge
                    e_new = g_new.add_edge(qn, v_new)
                    # copy the attributes
                    for attr in edge.attribute_names():
                        e_new[attr] = edge[attr]
                    # normalize the probability
                    if prob:
                        e_new["prob"] = e_new["prob"] / np.sum(act_probs)
        front_vs = new_front
        frag_partitions = new_partitions

    # turn the graph we've built into an automaton
    g_new.vs["target"] = [False] * g_new.vcount()
    g0.graph.vs["target"] = [True] * g0.graph.vcount()
    g_new = g_new.disjoint_union(g0.graph)
    move_edges = g_new.es.select(_target_in=front_vs)
    mv_edg_tuples = []
    mv_edg_attr = {}
    for attr in move_edges.attribute_names():
        mv_edg_attr[attr] = []
    for move_edge in move_edges:
        target_name = move_edge.target_vertex["name"]
        move_target = g_new.vs.select(name=target_name, target=True)[0]
        new_tuple = (move_edge.tuple[0], move_target.index)
        mv_edg_tuples.append(new_tuple)
        for attr in move_edge.attribute_names():
            mv_edg_attr[attr].append(move_edge[attr])
    g_new.delete_edges(move_edges)
    g_new.add_edges(mv_edg_tuples, mv_edg_attr)
    del_vs = VertexSeq(g_new, front_vs)
    g_new.delete_vertices(del_vs)

    cond_auto = Automaton(g_new, q0)
    cond_auto.q_previous = visited
    cond_auto.num_clones = t
    # cond_auto.graph = cond_auto.graph.disjoint_union(g0.graph)

    for fragment in fragments:
        fragment.q0 = q0

    return cond_auto
