import itertools
import copy
import networkx as nx
import math
import random
import time

ids = ["209317239", "208165969"]

MARINE_COLLISION_PENALTY = 1


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.map = initial["map"]
        # self.treasures = initial['treasures']
        self.V = 0
        self.Q = 0
        first_pirate_ship_info = next(iter(initial['pirate_ships'].values()))
        self.base = first_pirate_ship_info['location']
        # Initialize pirate ships with your new structure
        pirate_ships_state = {}
        for ship_name, ship_info in initial['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        # Initialize the initial_state tuple
        self.initial_state = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in initial['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        self.initial_state += (marine_ships_state,)
        self.initial_state += (initial['treasures'],)
        self.turns_to_go = initial["turns to go"]

        self.cinitial = copy.deepcopy(self.initial)
        self.num_of_tres = 0
        self.num_of_pirates = 0
        self.num_of_marins = 0
        self.deleted_ships = []
        self.deleted_marines = []

        for t in self.initial['treasures'].keys():
            self.num_of_tres += 1

        for m in self.initial['marine_ships'].keys():
            self.num_of_marins += 1

        for ship in self.initial['pirate_ships'].keys():
            self.num_of_pirates += 1

        # only one ship
        for ship in self.initial['pirate_ships']:
            if self.num_of_pirates > 1:
                self.deleted_ships.append(ship)
                del self.cinitial['pirate_ships'][ship]
                self.num_of_pirates -= 1

        # only one marine
        for marin in self.initial['marine_ships']:
            pgl = len(self.cinitial['marine_ships'][marin]['path'])
            if self.num_of_marins > 1:
                self.deleted_marines.append(marin)
                del self.cinitial['marine_ships'][marin]
                self.num_of_marins -= 1

        self.forbidden_tiles = []
        for marin in self.deleted_marines:
            for index in self.initial['marine_ships'][marin]['path']:
                self.forbidden_tiles.append(index)


        # only one treasure

        for tres in self.initial['treasures']:
            if self.num_of_tres > 1:
                del self.cinitial['treasures'][tres]
                self.num_of_tres -= 1

        # Initialize the cinitial tuple

        pirate_ships_state = {}
        for ship_name, ship_info in self.cinitial['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        self.tuple_cinitial = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in self.cinitial['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        self.tuple_cinitial += (marine_ships_state,)
        self.tuple_cinitial += (self.cinitial['treasures'],)
        self.S = self.generate_all_states(self.tuple_cinitial)
        self.initial_state = self.tuple_cinitial
        self.VI()

    def action_per_ship(self, ship, state):
        n = len(self.map) - 1
        m = len(self.map[0]) - 1
        ship_loc = state[0][ship]['location']
        ac_list = []
        if (ship_loc[0] + 1) <= n and self.map[ship_loc[0] + 1][ship_loc[1]] != 'I' and (
        ship_loc[0] + 1, ship_loc[1]) not in self.forbidden_tiles:
            ac_list.append(('sail', ship, (ship_loc[0] + 1, ship_loc[1])))
        if (ship_loc[1] + 1) <= m and self.map[ship_loc[0]][ship_loc[1] + 1] != 'I' and (
        ship_loc[0], ship_loc[1] + 1) not in self.forbidden_tiles:
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] + 1)))
        if (ship_loc[0] - 1) >= 0 and self.map[ship_loc[0] - 1][ship_loc[1]] != 'I' and (
        ship_loc[0] - 1, ship_loc[1]) not in self.forbidden_tiles:
            ac_list.append(('sail', ship, (ship_loc[0] - 1, ship_loc[1])))
        if (ship_loc[1] - 1) >= 0 and self.map[ship_loc[0]][ship_loc[1] - 1] != 'I' and (
        ship_loc[0], ship_loc[1] - 1) not in self.forbidden_tiles:
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] - 1)))

        if state[0][ship]['capacity'] > 0:
            for tres, tres_value in state[2].items():
                tres_pos = tres_value['location']
                if (ship_loc[0] + 1) <= n:
                    if (ship_loc[0] + 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] + 1) <= m:
                    if (ship_loc[0], ship_loc[1] + 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[0] - 1) >= 0:
                    if (ship_loc[0] - 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] - 1) >= 0:
                    if (ship_loc[0], ship_loc[1] - 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))

        if state[0][ship]['capacity'] < self.initial_state[0][ship]['capacity']:
            if ship_loc == self.base:
                ac_list.append(('deposit', ship))

        ac_list.append(('wait', ship))
        # ac_list.append('reset')
        # ac_list.append('terminate')
        return ac_list

    def yield_all_actions(self, ship_actions):
        ship_action_lists = list(ship_actions.values())
        for combination in itertools.product(*ship_action_lists):
            yield combination

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_ac = {}
        for ship in state[0]:
            dict_ac[ship] = self.action_per_ship(ship, state)

        for action_combination in self.yield_all_actions(dict_ac):
            yield action_combination
        yield 'terminate'
        yield 'reset'

    def action_per_marine(self, marine, state):
        loc = state[1][marine]['index']
        path_len = len(state[1][marine]['path'])
        poss_ac = []
        if path_len == 1:
            poss_ac.append({marine: (0, 1)})
        else:
            if loc == 0:
                poss_ac.append({marine: (0, 0.5)})
                poss_ac.append({marine: (1, 0.5)})
            elif loc == path_len - 1:
                poss_ac.append({marine: (loc, 0.5)})
                poss_ac.append({marine: (loc - 1, 0.5)})
            else:
                poss_ac.append({marine: (loc, 1 / 3)})
                poss_ac.append({marine: (loc - 1, 1 / 3)})
                poss_ac.append({marine: (loc + 1, 1 / 3)})
        return poss_ac

    def action_per_treasure(self, treasure, state):
        loc = state[2][treasure]['location']
        poss_loc = state[2][treasure]['possible_locations']
        len_path = len(poss_loc)
        prob_change = state[2][treasure]['prob_change_location']
        p = prob_change / len_path
        poss_ac = []
        for l in poss_loc:
            if l == loc:
                poss_ac.append({treasure: (l, (1 - prob_change) + p)})
            else:
                poss_ac.append({treasure: (l, p)})
        return poss_ac

    def yield_all_states(self, states):
        states_lists = list(states.values())
        for combination in itertools.product(*states_lists):
            yield combination

    def poss_states_marines_treasures(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_states = {}
        for marine in state[1]:
            dict_states[marine] = self.action_per_marine(marine, state)
        for treasure in state[2]:
            dict_states[treasure] = self.action_per_treasure(treasure, state)

        for action_combination in self.yield_all_actions(dict_states):
            yield action_combination

    def apply_action_sail(self, state, ac, reward):
        state[0][ac[1]]['location'] = ac[2]  # update pirate loc
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            if ac[2] == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_collect(self, state, ac, reward):
        state[0][ac[1]]['capacity'] -= 1
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_deposit(self, state, ac, reward):
        num_tre = self.initial_state[0][ac[1]]['capacity'] - state[0][ac[1]]['capacity']
        reward += 4 * num_tre
        state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
        return reward

    def apply_action_wait(self, state, ac, reward):
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def all_possible_next_states(self, state, action):

        RESET_PENALTY = 2
        reward = 0
        possible_next_states = []
        new_state = copy.deepcopy(state)
        for ac in action:
            if ac == "reset":
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac == 'terminate':
                self.turns_to_go = 0
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac[0] == 'sail':
                reward = self.apply_action_sail(new_state, ac, reward)
            if ac[0] == "collect":
                reward = self.apply_action_collect(new_state, ac, reward)
            if ac[0] == "deposit":
                reward = self.apply_action_deposit(new_state, ac, reward)
            if ac[0] == 'wait':
                reward = self.apply_action_wait(new_state, ac, reward)

        possible = self.poss_states_marines_treasures(new_state)
        list_possible = list(possible)  # list of possible combinations of marin treasure locations
        for index_p, p in enumerate(list_possible):
            i = 0
            poss_state = copy.deepcopy(new_state)
            prob_state = 1
            for marine in poss_state[1]:
                poss_state[1][marine]['index'] = p[i][marine][0]
                prob_state *= p[i][marine][1]

                i += 1
            for treasure in poss_state[2]:
                poss_state[2][treasure]['location'] = p[i][treasure][0]
                prob_state *= p[i][treasure][1]
                i += 1

            d = {'state': poss_state, 'prob': prob_state, 'reward': reward}
            possible_next_states.append(d)
        return possible_next_states

    def state_to_tuple(self, state):
        # Assuming state is a tuple of three dictionaries: (pirate_ships_dict, marine_ships_dict, treasures_dict)
        pirate_ships, marine_ships, treasures = state

        # Convert pirate_ships dictionary to a tuple of tuples
        pirate_ships_tuple = tuple(
            (name,
             ('location', info['location']),
             ('capacity', info['capacity']),
             ) for name, info in pirate_ships.items()
        )

        # Convert marine_ships dictionary to a tuple of tuples
        marine_ships_tuple = tuple(
            (name,
             ('index', info['index']),
             ('path', tuple(info['path']))
             ) for name, info in marine_ships.items()
        )

        # Convert treasures dictionary to a tuple of tuples
        treasures_tuple = tuple(
            (name,
             ('location', info['location']),
             ('possible_locations', tuple(info['possible_locations'])),
             ('prob_change_location', info['prob_change_location'])
             ) for name, info in treasures.items()
        )

        # Combine all parts into a single state tuple
        return (pirate_ships_tuple, marine_ships_tuple, treasures_tuple)

    def tuple_to_state(self, state_tuple):
        pirate_ships_tuple, marine_ships_tuple, treasures_tuple = state_tuple

        pirate_ships_dict = {
            name: {
                'location': location,
                'capacity': capacity,
            } for name, (_, location), (_, capacity) in pirate_ships_tuple
        }

        marine_ships_dict = {
            name: {
                'index': index,
                'path': list(path)  # Convert back to mutable list
            } for name, (_, index), (_, path) in marine_ships_tuple
        }

        treasures_dict = {
            name: {
                'location': location,
                'possible_locations': (possible_locations),  # Convert back to mutable list
                'prob_change_location': prob_change_location
            } for name, (_, location), (_, possible_locations), (_, prob_change_location) in treasures_tuple
        }

        return (pirate_ships_dict, marine_ships_dict, treasures_dict)

    def generate_states(self, initial_state):
        # Get all possible 'S' locations for pirate ships
        pirate_ships = initial_state[0]
        marine_ships = initial_state[1]
        treasures = initial_state[2]
        sea_locations = [(i, j) for i, row in enumerate(self.map) for j, cell in enumerate(row) if
                         (cell == 'S' or cell == 'B')]
        # Generate all possible locations for each pirate ship
        pirate_ships_locations = {}
        for name, ship in pirate_ships.items():
            l = []
            for i in range(ship['capacity'] + 1):
                for loc in sea_locations:
                    l.append((loc, i))
            pirate_ships_locations[name] = l
        treasure_states = {name: treasure['possible_locations'] for name, treasure in treasures.items()}
        # # Generate all possible states for marine ships
        marine_ships_states = {}
        for name, ship in marine_ships.items():
            l = []
            for i in range(len(ship['path'])):
                l.append((i, ship['path'][i]))
            marine_ships_states[name] = l

        # Generate all combinations
        all_states = []
        for pirate_combinations in itertools.product(*[pirate_ships_locations[name] for name in pirate_ships]):
            for treasure_combinations in itertools.product(*[treasure_states[name] for name in treasures]):
                for marine_combinations in itertools.product(*[marine_ships_states[name] for name in marine_ships]):
                    # Combine current state configurations
                    combined_state = {
                        'pirate_ships': dict(zip(pirate_ships.keys(), pirate_combinations)),
                        'treasures': dict(zip(treasures.keys(), treasure_combinations)),
                        'marine_ships': dict(zip(marine_ships.keys(), marine_combinations)),
                    }
                    all_states.append(combined_state)
        return all_states

    def generate_all_states(self, initial_state):
        states = self.generate_states(initial_state)
        all_states = []
        for s in states:
            new_state = copy.deepcopy(initial_state)
            for ship in new_state[0]:
                new_state[0][ship]['location'] = s['pirate_ships'][ship][0]
                new_state[0][ship]['capacity'] = s['pirate_ships'][ship][1]
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = s['treasures'][treasure]
            for marine in new_state[1]:
                new_state[1][marine]['index'] = s['marine_ships'][marine][0]
            all_states.append(new_state)
        return all_states

    def VI(self):
        S = self.S
        s_a_prob = {}
        max_iter = self.turns_to_go  # Maximum number of iterations
        S = [s for s in S]
        self.V = {self.state_to_tuple(s): 0 for s in S}
        self.Q = {(self.state_to_tuple(s), i): 0 for s in S for i in
                  range(max_iter + 1)}  # key=(state,turns to go) ,value=(value of VI,best action)
        for i in range(max_iter + 1):
            newV = {self.state_to_tuple(s): 0 for s in S}
            for s in S:
                s_dict = s
                max_val = float('-inf')
                A = self.actions(s_dict)  # all actions that can be performed is state
                best_a = 0
                for a in A:
                    s_a_prob[(self.state_to_tuple(s), a)] = self.all_possible_next_states(s, a)
                    val = s_a_prob[(self.state_to_tuple(s), a)][0]['reward']

                    for state_prob_r in s_a_prob[(self.state_to_tuple(s), a)]:
                        val += state_prob_r['prob'] * self.V[self.state_to_tuple(state_prob_r['state'])]

                    if max_val < val:
                        max_val = val
                        best_a = a
                newV[self.state_to_tuple(s)] = max_val
                self.Q[(self.state_to_tuple(s), i)] = best_a
            self.V = newV

    def act(self, state):
        state1 = copy.deepcopy(state)
        num_of_tres = 0
        num_of_pirates = 0
        num_of_marins = 0
        deleted_ships = []
        deleted_marines = []

        for t in state['treasures'].keys():
            num_of_tres += 1

        for m in state['marine_ships'].keys():
            num_of_marins += 1

        for ship in state['pirate_ships'].keys():
            num_of_pirates += 1

        # only one ship
        for ship in state['pirate_ships']:
            if num_of_pirates > 1:
                deleted_ships.append(ship)
                del state1['pirate_ships'][ship]
                num_of_pirates -= 1

        # only one marine
        for marin in state['marine_ships']:
            if num_of_marins > 1:
                deleted_marines.append(marin)
                del state1['marine_ships'][marin]
                num_of_marins -= 1

        # only one treasure

        for tres in state['treasures']:
            if num_of_tres > 1:
                del state1['treasures'][tres]
                num_of_tres -= 1

        # Initialize the cinitial tuple

        pirate_ships_state = {}
        for ship_name, ship_info in state1['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        s = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in state1['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        s += (marine_ships_state,)
        s += (state1['treasures'],)

        a = self.Q[(self.state_to_tuple(s), state1['turns to go'])]
        if (a == 'reset' or a == 'terminate'):
            return a

        # for ship in self.deleted_ships:
        # action = ('wait', str(ship))
        # a = a +(action,)
        temp = a
        for ship in self.deleted_ships:
            if (a[0][0] == 'deposit' or a[0][0] == 'wait'):
                action = (str(temp[0][0]), str(ship))
                a = a + (action,)
                continue
            action = (str(temp[0][0]), str(ship), temp[0][2])
            a = a + (action,)

        return a


class OptimalPirateAgent:
    def __init__(self, initial):
        self.initial = initial
        self.map = initial["map"]
        self.treasures = initial['treasures']
        self.V = 0
        self.Q = 0
        first_pirate_ship_info = next(iter(initial['pirate_ships'].values()))
        self.base = first_pirate_ship_info['location']
        # Initialize pirate ships with your new structure
        pirate_ships_state = {}
        for ship_name, ship_info in initial['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        # Initialize the initial_state tuple
        self.initial_state = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in initial['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        self.initial_state += (marine_ships_state,)
        self.initial_state += (initial['treasures'],)
        self.turns_to_go = initial["turns to go"]
        self.VI()

    def action_per_ship(self, ship, state):
        n = len(self.map) - 1
        m = len(self.map[0]) - 1
        ship_loc = state[0][ship]['location']
        ac_list = []
        if (ship_loc[0] + 1) <= n and self.map[ship_loc[0] + 1][ship_loc[1]] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0] + 1, ship_loc[1])))
        if (ship_loc[1] + 1) <= m and self.map[ship_loc[0]][ship_loc[1] + 1] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] + 1)))
        if (ship_loc[0] - 1) >= 0 and self.map[ship_loc[0] - 1][ship_loc[1]] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0] - 1, ship_loc[1])))
        if (ship_loc[1] - 1) >= 0 and self.map[ship_loc[0]][ship_loc[1] - 1] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] - 1)))

        if state[0][ship]['capacity'] > 0:
            for tres, tres_value in state[2].items():
                tres_pos = tres_value['location']
                if (ship_loc[0] + 1) <= n:
                    if (ship_loc[0] + 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] + 1) <= m:
                    if (ship_loc[0], ship_loc[1] + 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[0] - 1) >= 0:
                    if (ship_loc[0] - 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] - 1) >= 0:
                    if (ship_loc[0], ship_loc[1] - 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))

        if state[0][ship]['capacity'] < self.initial_state[0][ship]['capacity']:
            if ship_loc == self.base:
                ac_list.append(('deposit', ship))

        ac_list.append(('wait', ship))
        # ac_list.append('reset')
        # ac_list.append('terminate')
        return ac_list

    def yield_all_actions(self, ship_actions):
        ship_action_lists = list(ship_actions.values())
        for combination in itertools.product(*ship_action_lists):
            yield combination

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_ac = {}
        for ship in state[0]:
            dict_ac[ship] = self.action_per_ship(ship, state)

        for action_combination in self.yield_all_actions(dict_ac):
            yield action_combination
        yield ('terminate',)
        yield ('reset',)

    def action_per_marine(self, marine, state):
        loc = state[1][marine]['index']
        path_len = len(state[1][marine]['path'])
        poss_ac = []
        if path_len == 1:
            poss_ac.append({marine: (0, 1)})
        else:
            if loc == 0:
                poss_ac.append({marine: (0, 0.5)})
                poss_ac.append({marine: (1, 0.5)})
            elif loc == path_len - 1:
                poss_ac.append({marine: (loc, 0.5)})
                poss_ac.append({marine: (loc - 1, 0.5)})
            else:
                poss_ac.append({marine: (loc, 1 / 3)})
                poss_ac.append({marine: (loc - 1, 1 / 3)})
                poss_ac.append({marine: (loc + 1, 1 / 3)})
        return poss_ac

    def action_per_treasure(self, treasure, state):
        loc = state[2][treasure]['location']
        poss_loc = state[2][treasure]['possible_locations']
        len_path = len(poss_loc)
        prob_change = state[2][treasure]['prob_change_location']
        p = prob_change / len_path
        poss_ac = []
        for l in poss_loc:
            if l == loc:
                poss_ac.append({treasure: (l, (1 - prob_change) + p)})
            else:
                poss_ac.append({treasure: (l, p)})
        return poss_ac

    def yield_all_states(self, states):
        states_lists = list(states.values())
        for combination in itertools.product(*states_lists):
            yield combination

    def poss_states_marines_treasures(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_states = {}
        for marine in state[1]:
            dict_states[marine] = self.action_per_marine(marine, state)
        for treasure in state[2]:
            dict_states[treasure] = self.action_per_treasure(treasure, state)

        for action_combination in self.yield_all_actions(dict_states):
            yield action_combination

    def apply_action_sail(self, state, ac, reward):
        state[0][ac[1]]['location'] = ac[2]  # update pirate loc
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            if ac[2] == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_collect(self, state, ac, reward):
        state[0][ac[1]]['capacity'] -= 1
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_deposit(self, state, ac, reward):
        num_tre = self.initial_state[0][ac[1]]['capacity'] - state[0][ac[1]]['capacity']
        reward += 4 * num_tre
        state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
        return reward

    def apply_action_wait(self, state, ac, reward):
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def all_possible_next_states(self, state, action):

        RESET_PENALTY = 2
        reward = 0
        possible_next_states = []
        new_state = copy.deepcopy(state)
        for ac in action:
            if ac == "reset":
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac == 'terminate':
                self.turns_to_go = 0
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac[0] == 'sail':
                reward = self.apply_action_sail(new_state, ac, reward)
            if ac[0] == "collect":
                reward = self.apply_action_collect(new_state, ac, reward)
            if ac[0] == "deposit":
                reward = self.apply_action_deposit(new_state, ac, reward)
            if ac[0] == 'wait':
                reward = self.apply_action_wait(new_state, ac, reward)

        possible = self.poss_states_marines_treasures(new_state)
        list_possible = list(possible)  # list of possible combinations of marin treasure locations
        for index_p, p in enumerate(list_possible):
            i = 0
            prob_state = 1
            for marine in new_state[1]:
                new_state[1][marine]['index'] = p[i][marine][0]
                prob_state *= p[i][marine][1]
                i += 1
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = p[i][treasure][0]
                prob_state *= p[i][treasure][1]
                i += 1
            d = {'state': new_state, 'prob': prob_state, 'reward': reward}
            possible_next_states.append(d)
        return possible_next_states

    def state_to_tuple(self, state):
        # Assuming state is a tuple of three dictionaries: (pirate_ships_dict, marine_ships_dict, treasures_dict)
        pirate_ships, marine_ships, treasures = state

        # Convert pirate_ships dictionary to a tuple of tuples
        pirate_ships_tuple = tuple(
            (name,
             ('location', info['location']),
             ('capacity', info['capacity']),
             ) for name, info in pirate_ships.items()
        )

        # Convert marine_ships dictionary to a tuple of tuples
        marine_ships_tuple = tuple(
            (name,
             ('index', info['index']),
             ('path', tuple(info['path']))
             ) for name, info in marine_ships.items()
        )

        # Convert treasures dictionary to a tuple of tuples
        treasures_tuple = tuple(
            (name,
             ('location', info['location']),
             ('possible_locations', tuple(info['possible_locations'])),
             ('prob_change_location', info['prob_change_location'])
             ) for name, info in treasures.items()
        )

        # Combine all parts into a single state tuple
        return (pirate_ships_tuple, marine_ships_tuple, treasures_tuple)

    def tuple_to_state(self, state_tuple):
        pirate_ships_tuple, marine_ships_tuple, treasures_tuple = state_tuple

        pirate_ships_dict = {
            name: {
                'location': location,
                'capacity': capacity,
            } for name, (_, location), (_, capacity) in pirate_ships_tuple
        }

        marine_ships_dict = {
            name: {
                'index': index,
                'path': list(path)  # Convert back to mutable list
            } for name, (_, index), (_, path) in marine_ships_tuple
        }

        treasures_dict = {
            name: {
                'location': location,
                'possible_locations': (possible_locations),  # Convert back to mutable list
                'prob_change_location': prob_change_location
            } for name, (_, location), (_, possible_locations), (_, prob_change_location) in treasures_tuple
        }

        return (pirate_ships_dict, marine_ships_dict, treasures_dict)

    def generate_states(self, initial_state):
        # Get all possible 'S' locations for pirate ships
        pirate_ships = initial_state[0]
        marine_ships = initial_state[1]
        treasures = initial_state[2]
        sea_locations = [(i, j) for i, row in enumerate(self.map) for j, cell in enumerate(row) if
                         (cell == 'S' or cell == 'B')]
        # Generate all possible locations for each pirate ship
        pirate_ships_locations = {}
        for name, ship in pirate_ships.items():
            l = []
            for i in range(ship['capacity'] + 1):
                for loc in sea_locations:
                    l.append((loc, i))
            pirate_ships_locations[name] = l
        # Generate all possible treasure locations
        treasure_states = {name: treasure['possible_locations'] for name, treasure in treasures.items()}

        # # Generate all possible states for marine ships
        marine_ships_states = {}
        for name, ship in marine_ships.items():
            l = []
            for i in range(len(ship['path'])):
                l.append((i, ship['path'][i]))
            marine_ships_states[name] = l

        # Generate all combinations
        all_states = []
        for pirate_ship_combinations in itertools.product(*pirate_ships_locations.values()):
            for treasure_combinations in itertools.product(*treasure_states.values()):
                for marine_ship_combinations in itertools.product(*marine_ships_states.values()):
                    # Combine current state
                    state = {
                        'pirate_ships': dict(zip(pirate_ships.keys(), pirate_ship_combinations)),
                        'treasures': dict(zip(treasures.keys(), treasure_combinations)),
                        'marine_ships': dict(zip(marine_ships.keys(), marine_ship_combinations)),
                    }
                    all_states.append(state)

        return all_states

    def generate_all_states(self, initial_state):
        states = self.generate_states(initial_state)
        all_states = []
        for s in states:
            new_state = copy.deepcopy(initial_state)
            for ship in new_state[0]:
                new_state[0][ship]['location'] = s['pirate_ships'][ship][0]
                new_state[0][ship]['capacity'] = s['pirate_ships'][ship][1]
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = s['treasures'][treasure]
            for marine in new_state[1]:
                new_state[1][marine]['index'] = s['marine_ships'][marine][0]
            all_states.append(new_state)
        return all_states

    def VI(self):
        S = self.generate_all_states(self.initial_state)  # list of all states as dictionaries
        s_a_prob = {}
        max_iter = self.turns_to_go  # Maximum number of iterations
        self.V = {self.state_to_tuple(s): 0 for s in S}
        self.Q = {(self.state_to_tuple(s), i): 0 for s in S for i in
                  range(max_iter + 1)}  # key=(state,turns to go) ,value=(value of VI,best action)

        for i in range(max_iter + 1):
            newV = {self.state_to_tuple(s): 0 for s in S}
            for s in S:
                s_dict = s
                max_val = float('-inf')
                A = self.actions(s_dict)  # all actions that can be performed is state
                best_a = 0
                for j, a in enumerate(A):
                    s_a_prob[(self.state_to_tuple(s), a)] = self.all_possible_next_states(s, a)
                    val = s_a_prob[(self.state_to_tuple(s), a)][0]['reward']

                    for state_prob_r in s_a_prob[(self.state_to_tuple(s), a)]:
                        val += state_prob_r['prob'] * self.V[self.state_to_tuple(state_prob_r['state'])]

                    if max_val < val:
                        max_val = val
                        best_a = a
                newV[self.state_to_tuple(s)] = max_val
                self.Q[(self.state_to_tuple(s), i)] = best_a
            self.V = newV

    def act(self, state):
        # Initialize pirate ships with your new structure
        pirate_ships_state = {}
        for ship_name, ship_info in state['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }
        # Initialize the initial_state tuple
        s = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in state['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        s += (marine_ships_state,)
        s += (state['treasures'],)

        a = self.Q[(self.state_to_tuple(s), state['turns to go'])]
        return a
class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.map = initial["map"]
        self.treasures = initial['treasures']
        self.V = {}
        self.policy = {}
        self.gamma = gamma
        first_pirate_ship_info = next(iter(initial['pirate_ships'].values()))
        self.base = first_pirate_ship_info['location']
        # Initialize pirate ships with your new structure
        pirate_ships_state = {}
        for ship_name, ship_info in initial['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        # Initialize the initial_state tuple
        self.initial_state = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in initial['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        self.initial_state += (marine_ships_state,)
        self.initial_state += (initial['treasures'],)
        self.value_iteration(0.01)

    def action_per_ship(self, ship, state):
        n = len(self.map) - 1
        m = len(self.map[0]) - 1
        ship_loc = state[0][ship]['location']
        ac_list = []
        if (ship_loc[0] + 1) <= n and self.map[ship_loc[0] + 1][ship_loc[1]] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0] + 1, ship_loc[1])))
        if (ship_loc[1] + 1) <= m and self.map[ship_loc[0]][ship_loc[1] + 1] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] + 1)))
        if (ship_loc[0] - 1) >= 0 and self.map[ship_loc[0] - 1][ship_loc[1]] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0] - 1, ship_loc[1])))
        if (ship_loc[1] - 1) >= 0 and self.map[ship_loc[0]][ship_loc[1] - 1] != 'I':
            ac_list.append(('sail', ship, (ship_loc[0], ship_loc[1] - 1)))

        if state[0][ship]['capacity'] > 0:
            for tres, tres_value in state[2].items():
                tres_pos = tres_value['location']
                if (ship_loc[0] + 1) <= n:
                    if (ship_loc[0] + 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] + 1) <= m:
                    if (ship_loc[0], ship_loc[1] + 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[0] - 1) >= 0:
                    if (ship_loc[0] - 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect', ship, tres))
                if (ship_loc[1] - 1) >= 0:
                    if (ship_loc[0], ship_loc[1] - 1) == tres_pos:
                        ac_list.append(('collect', ship, tres))

        if state[0][ship]['capacity'] < self.initial_state[0][ship]['capacity']:
            if ship_loc == self.base:
                ac_list.append(('deposit', ship))

        ac_list.append(('wait', ship))
        # ac_list.append('reset')
        # ac_list.append('terminate')
        return ac_list

    def yield_all_actions(self, ship_actions):
        ship_action_lists = list(ship_actions.values())
        for combination in itertools.product(*ship_action_lists):
            yield combination

    def actions(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_ac = {}
        for ship in state[0]:
            dict_ac[ship] = self.action_per_ship(ship, state)

        for action_combination in self.yield_all_actions(dict_ac):
            yield action_combination
        yield ('terminate',)
        yield ('reset',)

    def action_per_marine(self, marine, state):
        loc = state[1][marine]['index']
        path_len = len(state[1][marine]['path'])
        poss_ac = []
        if path_len == 1:
            poss_ac.append({marine: (0, 1)})
        else:
            if loc == 0:
                poss_ac.append({marine: (0, 0.5)})
                poss_ac.append({marine: (1, 0.5)})
            elif loc == path_len - 1:
                poss_ac.append({marine: (loc, 0.5)})
                poss_ac.append({marine: (loc - 1, 0.5)})
            else:
                poss_ac.append({marine: (loc, 1 / 3)})
                poss_ac.append({marine: (loc - 1, 1 / 3)})
                poss_ac.append({marine: (loc + 1, 1 / 3)})
        return poss_ac

    def action_per_treasure(self, treasure, state):
        loc = state[2][treasure]['location']
        poss_loc = state[2][treasure]['possible_locations']
        len_path = len(poss_loc)
        prob_change = state[2][treasure]['prob_change_location']
        p = prob_change / len_path
        poss_ac = []
        for l in poss_loc:
            if l == loc:
                poss_ac.append({treasure: (l, (1 - prob_change) + p)})
            else:
                poss_ac.append({treasure: (l, p)})
        return poss_ac

    def yield_all_states(self, states):
        states_lists = list(states.values())
        for combination in itertools.product(*states_lists):
            yield combination

    def poss_states_marines_treasures(self, state):
        """Returns all the actions that can be executed in the given
        state. The result should be a tuple (or other iterable) of actions
        as defined in the problem description file"""
        dict_states = {}
        for marine in state[1]:
            dict_states[marine] = self.action_per_marine(marine, state)
        for treasure in state[2]:
            dict_states[treasure] = self.action_per_treasure(treasure, state)

        for action_combination in self.yield_all_actions(dict_states):
            yield action_combination

    def apply_action_sail(self, state, ac, reward):
        state[0][ac[1]]['location'] = ac[2]  # update pirate loc
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            if ac[2] == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_collect(self, state, ac, reward):
        state[0][ac[1]]['capacity'] -= 1
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def apply_action_deposit(self, state, ac, reward):
        num_tre = self.initial_state[0][ac[1]]['capacity'] - state[0][ac[1]]['capacity']
        reward += 4 * num_tre
        state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
        return reward

    def apply_action_wait(self, state, ac, reward):
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['location']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
        return reward

    def all_possible_next_states(self, state, action):

        RESET_PENALTY = 2
        reward = 0
        possible_next_states = []
        new_state = copy.deepcopy(state)
        for ac in action:
            if ac == "reset":
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac == 'terminate':
                self.turns_to_go = 0
                return [{'state': self.initial_state, 'prob': 1, 'reward': -RESET_PENALTY}]
            if ac[0] == 'sail':
                reward = self.apply_action_sail(new_state, ac, reward)
            if ac[0] == "collect":
                reward = self.apply_action_collect(new_state, ac, reward)
            if ac[0] == "deposit":
                reward = self.apply_action_deposit(new_state, ac, reward)
            if ac[0] == 'wait':
                reward = self.apply_action_wait(new_state, ac, reward)

        possible = self.poss_states_marines_treasures(new_state)
        list_possible = list(possible)  # list of possible combinations of marin treasure locations
        for index_p, p in enumerate(list_possible):
            i = 0
            poss_state = copy.deepcopy(new_state)
            prob_state = 1
            for marine in poss_state[1]:
                poss_state[1][marine]['index'] = p[i][marine][0]
                prob_state *= p[i][marine][1]
                i += 1
            for treasure in poss_state[2]:
                poss_state[2][treasure]['location'] = p[i][treasure][0]
                prob_state *= p[i][treasure][1]
                i += 1

            d = {'state': poss_state, 'prob': prob_state, 'reward': reward}
            possible_next_states.append(d)
        return possible_next_states

    def state_to_tuple(self, state):
        # Assuming state is a tuple of three dictionaries: (pirate_ships_dict, marine_ships_dict, treasures_dict)
        pirate_ships, marine_ships, treasures = state

        # Convert pirate_ships dictionary to a tuple of tuples
        pirate_ships_tuple = tuple(
            (name,
             ('location', info['location']),
             ('capacity', info['capacity']),
             ) for name, info in pirate_ships.items()
        )

        # Convert marine_ships dictionary to a tuple of tuples
        marine_ships_tuple = tuple(
            (name,
             ('index', info['index']),
             ('path', tuple(info['path']))
             ) for name, info in marine_ships.items()
        )

        # Convert treasures dictionary to a tuple of tuples
        treasures_tuple = tuple(
            (name,
             ('location', info['location']),
             ('possible_locations', tuple(info['possible_locations'])),
             ('prob_change_location', info['prob_change_location'])
             ) for name, info in treasures.items()
        )

        # Combine all parts into a single state tuple
        return (pirate_ships_tuple, marine_ships_tuple, treasures_tuple)

    def tuple_to_state(self, state_tuple):
        pirate_ships_tuple, marine_ships_tuple, treasures_tuple = state_tuple

        pirate_ships_dict = {
            name: {
                'location': location,
                'capacity': capacity,
            } for name, (_, location), (_, capacity) in pirate_ships_tuple
        }

        marine_ships_dict = {
            name: {
                'index': index,
                'path': list(path)  # Convert back to mutable list
            } for name, (_, index), (_, path) in marine_ships_tuple
        }

        treasures_dict = {
            name: {
                'location': location,
                'possible_locations': (possible_locations),  # Convert back to mutable list
                'prob_change_location': prob_change_location
            } for name, (_, location), (_, possible_locations), (_, prob_change_location) in treasures_tuple
        }

        return (pirate_ships_dict, marine_ships_dict, treasures_dict)

    def generate_states(self, initial_state):
        # Get all possible 'S' locations for pirate ships
        pirate_ships = initial_state[0]
        marine_ships = initial_state[1]
        treasures = initial_state[2]
        sea_locations = [(i, j) for i, row in enumerate(self.map) for j, cell in enumerate(row) if
                         (cell == 'S' or cell == 'B')]
        # Generate all possible locations for each pirate ship
        pirate_ships_locations = {}
        for name, ship in pirate_ships.items():
            l = []
            for i in range(ship['capacity'] + 1):
                for loc in sea_locations:
                    l.append((loc, i))
            pirate_ships_locations[name] = l
        # Generate all possible treasure locations
        treasure_states = {name: treasure['possible_locations'] for name, treasure in treasures.items()}

        # # Generate all possible states for marine ships
        marine_ships_states = {}
        for name, ship in marine_ships.items():
            l = []
            for i in range(len(ship['path'])):
                l.append((i, ship['path'][i]))
            marine_ships_states[name] = l

        # Generate all combinations
        all_states = []
        for pirate_ship_combinations in itertools.product(*pirate_ships_locations.values()):
            for treasure_combinations in itertools.product(*treasure_states.values()):
                for marine_ship_combinations in itertools.product(*marine_ships_states.values()):
                    # Combine current state
                    state = {
                        'pirate_ships': dict(zip(pirate_ships.keys(), pirate_ship_combinations)),
                        'treasures': dict(zip(treasures.keys(), treasure_combinations)),
                        'marine_ships': dict(zip(marine_ships.keys(), marine_ship_combinations)),
                    }
                    all_states.append(state)

        return all_states

    def generate_all_states(self, initial_state):
        states = self.generate_states(initial_state)
        all_states = []
        for s in states:
            new_state = copy.deepcopy(initial_state)
            for ship in new_state[0]:
                new_state[0][ship]['location'] = s['pirate_ships'][ship][0]
                new_state[0][ship]['capacity'] = s['pirate_ships'][ship][1]
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = s['treasures'][treasure]
            for marine in new_state[1]:
                new_state[1][marine]['index'] = s['marine_ships'][marine][0]
            all_states.append(new_state)
        return all_states

    def value_iteration(self, epsilon=0.01):
        all_states = self.generate_all_states(self.initial_state)
        # Initialize all V(s) to 0 or some other small values for all states in S
        self.V = {self.state_to_tuple(state): 0 for state in all_states}
        while True:
            delta = 0  # This will track the maximum change in value for any state in an iteration
            # Loop over all states
            for state in all_states:
                v = self.V[self.state_to_tuple(state)]  # Store the current value of the state
                # Bellman update, set V(s) to the max_a of expected value of doing a in state s
                max_val = float('-inf')  # Start with minus infinity to ensure any real value is larger
                for action in self.actions(state):
                    expected_val = 0
                    # Calculate the expected value of this action
                    for prob_state in self.all_possible_next_states(state, action):
                        s_prime, prob, reward = prob_state['state'], prob_state['prob'], prob_state['reward']
                        expected_val += prob * (reward + self.gamma * self.V[self.state_to_tuple(s_prime)])
                    # Update max_val if this action is better than the previous best
                    if expected_val > max_val:
                        max_val = expected_val
                # Calculate the maximum difference for this state and update the state-value function
                delta = max(delta, abs(v - max_val))
                self.V[self.state_to_tuple(state)] = max_val  # Update the value of the state to the max_val found

            # Check for convergence, break the while loop if values converged (change is smaller than epsilon)
            if delta < epsilon:
                break  # Exit the loop if the value function change is below the threshold

    def act(self, state):
        # Initialize variables for best action and highest value found so far
        best_action = None
        best_value = float('-inf')

        pirate_ships_state = {}
        for ship_name, ship_info in state['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        # Initialize the initial_state tuple
        s = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in state['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        s += (marine_ships_state,)
        s += (state['treasures'],)

        # Iterate through all possible actions to find the best one according to the infinite-horizon value function
        for action in self.actions(s):
            expected_value = 0
            for outcome in self.all_possible_next_states(s, action):
                next_state, probability, reward = outcome['state'], outcome['prob'], outcome['reward']
                next_state_tuple = self.state_to_tuple(next_state)
                expected_value += probability * (reward + self.gamma * self.V[next_state_tuple])

            # Update the best action if this action is better than the current best
            if expected_value > best_value:
                best_value = expected_value
                best_action = action

        return best_action

    def value(self, state):
        # Returns the value of the state
        # Ensure you've run value_iteration at least once before calling this
        pirate_ships_state = {}
        for ship_name, ship_info in state['pirate_ships'].items():
            # Here, we create a new structure for each pirate ship
            pirate_ships_state[ship_name] = {
                'location': ship_info['location'],  # 'position' now directly holds the location
                'capacity': ship_info['capacity'],
            }

        # Initialize the initial_state tuple
        s = (pirate_ships_state,)
        marine_ships_state = {}
        for ship_name, ship_info in state['marine_ships'].items():
            marine_ships_state[ship_name] = ship_info
        s += (marine_ships_state,)
        s += (state['treasures'],)
        return self.V[self.state_to_tuple(s)]
