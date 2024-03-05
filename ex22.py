import itertools
import copy

ids = ["209317239", "208165969"]


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
                'position': ship_info['location'],  # 'position' now directly holds the location
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
        self.points_game = 0
        self.VI()

    def action_per_ship(self, ship, state):
        n = len(self.map) - 1
        m = len(self.map[0]) - 1
        ship_loc = state[0][ship]['position']
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
                        ac_list.append(('collect_treasure', ship, tres))
                if (ship_loc[1] + 1) <= m:
                    if (ship_loc[0], ship_loc[1] + 1) == tres_pos:
                        ac_list.append(('collect_treasure', ship, tres))
                if (ship_loc[0] - 1) >= 0:
                    if (ship_loc[0] - 1, ship_loc[1]) == tres_pos:
                        ac_list.append(('collect_treasure', ship, tres))
                if (ship_loc[1] - 1) >= 0:
                    if (ship_loc[0], ship_loc[1] - 1) == tres_pos:
                        ac_list.append(('collect_treasure', ship, tres))

        if state[0][ship]['capacity'] < self.initial_state[0][ship]['capacity']:
            if ship_loc == self.base:
                ac_list.append(('deposit_treasures', ship))

        ac_list.append(('wait', ship))
        ac_list.append('reset')
        ac_list.append('terminate')
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

    def action_per_marine(self, marine, state):
        loc = state[1][marine]['index']
        path_len = len(state[1][marine]['path'])
        poss_ac = []
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
        state[0][ac[1]]['position'] = ac[2]  # update pirate loc
        for marine in state[1]:
            index = state[1][marine]['index']
            print(index)
            print(state[1][marine]['path'])
            loc = state[1][marine]['path'][index]
            if ac[2] == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
                return reward
        return reward

    def apply_action_collect(self, state, ac, reward):
        state[0][ac[1]]['capacity'] -= 1
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['position']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
                return reward
        return reward

    def apply_action_deposit(self, state, ac, reward):
        num_tre = len(state[0][ac[1]]['treasure_set'])
        reward += 4 * num_tre
        state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
        return reward

    def apply_action_wait(self, state, ac, reward):
        for marine in state[1]:
            index = state[1][marine]['index']
            loc = state[1][marine]['path'][index]
            loc_pirate = state[0][ac[1]]['position']
            if loc_pirate == loc:
                state[0][ac[1]]['capacity'] = self.initial_state[0][ac[1]]['capacity']
                reward -= 1
                return reward
        return reward

    def all_possible_next_states(self, state, action):

        RESET_PENALTY = 2

        possible_next_states = []
        possible = self.poss_states_marines_treasures(state)
        list_possible = list(possible)  # list of possible combinations of marin treasure locations
        for index_p, p in enumerate(list_possible):
            i = 0
            new_state = copy.deepcopy(state)
            prob_state = 1
            reward = 0
            for marine in new_state[1]:
                new_state[1][marine]['index'] = p[i][marine][0]
                prob_state *= p[i][marine][1]
                i += 1
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = p[i][treasure][0]
                prob_state *= p[i][treasure][1]
                i += 1

            for ac in action:
                if ac == "reset":
                    state = self.initial_state
                    return [{'state': state, 'prob': 1, 'reward': -RESET_PENALTY}]
                if ac == 'terminate':
                    self.turns_to_go = 0
                    return [{'state': state, 'prob': 1, 'reward': -RESET_PENALTY}]

                if ac[0] == 'sail':
                    reward = self.apply_action_sail(new_state, ac, reward)
                if ac[0] == "collect_treasure":
                    reward = self.apply_action_collect(new_state, ac, reward)
                if ac[0] == "deposit_treasures":
                    reward = self.apply_action_deposit(new_state, ac, reward)
                if ac[0] == 'wait':
                    reward = self.apply_action_wait(new_state, ac, reward)

            d = {'state': new_state, 'prob': prob_state, 'reward': reward}
            possible_next_states.append(d)
        return possible_next_states

    def generate_states(self, initial_state):
        # Extract initial states for different entities
        pirate_ships, marine_ships, treasures = initial_state

        # Get all possible 'S' and 'B' locations for pirate ships
        sea_locations = [(i, j) for i, row in enumerate(self.map) for j, cell in enumerate(row) if cell in ['S', 'B']]

        # Generate all possible locations and capacities for each pirate ship
        pirate_ships_locations = {
            name: [(loc, capacity) for loc in sea_locations for capacity in
                   range(ship['capacity'] + 1)]
            for name, ship in pirate_ships.items()
        }

        # Generate all possible states for treasures
        treasure_states = {name: treasure['possible_locations'] for name, treasure in treasures.items()}

        # Generate all possible states for marine ships
        marine_ships_states = {(name, idx) for name, ship in marine_ships.items() for idx in range(len(ship['path']))}

        # Generate all combinations of states
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

    def all_states(self, initial_state):
        states = self.generate_states(initial_state)
        all_states = []
        for s in states:
            new_state = copy.deepcopy(initial_state)
            for ship in new_state[0]:
                new_state[0][ship]['position'] = s['pirate_ships'][ship][0]
                new_state[0][ship]['capacity'] = s['pirate_ships'][ship][1]
            for treasure in new_state[2]:
                new_state[2][treasure]['location'] = s['treasures'][treasure]
            for marine in new_state[1]:
                new_state[1][marine]['index'] = s['marine_ships'][marine][0]

            all_states.append(new_state)
        return all_states

    # def all_states(self, initial_state):
    #     states = self.generate_states(initial_state)
    #     all_states = []
    #     for s in states:
    #         new_state = copy.deepcopy(initial_state)
    #         for ship in new_state[0]:
    #             new_state[0][ship]['position'] = s['pirate_ships'][ship]['position']
    #             new_state[0][ship]['capacity'] = s['pirate_ships'][ship]['capacity']
    #             new_state[0][ship]['treasure_set'] = s['pirate_ships'][ship]['treasure_set']
    #
    #         for treasure in new_state[2]:
    #             new_state[2][treasure]['location'] = s['treasures'][treasure]
    #         for marine in new_state[1]:
    #             new_state[1][marine]['index'] = s['marine_ships'][marine][0]
    #         all_states.append(new_state)
    #     return all_states

    def state_to_tuple(self, state):
        # Assuming state is a tuple of three dictionaries: (pirate_ships_dict, marine_ships_dict, treasures_dict)
        pirate_ships, marine_ships, treasures = state

        # Convert pirate_ships dictionary to a tuple of tuples
        pirate_ships_tuple = tuple(
            (name,
             ('position', info['position']),
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
                'position': position,
                'capacity': capacity,
            } for name, (_, position), (_, capacity) in pirate_ships_tuple
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

    def VI(self):
        S = self.all_states(self.initial_state)  # list of all states as dictionaries
        len(S)
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
                for j, a in enumerate(A):
                    s_a_prob[(self.state_to_tuple(s), a)] = self.all_possible_next_states(s, a)  # this function returns
                    # list of dictionaries that have the form:
                    # {'state': the next state(one of them),
                    #  'prob': probability of getting 'state' value,
                    #  'reward':score(reward from s_dict and a or 'state reward')}
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
        if state == self.initial:
            # Initialize pirate ships with your new structure
            pirate_ships_state = {}
            for ship_name, ship_info in state['pirate_ships'].items():
                # Here, we create a new structure for each pirate ship
                pirate_ships_state[ship_name] = {
                    'position': ship_info['location'],  # 'position' now directly holds the location
                    'capacity': ship_info['capacity'],
                    'treasure_set': set(),
                    # 'points': 0  # Assuming initial points are 0
                }
            # Initialize the initial_state tuple
            s = (pirate_ships_state,)
            marine_ships_state = {}
            for ship_name, ship_info in state['marine_ships'].items():
                marine_ships_state[ship_name] = ship_info
            s += (marine_ships_state,)
            s += (state['treasures'],)
        else:
            s = state
        a = self.Q[(self.state_to_tuple(s), state['turns to go'])]
        # print(self.V[self.create_tuple_state(self.initial)])
        return a


class PirateAgent:
    def __init__(self, initial):
        self.initial = initial

    def act(self, state):
        raise NotImplemented


class InfinitePirateAgent:
    def __init__(self, initial, gamma):
        self.initial = initial
        self.gamma = gamma

    def act(self, state):
        raise NotImplemented

    def value(self, state):
        raise NotImplemented
