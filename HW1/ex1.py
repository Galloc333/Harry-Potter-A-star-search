import search
import random
import math
import itertools

ids = ["318931672", "208302661"]


class HarryPotterProblem(search.Problem):

    def __init__(self, initial_config):
        self.board_map = initial_config["map"]
        self.wizard_data = initial_config["wizards"]
        self.horcrux_positions = initial_config["horcruxes"]
        self.voldemort_position = self.find_voldemort_position()

        death_eaters = initial_config["death_eaters"]
        death_eaters_state = {
            name: {'path': path, 'index': 0, 'direction': "forward"}
            for name, path in death_eaters.items()
        }

        initial_state = (
            tuple(tuple(row) for row in self.board_map),  # Immutable board map
            tuple((wizard_name, tuple(wiz_pos), lives) for wizard_name, (wiz_pos, lives) in self.wizard_data.items()),
            tuple(self.horcrux_positions),  # Immutable horcrux positions
            self.voldemort_position,
            tuple(
                (name, tuple(de['path']), de['index'], de['direction'])
                for name, de in death_eaters_state.items()
            )
        )

        super().__init__(initial_state)

    def find_voldemort_position(self):
        for row_index, row in enumerate(self.board_map):
            for col_index, cell in enumerate(row):
                if cell == "V":
                    return (row_index, col_index)
        return None

    def actions(self, state):
        board_state, wizard_states, horcruxes, voldemort_position, death_eaters_state = state

        updated_death_eaters = self.update_death_eaters_positions(death_eaters_state)
        death_eater_next_positions = {
            name: de['path'][de['index']] for name, de in updated_death_eaters.items()
        }
        next_horcrux = next((h for h in horcruxes if h != (-1, -1)), None)

        wizard_actions = self.generate_actions_for_all_wizards(
            wizard_states, horcruxes, voldemort_position,
            death_eater_next_positions, next_horcrux
        )
        if all(h == (-1, -1) for h in horcruxes):
            for wizard_name, wizard_position, _ in wizard_states:
                if wizard_name == "Harry Potter" and wizard_position == voldemort_position:
                    wizard_actions["Harry Potter"].append(("kill", "Harry Potter"))

        all_action_combinations = list(itertools.product(*wizard_actions.values()))
        valid_action_combinations = self.filter_duplicate_destroy_actions(all_action_combinations)

        return valid_action_combinations

    def generate_actions_for_all_wizards(
            self,
            wizard_states,
            horcruxes,
            voldemort_position,
            death_eater_next_positions,
            next_horcrux
    ):
        wizard_actions_map = {}

        for wizard_name, (row, col), lives in wizard_states:
            if lives <= 0:
                continue

            actions_for_this_wizard = []

            # Determine destroy actions
            if next_horcrux and (row, col) == next_horcrux:
                idx = horcruxes.index(next_horcrux)
                actions_for_this_wizard.append(("destroy", wizard_name, idx))
            else:
                for h in horcruxes:
                    if h != (-1, -1) and (row, col) == h:
                        idx = horcruxes.index(h)
                        actions_for_this_wizard.append(("destroy", wizard_name, idx))

            # Determine move actions
            possible_moves = False
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            for d_row, d_col in directions:
                nr, nc = row + d_row, col + d_col

                if lives == 1 and (nr, nc) in death_eater_next_positions.values():
                    continue

                if ((nr, nc) == voldemort_position and
                        all(h == (-1, -1) for h in horcruxes) and wizard_name == "Harry Potter"):
                    actions_for_this_wizard.append(("move", wizard_name, (nr, nc)))
                    possible_moves = True
                elif 0 <= nr < len(self.board_map) and 0 <= nc < len(self.board_map[0]):
                    if self.board_map[nr][nc] == "P":
                        actions_for_this_wizard.append(("move", wizard_name, (nr, nc)))
                        possible_moves = True

            # If no destroy and no move possible, wizard waits
            if not any(a[0] == "destroy" for a in actions_for_this_wizard) and not possible_moves:
                actions_for_this_wizard.append(("wait", wizard_name))

            wizard_actions_map[wizard_name] = actions_for_this_wizard

        return wizard_actions_map

    def filter_duplicate_destroy_actions(self, action_combinations):
        valid_combos = []
        for combo in action_combinations:
            destroyed_indices = [act[2] for act in combo if act[0] == "destroy"]
            if len(destroyed_indices) == len(set(destroyed_indices)):
                valid_combos.append(combo)
        return valid_combos

    def result(self, state, actions):
        map_state, wizard_states, horcruxes, voldemort_position, death_eaters_state = state

        updated_wizard_states = [list(wiz) for wiz in wizard_states]
        updated_horcruxes = list(horcruxes)

        updated_wizard_states = self.apply_wizard_actions(actions, updated_wizard_states, updated_horcruxes)
        voldemort_position = self.apply_kill_action(actions, voldemort_position)

        death_eaters_data = {
            name: {'path': list(path), 'index': idx, 'direction': direction}
            for name, path, idx, direction in death_eaters_state
        }
        updated_death_eaters = self.update_death_eaters(death_eaters_data)

        self.update_wizard_lives_after_deatheaters(updated_wizard_states, updated_death_eaters)

        new_wizards_tuple = tuple(tuple(w) for w in updated_wizard_states)
        new_horcruxes_tuple = tuple(updated_horcruxes)
        new_death_eaters_tuple = tuple(
            (name, tuple(de_state['path']), de_state['index'], de_state['direction'])
            for name, de_state in updated_death_eaters.items()
        )

        new_state = (
            map_state,
            new_wizards_tuple,
            new_horcruxes_tuple,
            voldemort_position,
            new_death_eaters_tuple
        )
        return new_state

    def apply_wizard_actions(self, actions, wizard_states, horcruxes):
        for action in actions:
            action_type, wizard_name, *args = action

            if action_type == "move":
                new_pos = args[0]
                # Update the wizard position
                for i, (existing_name, _, existing_lives) in enumerate(wizard_states):
                    if existing_name == wizard_name:
                        wizard_states[i] = (existing_name, new_pos, existing_lives)

            elif action_type == "destroy":
                horcrux_idx = args[0]
                if 0 <= horcrux_idx < len(horcruxes):
                    horcruxes[horcrux_idx] = (-1, -1)

        return wizard_states

    def apply_kill_action(self, actions, voldemort_position):
        for action in actions:
            action_type = action[0]
            if action_type == "kill":
                voldemort_position = (-1, -1)
        return voldemort_position

    def update_wizard_lives_after_deatheaters(self, wizard_states, updated_death_eaters):
        for i, (wizard_name, wiz_pos, lives) in enumerate(wizard_states):
            for death_eater in updated_death_eaters.values():
                death_eater_pos = death_eater['path'][death_eater['index']]
                if wiz_pos == death_eater_pos and lives > 0:
                    lives -= 1
                    wizard_states[i] = (wizard_name, wiz_pos, lives)

    def goal_test(self, state):
        _, wizard_states, horcruxes, voldemort_position, _ = state

        if any(h != (-1, -1) for h in horcruxes):
            return False

        for wizard_name, _, _ in wizard_states:
            if wizard_name == "Harry Potter" and voldemort_position == (-1, -1):
                return True
        return False

    def h(self, node):
        """
        Combine our two heuristics with chosen weighting factors.
        """
        assignment_score = self.horcrux_assignment_heuristic(node)
        proximity_score = self.horcrux_proximity_heuristic(node)
        return 0.43 * assignment_score + 0.37 * proximity_score

    def horcrux_assignment_heuristic(self, node):
        """
        Minimal assignment cost + penalty + distance to Voldemort if no horcruxes left.
        """
        state = node.state
        _, wizards, horcruxes, voldemort_position, _ = state

        # If Voldemort is at (-1, -1) he's dead
        if voldemort_position == (-1, -1):
            return 0.0

        for _, _, lives in wizards:
            if lives <= 0:
                return float('inf')

        active_horcruxes = [h for h in horcruxes if h != (-1, -1)]
        num_horcruxes = len(active_horcruxes)

        harry_position = None
        alive_wizards = []
        for wizard_name, wiz_pos, lives in wizards:
            if lives > 0:
                alive_wizards.append((wizard_name, wiz_pos, lives))
            if wizard_name == "Harry Potter":
                harry_position = wiz_pos

        if not harry_position:
            return float('inf')

        if num_horcruxes == 0:
            distance_to_voldemort = self.manhattan_distance(harry_position, voldemort_position)
            return float(distance_to_voldemort)

        cost_matrix = []
        for _, wizard_position, _ in alive_wizards:
            row = []
            for hrcx_pos in active_horcruxes:
                dist = self.manhattan_distance(wizard_position, hrcx_pos)
                row.append(dist)
            cost_matrix.append(row)

        n = max(len(alive_wizards), num_horcruxes)
        for row in cost_matrix:
            while len(row) < n:
                row.append(0)
        while len(cost_matrix) < n:
            cost_matrix.append([0] * n)

        min_assignment_cost = self.hungarian_algorithm(cost_matrix)
        uncovered_horcruxes = max(0, num_horcruxes - len(alive_wizards))
        distance_to_voldemort = self.manhattan_distance(harry_position, voldemort_position)

        heuristic_value = min_assignment_cost + uncovered_horcruxes + distance_to_voldemort
        return heuristic_value

    def horcrux_proximity_heuristic(self, node):
        """
        Min distance wizard->horcrux + horcrux count + distance Harry->Voldemort.
        """
        state = node.state
        _, wizards, horcruxes, voldemort_position, _ = state

        if voldemort_position == (-1, -1):
            return 0.0

        for _, _, lives in wizards:
            if lives <= 0:
                return float('inf')

        remaining_horcruxes = sum(1 for h in horcruxes if h != (-1, -1))

        harry_position = None
        for wizard_name, wiz_pos, _ in wizards:
            if wizard_name == "Harry Potter":
                harry_position = wiz_pos
                break

        if not harry_position:
            return float('inf')

        horcrux_to_wizard_distances = []
        for horcrux_pos in horcruxes:
            if horcrux_pos != (-1, -1):
                min_distance = float('inf')
                for _, wizard_position, _ in wizards:
                    dist = self.manhattan_distance(wizard_position, horcrux_pos)
                    if dist < min_distance:
                        min_distance = dist
                horcrux_to_wizard_distances.append(min_distance)

        closest_horcrux_distance = min(horcrux_to_wizard_distances) if horcrux_to_wizard_distances else 0
        distance_to_voldemort = self.manhattan_distance(harry_position, voldemort_position)
        heuristic1 = closest_horcrux_distance + remaining_horcruxes + distance_to_voldemort

        heuristic_horcrux_sum = sum(d + 1 for d in horcrux_to_wizard_distances)
        heuristic2 = heuristic_horcrux_sum + distance_to_voldemort + 1

        combined_heuristic = (heuristic1 + heuristic2) / 2.5
        return combined_heuristic

    @staticmethod
    def update_death_eaters(death_eaters_data):
        def direction_to_step(direction_str):
            return 1 if direction_str == "forward" else -1

        updated_death_eaters = {}
        for name, de_state in death_eaters_data.items():
            path = de_state['path']
            index = de_state['index']
            direction = de_state['direction']

            if len(path) == 1:
                updated_death_eaters[name] = {
                    'path': path,
                    'index': index,
                    'direction': direction
                }
                continue

            step = direction_to_step(direction)
            new_index = index + step
            if new_index >= len(path):
                new_index = len(path) - 2
                direction = "backward"
            elif new_index < 0:
                new_index = 1
                direction = "forward"

            updated_death_eaters[name] = {
                'path': path,
                'index': new_index,
                'direction': direction
            }
        return updated_death_eaters

    def update_death_eaters_positions(self, death_eaters_state):
        death_eaters_data = {
            name: {'path': list(path), 'index': idx, 'direction': direction}
            for name, path, idx, direction in death_eaters_state
        }
        return self.update_death_eaters(death_eaters_data)

    @staticmethod
    def manhattan_distance(pos1, pos2):
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])

    @staticmethod
    def hungarian_algorithm(cost_matrix):
        n = len(cost_matrix)
        u = [0] * (n + 1)
        v = [0] * (n + 1)
        p = [0] * (n + 1)
        way = [0] * (n + 1)

        for i in range(1, n + 1):
            p[0] = i
            j0 = 0
            minv = [math.inf] * (n + 1)
            used = [False] * (n + 1)
            while True:
                used[j0] = True
                i0 = p[j0]
                j1 = 0
                delta = math.inf
                for j in range(1, n + 1):
                    if not used[j]:
                        cur = cost_matrix[i0 - 1][j - 1] - u[i0] - v[j]
                        if cur < minv[j]:
                            minv[j] = cur
                            way[j] = j0
                        if minv[j] < delta:
                            delta = minv[j]
                            j1 = j
                for jj in range(n + 1):
                    if used[jj]:
                        u[p[jj]] += delta
                        v[jj] -= delta
                    else:
                        minv[jj] -= delta
                j0 = j1
                if p[j0] == 0:
                    break
            while True:
                j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
                if j0 == 0:
                    break

        assignment = p[1:]
        total_cost = 0
        for j in range(1, n + 1):
            i = assignment[j - 1]
            total_cost += cost_matrix[i - 1][j - 1]
        return total_cost


def create_harrypotter_problem(game):
    return HarryPotterProblem(game)
