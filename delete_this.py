from typing import List, Dict, Tuple

from representation import Representation


def viterbi_conditional_sequence[C, E, S](
        observations: List[E],
        states: List[C],
        start_probabilities: Dict[C, float],
        state_transitions: Representation[C, C, S],
        emissions: Representation[C, E, S]) -> List[C]:
    num_observations = len(observations)
    viterbi_matrix = [{}]
    best_path = {}

    # Initialize the base cases (t == 0)
    for each_state in states:
        transition_info = emissions.get_transition_info(each_state, observations[0])
        viterbi_matrix[0][each_state] = start_probabilities[each_state] * transition_info.frequency
        best_path[each_state] = [each_state]

    # Build the Viterbi matrix for t > 0
    for t in range(1, num_observations):
        viterbi_matrix.append({})
        new_path = {}

        for current_state in states:
            max_prob, previous_state = max(
                (viterbi_matrix[t - 1][prev_state] *
                 state_transitions.get_transition_info(prev_state, current_state).frequency *
                 emissions.get_transition_info(current_state, observations[t]).frequency, prev_state)
                for prev_state in states
            )
            viterbi_matrix[t][current_state] = max_prob
            new_path[current_state] = best_path[previous_state] + [current_state]

        best_path = new_path

    # Find the most probable final state
    last_time_step = num_observations - 1
    final_max_prob, final_state = max((viterbi_matrix[last_time_step][state], state) for state in states)
    return best_path[final_state]


def viterbi_observation_transitions(obs_transitions: List[Tuple[str, str]],
                                    states: List[str],
                                    start_probabilities: Dict[str, float],
                                    trans_probabilities: Dict[str, Dict[str, float]],
                                    emit_probabilities: Dict[str, Dict[Tuple[str, str], float]]) -> List[str]:
    viterbi_matrix = [{}]
    path = {}

    # Initialize the base cases using start probabilities
    for state in states:
        first_transition = obs_transitions[0]
        viterbi_matrix[0][state] = start_probabilities[state] * emit_probabilities[state].get(first_transition, 0)
        path[state] = [state]

    # Iterate over observation transitions
    for i in range(1, len(obs_transitions)):
        viterbi_matrix.append({})
        new_path = {}

        each_transition = obs_transitions[i]
        matrix_conditional = viterbi_matrix[i - 1]
        matrix_consequence = viterbi_matrix[i]
        for curr_state in states:
            prob, state = max(
                (
                    matrix_conditional[prev_state] * trans_probabilities[prev_state][curr_state] *
                    emit_probabilities[curr_state].get(each_transition, 0),
                    prev_state
                )
                for prev_state in states
            )

            matrix_consequence[curr_state] = prob
            new_path[curr_state] = path[state] + [curr_state]

        path = new_path

    # Find the most probable state sequence
    prob, state = max((viterbi_matrix[-1][y], y) for y in states)
    return path[state]


def test_sequence() -> None:
    # Example setup for states, observations, and start probabilities
    states = ['Rainy', 'Sunny']  # Example states
    observations = ['walk', 'shop', 'clean']  # Example observations
    start_probabilities = {'Rainy': 0.6, 'Sunny': 0.4}

    # Creating example state transitions and emissions in Representation format
    state_transitions = Representation(shape='state_transitions')
    emissions = Representation(shape='emissions')

    # Populating the state transitions and emissions with example data
    # Note: In a real scenario, this data should be based on actual observations or domain knowledge

    # Transitioning from Rainy to Rainy with frequency 7
    state_transitions.transition('Rainy', 'Rainy', 7)
    state_transitions.transition('Rainy', 'Sunny', 3)
    state_transitions.transition('Sunny', 'Rainy', 4)
    state_transitions.transition('Sunny', 'Sunny', 6)

    # When Rainy, observation 'walk' occurred 1 time
    emissions.transition('Rainy', 'walk', 1)
    emissions.transition('Rainy', 'shop', 4)
    emissions.transition('Rainy', 'clean', 5)
    emissions.transition('Sunny', 'walk', 6)
    emissions.transition('Sunny', 'shop', 3)
    emissions.transition('Sunny', 'clean', 1)

    # Running the Viterbi algorithm
    most_probable_states = viterbi_conditional_sequence(
        observations, states, start_probabilities, state_transitions, emissions)
    print("Most probable sequence of states:")
    print(most_probable_states)


def test_transitions() -> None:
    # Example states and observation transitions
    states = ['Rainy', 'Sunny']
    obs_transitions = [('walk', 'shop'), ('shop', 'clean'), ('clean', 'walk')]

    # Define start probabilities
    start_probabilities = {'Rainy': 0.6, 'Sunny': 0.4}

    # Define transition probabilities between states
    trans_probabilities = {
        'Rainy': {'Rainy': 0.7, 'Sunny': 0.3},
        'Sunny': {'Rainy': 0.4, 'Sunny': 0.6},
    }

    # Define emission probabilities for observation transitions
    emit_probabilities = {
        'Rainy': {('walk', 'shop'): 0.1, ('shop', 'clean'): 0.4, ('clean', 'walk'): 0.5},
        'Sunny': {('walk', 'shop'): 0.6, ('shop', 'clean'): 0.3, ('clean', 'walk'): 0.1},
    }

    # Run the Viterbi algorithm
    most_probable_states = viterbi_observation_transitions(
        obs_transitions, states, start_probabilities, trans_probabilities, emit_probabilities)
    print("Most probable sequence of states:")
    print(most_probable_states)


if __name__ == "__main__":
    test_sequence()
    test_transitions()
