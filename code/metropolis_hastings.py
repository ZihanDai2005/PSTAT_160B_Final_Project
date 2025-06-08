import numpy as np
import time
import shutil
import random
from deciphering_utils import *

def metropolis_hastings(initial_state, proposal_function, log_density, iters=1000, print_every=500,
                        tolerance=0.02, temperature=1.0, cooling_rate=1.0, error_function=None,
                        pretty_state=None, ground_truth_text_chars=None):
    """
    Extended Metropolis-Hastings with support for:
    - Weighted proposal with asymmetric q(x, y)
    - Temperature-based acceptance (simulated annealing)
    """

    p1 = log_density(initial_state)
    state = initial_state

    errors = []
    cross_entropies = []
    states = [initial_state]

    cnt, accept_cnt, it = 0, 0, 0

    while it < iters:
        if proposal_function.__name__ == "weighted_proposal":
            proposed_state, swap_info, q_xy = weighted_proposal(state, log_density)
            _, _, q_yx = weighted_proposal(proposed_state, log_density)  # Reverse move
        else:
            proposed_state = proposal_function(state)
            q_xy = q_yx = 1.0  # symmetric case

        p2 = log_density(proposed_state)
        cnt += 1

        # Adjust acceptance probability with temp and asymmetric q
        acceptance_ratio = (p2 - p1) / temperature + np.log(q_yx / q_xy)
        if np.log(random.random()) < acceptance_ratio:
            state = proposed_state
            p1 = p2
            it += 1
            accept_cnt += 1

            cross_entropies.append(p1)
            states.append(state)

            if error_function:
                error = error_function(state)
                errors.append(error)

            if it % print_every == 0:
                acc_rate = accept_cnt / cnt
                acc_log = f"\n Entropy: {p1:.4f}, Iteration: {it}, Acceptance Rate: {acc_rate:.4f}"

                if ground_truth_text_chars:
                    deciphered = scramble_text(state["text"], state["permutation_map"])
                    accuracy = calculate_text_similarity(ground_truth_text_chars, deciphered)
                    acc_log += f", Accuracy: {accuracy:.2f}%"

                print("-" * shutil.get_terminal_size().columns)
                print(acc_log)
                print("-" * shutil.get_terminal_size().columns)
                if pretty_state:
                    print(pretty_state(state))

                if acc_rate < tolerance:
                    break

                cnt = 0
                accept_cnt = 0

        temperature *= cooling_rate  # Apply annealing schedule

    return states, cross_entropies, errors if error_function else None
