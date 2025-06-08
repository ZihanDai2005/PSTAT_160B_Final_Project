import numpy as np
import random
from utils import *
import math
import copy

def compute_log_probability(text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the 
    charcter c from permutation_map[c]), given the text statistics
    
    Note: This is quite slow, as it goes through the whole text to compute the probability,
    if you need to compute the probabilities frequently, see compute_log_probability_by_counts.
    
    Arguments:
    text: text, list of characters
    
    permutation_map[c]: gives the character to replace 'c' by
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i
    
    Returns:
    p: log likelihood of the given text
    """
    t = text
    p_map = permutation_map
    cix = char_to_ix
    fr = frequency_statistics
    tm = transition_matrix
    
    i0 = cix[p_map[t[0]]]
    p = np.log(fr[i0])
    i = 0
    while i < len(t)-1:
        subst = p_map[t[i+1]]
        i1 = cix[subst]
        p += np.log(tm[i0, i1])
        i0 = i1
        i += 1
        
    return p

def compute_transition_counts(text, char_to_ix):
    """
    Computes transition counts for a given text, useful to compute if you want to compute 
    the probabilities again and again, using compute_log_probability_by_counts.
    
    Arguments:
    text: Text as a list of characters
    
    char_to_ix: character to index mapping
    
    Returns:
    transition_counts: transition_counts[i, j] gives number of times character j follows i
    """
    N = len(char_to_ix)
    transition_counts = np.zeros((N, N))
    c1 = text[0]
    i = 0
    while i < len(text)-1:
        c2 = text[i+1]
        transition_counts[char_to_ix[c1],char_to_ix[c2]] += 1
        c1 = c2
        i += 1
    
    return transition_counts

def compute_log_probability_by_counts(transition_counts, text, permutation_map, char_to_ix, frequency_statistics, transition_matrix):
    """
    Computes the log probability of a text under a given permutation map (switching the 
    charcter c from permutation_map[c]), given the transition counts and the text
    
    Arguments:
    
    transition_counts: a matrix such that transition_counts[i, j] gives the counts of times j follows i,
                       see compute_transition_counts
    
    text: text to compute probability of, should be list of characters
    
    permutation_map[c]: gives the character to replace 'c' by
    
    char_to_ix: characters to index mapping
    
    frequency_statistics: frequency of character i is stored in frequency_statistics[i]
    
    transition_matrix: probability of j following i stored at [i, j] in this matrix
    
    Returns:
    
    p: log likelihood of the given text
    """
    c0 = char_to_ix[permutation_map[text[0]]]
    p = np.log(frequency_statistics[c0])
    
    p_map_indices = {}
    for c1, c2 in permutation_map.items():
        p_map_indices[char_to_ix[c1]] = char_to_ix[c2]
    
    indices = [value for (key, value) in sorted(p_map_indices.items())]
    
    p += np.sum(transition_counts*np.log(transition_matrix[indices,:][:, indices]))
    
    return p

def compute_difference(text_1, text_2):
    """
    Compute the number of times to text differ in character at same positions
    
    Arguments:
    
    text_1: first text list of characters
    text_2: second text, should have same length as text_1
    
    Returns
    cnt: number of times the texts differ in character at same positions
    """
    
    cnt = 0
    for x, y in zip(text_1, text_2):
        if y != x:
            cnt += 1
            
    return cnt

def get_state(text, transition_matrix, frequency_statistics, char_to_ix):
    """
    Generates a default state of given text statistics
    
    Arguments:
    pretty obvious
    
    Returns:
    state: A state that can be used along with,
           compute_probability_of_state, propose_a_move,
           and pretty_state for metropolis_hastings
    
    """
    transition_counts = compute_transition_counts(text, char_to_ix)
    p_map = generate_identity_p_map(char_to_ix.keys())
    
    state = {"text" : text, "transition_matrix" : transition_matrix, 
             "frequency_statistics" : frequency_statistics, "char_to_ix" : char_to_ix,
            "permutation_map" : p_map, "transition_counts" : transition_counts}
    
    return state

def compute_probability_of_state(state):
    """
    Computes the probability of given state using compute_log_probability_by_counts
    """
    
    p = compute_log_probability_by_counts(state["transition_counts"], state["text"], state["permutation_map"], 
                                          state["char_to_ix"], state["frequency_statistics"], state["transition_matrix"])
    
    return p

def propose_a_move(state):
    """
    Proposes a new move for the given state, 
    by moving one step (randomly swapping two characters)
    """
    new_state = {}
    for key, value in state.items():
        new_state[key] = value
    new_state["permutation_map"] = move_one_step(state["permutation_map"])
    return new_state

def pretty_state(state, full=True):
    """
    Returns the state in a pretty format
    """
    if not full:
        return pretty_string(scramble_text(state["text"][1:200], state["permutation_map"]), full)
    else:
        return pretty_string(scramble_text(state["text"], state["permutation_map"]), full)


def load_text_as_chars(filepath):
    """
    Reads a text file and returns its content as a list of characters.

    Args:
        filepath (str): The path to the .txt file.

    Returns:
        list: A list of characters from the file.
              Returns an empty list if the file cannot be read or is empty.
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
            return list(content)
    except FileNotFoundError:
        print(f"Error: File not found at {filepath}")
        return []
    except Exception as e:
        print(f"An error occurred: {e}")
        return []


def calculate_text_similarity(original_text_chars, deciphered_text_chars):
    """
    Calculates the character-by-character similarity percentage between two texts,
    even if they have different lengths (compares up to the shorter length).

    Args:
        original_text_chars (list): Original text as a list of characters.
        deciphered_text_chars (list): Deciphered text as a list of characters.

    Returns:
        float: Similarity percentage (0.0 to 100.0), based on the overlapping portion.
               Returns 0.0 if either text is empty.
    """
    if not original_text_chars or not deciphered_text_chars:
        print("Warning: One or both texts are empty.")
        return 0.0

    min_length = min(len(original_text_chars), len(deciphered_text_chars))
    if min_length == 0:
        return 0.0

    # Only compare up to the shorter length
    differences = compute_difference(
        original_text_chars[:min_length], 
        deciphered_text_chars[:min_length]
    )
    similarity_ratio = (min_length - differences) / min_length
    return similarity_ratio * 100.0

def weighted_proposal(state, log_density, sample_k=100):
    """
    Weighted proposal function for the Metropolis-Hastings algorithm.

    Compared to the original random swap strategy (e.g., propose_a_move),
    this function prioritizes swaps that are likely to improve the log-likelihood,
    thus improving acceptance rate and convergence speed.

    Args:
        state (dict): Current state containing fields like permutation_map, text, etc.
        log_density (function): A function that computes log-probability of the state.
        sample_k (int): Maximum number of candidate swaps to evaluate (for efficiency).

    Returns:
        proposed_state (dict): A new proposed state with a permutation_map updated via a weighted swap.
    """
    current_score = log_density(state)
    keys = list(state["permutation_map"].keys())
    n = len(keys)

    candidates = []
    weights = []

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sampled_pairs = random.sample(all_pairs, min(sample_k, len(all_pairs)))

    for i, j in sampled_pairs:
        new_perm_map = copy.deepcopy(state["permutation_map"])
        ki, kj = keys[i], keys[j]
        new_perm_map[ki], new_perm_map[kj] = new_perm_map[kj], new_perm_map[ki]

        new_state = dict(state)
        new_state["permutation_map"] = new_perm_map

        new_score = log_density(new_state)
        delta = new_score - current_score
        weight = math.exp(delta) if delta < 0 else delta

        if weight > 0:
            candidates.append(((ki, kj), new_state))
            weights.append(weight)

    if not candidates:
        # fallback to uniform move
        fallback_state = propose_a_move(state)
        return fallback_state, 1.0, 1.0

    selected_idx = random.choices(range(len(candidates)), weights=weights, k=1)[0]
    (ki, kj), proposed_state = candidates[selected_idx]

    # Recalculate q(x, y)
    q_xy = weights[selected_idx] / sum(weights)

    return proposed_state, (ki, kj), q_xy
