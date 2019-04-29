import itertools
import math
import pdb
import re

import numpy as np

PROBABILITY_OF = 0
PREVIOUS_STATE = 1
PREVIOUS_KNESS = 2

def lmap(*args):
    return list(map(*args))

def fidx(a_list):
    return range(len(a_list))

def state_smoother(from_name, to_name):
    if to_name == "BEGIN":
        return 0
    elif from_name == "END":
        return 0
    else:
        return 1

def parse_state(a_file):
    with open(a_file) as state_file:
        f = lmap(lambda x: x.strip(), state_file.readlines())

    no_states = int(f[0])
    state_list = [f[i] for i in range(1, no_states + 1)]

    parse = {
        ## The number of states
        'count': no_states,
        ## states[i] is the readable of state i.
        'states': [f[i] for i in range(1, no_states + 1)],
        ## transition[from_state][to_state] is frequency of transition, from -> to.
        ## initialise to 0, then set below
        'transition': None,
        'ways': None
    }

    ## Raw data from the input file - ones for smoothing
    ways = np.ones([no_states, no_states])
    ways[:, state_list.index("BEGIN")] = 0
    ways[state_list.index("END"), :] = 0
    ways[state_list.index("END"), state_list.index("END")] = 1 ## Make divide happy

    for x in range(no_states + 1, len(f)):
        from_state, to_state, this_ways = lmap(int, f[x].split())
        ways[from_state, to_state] += this_ways

    ## Convert raw number to probabilistics
    transitions = ways / ways.sum(axis=1)[:,None]

    return {
        'count': no_states,
        'states': state_list,
        'ways': ways,
        'transitions': transitions
    }

def parse_symbol(a_file, state_data):
    with open(a_file) as parse_file:
        f = lmap(lambda x: x.strip(), parse_file.readlines())

    M = int(f[0])
    no_symbols = int(f[0]) + 1 ## Add UNK
    ## Careful with the indexing; 0:M is all symbols, so
    ## 1:M+1 is also all symbols
    symbol_list = f[1:M+1] + ["UNK"]

    ## reverse lookup
    lookup = {sym: i for i, sym in enumerate(symbol_list)}

    ## Data from file. Raw symbol count, +1 for nex line, +1 for the count
    ## at the start of the file
    ways = np.ones([state_data['count'], no_symbols])
    for x in range(M + 1, len(f)):
        from_state, symbol, this_ways = lmap(int, f[x].split())
        ways[from_state, symbol] += this_ways

    ## Convert to probabilities
    emissions = ways / ways.sum(axis=1)[:, None]
    emissions[state_data['states'].index('END'), :] = 0

    return {
        'count': no_symbols,
        'symbols': symbol_list,
        'lookup': lookup,
        'ways': ways,
        'emissions': emissions
    }

def parse_query(a_file, symbol_data):
    with open(a_file) as query_file:
        f = lmap(lambda x: x.strip(), query_file.readlines())

    unk_idx = symbol_data['symbols'].index("UNK")

    def mapper(tokens):
        return {
            'tokens': tokens,
            'symbols': lmap(lambda x: symbol_data['lookup'].get(x, unk_idx), tokens)
        }

    tokens = lmap(lambda x: re.findall(r'([,()/&-]|[^ ,()/&\n-]+)', x), f)

    return lmap(mapper, tokens)

def viterbi_algorithm(State_File, Symbol_File, Query_File):
    pass

def viterbi_fill(state, symbol, k,
                 specific, sym, s):
    ## Given the general data [state, symbol, k],
    ## specific is the output of the previous most-likely states of the last interation:
    ##  specific[s][k][PREVIOUS_STATE]: The top k candidates of being in state s
    ##  specific[s][k][PROBABILITY_OF]: The top k probabilities of being state s
    ## specific can be garbage for sym = 'B'.
    ## s is an integer, representing a state.
    paths = []

    ## All possible ways to reach this cell in the table.
    for t in range(state['count']):
        for i in range(k):
            # probability of using this path
            if (not sym in ['B', 'E']):
                ## This is a normal symbol
                ## From encodes state and k-idx in that state
                prob = symbol['emissions'][s][sym] * state['transitions'][t][s] * specific[t, i, PROBABILITY_OF]
                paths.append({'prob': prob, 'from': t, 'from-k': i})
            elif (sym == 'E') and (s == state['states'].index('END')):
                prob = state['transitions'][t][s] * specific[t, i, PROBABILITY_OF]
                paths.append({'prob': prob, 'from': t, 'from-k': i})
            else:
                ## Not valid
                paths.append({'prob': 0, 'from': -1, 'from-k': -1})

    ## Starting out is a bit special
    if (sym == 'B') and (s == state['states'].index('BEGIN')):
        paths = [{'prob': 1, 'from': -1, 'from-k': -1}] + [{'prob': 0, 'from': -1, 'from-k': -1} for i in range(k - 1)]

    ## We only need the k most probable paths
    paths.sort(key = lambda x: x['prob'], reverse = True)
    ret = np.zeros([k, 3])
    ret[:, PREVIOUS_STATE] = lmap(lambda x: x['from'], paths[:k])
    ret[:, PREVIOUS_KNESS] = lmap(lambda x: x['from-k'], paths[:k])
    ret[:, PROBABILITY_OF] = lmap(lambda x: x['prob'], paths[:k])
    return ret

def viterbi_make_cache(state_data, symbol_data, observations, k):
    ## We need k paths, so we extend the matrix to be of size k.
    ## Observations +2 gives us initial and final state
    ## The final axis is:
    ##  PREVIOUS_STATE - The state we came from to achieve this result
    ##  PREVIOUS_KNESS - Are we most, 2nd most, etc likely path from previous state?
    ##  PROBABILITY_OF - The probability of being in state s after emitting token i.
    ## Pretend BEGIN emits B and END emits E
    padded_obs = ['B'] + observations + ['E']
    cache = np.zeros([len(padded_obs), state_data['count'], k, 3])

    ## We start in the BEGIN state
    for s, name in enumerate(state_data['states']):
        cache[0, s, 0, PREVIOUS_STATE] = -1
        cache[0, s, 0, PREVIOUS_KNESS] = -1
        if name == "BEGIN":
            cache[0, s, 0, PROBABILITY_OF] = 1

    ## Being sneaky - o is really the index of the /previous/ symbol.
    ## ie, conceptually, 0 is before time and 1 is the first symbol.
    for o, sym in enumerate(padded_obs):
        for s, name in enumerate(state_data['states']):
            cache[o, s, :, :] = viterbi_fill(state_data, symbol_data, k,
                                             cache[o - 1, :, :, :], sym, s)

    return cache

def top_k_cache_unwind(cache, obs, s, i, k):
    if obs == -1:
        return [], 0

    next_s = int(round(cache[obs, s, i, PREVIOUS_STATE]))
    next_i = int(round(cache[obs, s, i, PREVIOUS_KNESS]))

    return (top_k_cache_unwind(cache, obs-1, next_s, next_i, k)[0] + [s], math.log(cache[obs, s, i, PROBABILITY_OF]))

def top_k_viterbi(state, symbol, query, k):
    ret = []

    for q in query:
        cache = viterbi_make_cache(state, symbol, q['symbols'], k)
        tmp = lmap(lambda i: top_k_cache_unwind(cache, len(q['symbols']) + 1, state['states'].index('END'), i, k),
             range(k))
        tmp = lmap(lambda x: x[0] + [x[1]], tmp)
        ret = ret + tmp

    return ret

def advanced_decoding(State_File, Symbol_File, Query_File):
    ## remove some of the Add-1 smoothing? Enforce certain orderings?
    ## Assume queries end with STATE-Postcode (regex check?)
    ## numeric data can only correspond to numeric states
    ## is there additional information in the ','?
    ## Match states with regex. Set prob to 0 if regex doesn't match:
    ##  Punctuation is easy
    ##  Unit: [Uu].*[0-9]+
    pass

state = parse_state("./dev_set/State_File")
symbol = parse_symbol("./dev_set/Symbol_File", state)
query = parse_query("./dev_set/Query_File", symbol)

#state = parse_state("./toy_example/State_File")
#symbol = parse_symbol("./toy_example/Symbol_File", state)
#query = parse_query("./toy_example/Query_File", symbol)

def check_data(state, symbol, query):
    print('\n_state_')
    print('count', state['count'])
    print('states', list(enumerate(state['states'])))
    ##print('random transition (5 -> 6):', state['transition'][5][6])
    print('random transition (0 -> 1):', state['transitions'][0][1])

    print('\n_symbol_')
    print('count', symbol['count'])
    print('symbol', list(enumerate(symbol['symbols'][0:11])))
    #print('random emission (10 - {} | 3 - {}):'.format(symbol['symbol'][10], state['states'][3]),
    #      symbol['emission'][3][10])
    print('random emission (0 - {} | 1 - {}):'.format(symbol['symbols'][0], state['states'][1]),
          symbol['emissions'][0][1])
    print('there is a lookup table')

    print('\n_query_')
    print('count', len(query))
    ##print('random query:', query[3])
    print('random query:', query[0])

check_data(state, symbol, query)

def pdb_target():
    state = parse_state("./toy_example/State_File")
    symbol = parse_symbol("./toy_example/Symbol_File", state)
    query = parse_query("./toy_example/Query_File", symbol)
    viterbi_observation(state, symbol, query[0]['symbols'], 1)

##print(list(top_k_viterbi(state, symbol, query, 1)))

##unwind cache

#cache = viterbi_observation(state, symbol, query[1]['symbols'], 4)
#print(top_k_cache_unwind(cache, 4, 4, 3, 4))

#lmap(print, top_k_viterbi(state, symbol, query, 1))

pred = top_k_viterbi(state, symbol, query, 1)

labels = open("./dev_set/Query_Label", 'r')
f = lmap(lambda x: lmap(int, x.strip().split()), labels.readlines())

for i, preds in enumerate(pred):
    if f[i] != preds[0:-1]:
        to_state = lambda x: state['states'][x]
        print(['BEGIN'] + query[i]['tokens'])
        print(lmap(to_state, f[i]))
        print(lmap(to_state, preds[0:-1]))
        print('')
