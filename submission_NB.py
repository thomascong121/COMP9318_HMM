## COMP9318, 19s01
## Project 1
## Owen Riddy, z3251243
## Ann (Be?) z5193703

## Draft Submission

import math
import re
import itertools

import numpy as np

def read_state(File):
    file = open(File, 'r')
    text = file.readlines()
    file.close()
    N = int(text[0])
    begin = 0
    end = 0
    Transition_Probabilities = [0]*N
    for i in range(N):
        Transition_Probabilities[i] = [0]*N
    for l in range(len(text)):
        if l <= N:
            if text[l] == 'BEGIN\n':
                begin = l-1
            if text[l] == 'END\n':
                end = l-1
        else:
            i = int(text[l].split()[0])
            j = int(text[l].split()[1])
            Transition_Probabilities[i][j] = int(text[l].split()[2])
    for i in range(N):
        i_frequency = 0
        for j in range(N):
            i_frequency += Transition_Probabilities[i][j]
        for j in range(N):
            if i == end or j == begin:
                Transition_Probabilities[i][j] = 0
            else:
                Transition_Probabilities[i][j] = (Transition_Probabilities[i][j] + 1)/(i_frequency + N - 1)
    return Transition_Probabilities, N, begin, end

def read_symbol(File, N):
    file = open(File, 'r')
    text = file.readlines()
    file.close()
    M = int(text[0])
    Emission_Probabilities = [0]*N
    for i in range(N):
        Emission_Probabilities[i] = [0]*M
    for l in range(len(text)):
        if l > M:
            i = int(text[l].split()[0])
            j = int(text[l].split()[1])
            Emission_Probabilities[i][j] = int(text[l].split()[2])
    for i in range(N):
        i_frequency = 0
        for j in range(M):
            i_frequency += Emission_Probabilities[i][j]
        for j in range(M):
            Emission_Probabilities[i][j] = (Emission_Probabilities[i][j] + 1) / (i_frequency + M + 1)
        Emission_Probabilities[i].append(1 / (i_frequency + M + 1))
    return Emission_Probabilities, text[1:M+1] + ['UNK']

def read_query(File,symbol):
    file = open(File, 'r')
    text = file.readlines()
    file.close()
    for i in range(len(text)):
        query = re.findall(r'([,()/&-]|[^ ,()/&\n-]+)', text[i])
        for j in range(len(query)):
            try:
                query[j] = symbol.index(query[j]+'\n')
            except ValueError:
                query[j] = len(symbol) - 1
        text[i] = query
    return text

def query_states(query, Transition_Probabilities, Emission_Probabilities, begin, end):
    sublist = [begin]
    likelihood = [0]*(len(Transition_Probabilities)-2)
    for i in range(len(Transition_Probabilities)-2):
        likelihood[i] = [0]*len(query)
    pointer = [0]*(len(Transition_Probabilities)-2)
    for i in range(len(Transition_Probabilities)-2):
        pointer[i] = [0]*len(query)
    for q in range(len(query)):
        for s in range(len(Transition_Probabilities)-2):
            if q == 0:
                likelihood[s][q] = Transition_Probabilities[begin][s]
                likelihood[s][q] *= Emission_Probabilities[s][query[q]]
            else:
                max_likelihood = 0
                for mle_state in range(len(Transition_Probabilities)-2):
                    le = likelihood[mle_state][q-1] * Transition_Probabilities[mle_state][s]
                    le *= Emission_Probabilities[s][query[q]]
                    if le > max_likelihood:
                        pointer[s][q] = mle_state
                        max_likelihood = le
                likelihood[s][q] = max_likelihood
    ll = likelihood[0][len(query)-1] * Transition_Probabilities[0][end]
    last_state = 0
    for S in range(len(Transition_Probabilities)-2):
        if likelihood[S][len(query)-1] * Transition_Probabilities[S][end] > ll:
            last_state = S
            ll = likelihood[S][len(query)-1] * Transition_Probabilities[S][end]
    temp_sublist = []
    for Q in range(len(query)):
        if Q == 0:
            temp_sublist.append(last_state)
        else:
            temp_sublist.append(pointer[last_state][len(query)-Q])
            last_state = pointer[last_state][len(query)-Q]
    temp_sublist.reverse()
    sublist += temp_sublist
    sublist.append(end)
    sublist.append(math.log(ll))
    return sublist

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    Transition_Probabilities, N, begin, end = read_state(State_File)
    Emission_Probabilities, symbol = read_symbol(Symbol_File, N)
    query = read_query(Query_File, symbol)
    out_put_list = []
    for l in query:
        out_put_list.append(query_states(l, Transition_Probabilities, Emission_Probabilities, begin, end))
    return out_put_list

PROBABILITY_OF = 0
PREVIOUS_STATE = 1
PREVIOUS_KNESS = 2

def lmap(*args):
    return list(map(*args))

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

def viterbi_fill(config, specific, sym, s):
    state = config['state']
    symbol = config['symbol']
    k = config['k']

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


def gen_rules():
    ## A rule is (regex, state. probability)
    ## If the regex matches, then probability of state is probability
    return [
        ## Units start with Unit and end with a number
        (r'^[Uu](nit)?[0-9]+$', 'UnitNumber', 1),
        (r'(?!^[Uu](nit)?[0-9]+$)', 'UnitNumber', 0),
        ## $
        (r'^,$', ',', 1),
        (r'(?!^,$)', ',', 0),
        ## /
        (r'^/$', '/', 1),
        (r'(?!^/$)', '/', 0),
        ## -
        (r'^-$', '-', 1),
        (r'(?!^-$)', '-', 0),
        ## (
        (r'^\($', '(', 1),
        (r'(?!^\($)', '(', 0),
        ## )
        (r'^\)$', ')', 1),
        (r'(?!^\)$)', ')', 0),
        ## &
        (r'^&$', '&', 1),
        (r'(?!^&$)', '&', 0),
        ## Why not? Special case for Kiosks
        (r'^Kiosk[0-9]*$', 'Location-Inside-Building', 1),
        ## Quite common to see street numbers with 'Lot' in them
        (r'^Lot[0-9]+$', 'StreetNumber', 1),
        ## Some states must have a number in them.
        (r'(?!.*[0-9]+)', 'StreetNumber', 0),
        (r'(?!.*[0-9]+)', 'SubNumber', 0),
        (r'(?!.*[0-9]+)', 'UnitNumber', 0),
        ## This isn't a street name
        (r'^Floor[0-9]*$', 'LevelName', 1),
    ]

## Check if rules apply to sym
def apply_rules(config, specific, s, o, token):
    state_data = config['state']
    ##symbol_data = config['symbol']
    rules = config['rules']
    new_specific = np.array(specific)

    state_name = state_data['states'][s]
    for r in rules:
        if re.match(r[0], token) and (state_name == r[1]):
            new_specific[:, PROBABILITY_OF] = r[2]

    return new_specific

def viterbi_make_cache(config, a_query):
    state_data = config['state']
    symbol_data = config['symbol']
    k = config['k']
    sym_list = a_query['symbols']
    tok_list = a_query['tokens']

    ## We need k paths, so we extend the matrix to be of size k.
    ## Observations +2 gives us initial and final state
    ## The final axis is:
    ##  PREVIOUS_STATE - The state we came from to achieve this result
    ##  PREVIOUS_KNESS - Are we most, 2nd most, etc likely path from previous state?
    ##  PROBABILITY_OF - The probability of being in state s after emitting token i.
    ## Pretend BEGIN emits B and END emits E
    padded_obs = ['B'] + sym_list + ['E']
    padded_tok = ['UNK'] + tok_list + ['UNK']
    cache = np.zeros([len(padded_obs), state_data['count'], k, 3])
    print('cache set up',cache)

    ## We start in the BEGIN state
    for s, name in enumerate(state_data['states']):
        cache[0, s, 0, PREVIOUS_STATE] = -1
        cache[0, s, 0, PREVIOUS_KNESS] = -1
        if name == "BEGIN":
            cache[0, s, 0, PROBABILITY_OF] = 1
    print('cache after 1st',cache)

    ## Being sneaky - o is really the index of the /previous/ symbol.
    ## ie, conceptually, 0 is before time and 1 is the first symbol.
    for o, sym in enumerate(padded_obs):
        for s, name in enumerate(state_data['states']):
            next_cache = viterbi_fill(config, cache[o - 1, :, :, :], sym, s)
            print('next_cache without ruls',next_cache)
            if 'rules' in config:
                next_cache = apply_rules(config, next_cache, s, o, padded_tok[o])
                print('next_cache with ruls',next_cache)
            cache[o, s, :, :] = next_cache
            print('cache is',cache)
    return cache

def top_k_cache_unwind(cache, obs, s, i, k):
    if obs == -1:
        return [], 0

    next_s = int(round(cache[obs, s, i, PREVIOUS_STATE]))
    next_i = int(round(cache[obs, s, i, PREVIOUS_KNESS]))

    return (top_k_cache_unwind(cache, obs-1, next_s, next_i, k)[0] + [s], math.log(cache[obs, s, i, PROBABILITY_OF]))

def top_k_viterbi(State_File, Symbol_File, Query_File, k):
    state = parse_state(State_File)
    symbol = parse_symbol(Symbol_File, state)
    query = parse_query(Query_File, symbol)

    config = {
        'state': state,
        'symbol': symbol,
        'k': k,
        'rules': []
    }

    ret = []

    for q in query:
        cache = viterbi_make_cache(config, q)
        tmp = lmap(lambda i: top_k_cache_unwind(cache, len(q['symbols']) + 1, state['states'].index('END'), i, k),
             range(k))
        tmp = lmap(lambda x: x[0] + [x[1]], tmp)
        ret = ret + tmp

    return ret

def advanced_decoding(State_File, Symbol_File, Query_File):
    state = parse_state(State_File)
    symbol = parse_symbol(Symbol_File, state)
    query = parse_query(Query_File, symbol)

    config = {
        'state': state,
        'symbol': symbol,
        'k': 1,
        'rules': gen_rules()
    }

    ret = []

    for q in query:
        cache = viterbi_make_cache(config, q)
        tmp = [top_k_cache_unwind(cache, len(q['symbols']) + 1, state['states'].index('END'), 0, 1)]
        tmp = lmap(lambda x: x[0] + [x[1]], tmp)
        ret = ret + tmp

    return ret
    
a = advanced_decoding("./toy_example/State_File","./toy_example/Symbol_File","./toy_example/Query_File")
for i in a:
    state_seq_str = [str(i[j]) for j in range(len(i)-1)]
    print(' '.join(state_seq_str))


