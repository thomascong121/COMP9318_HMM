import pandas as pd
import re
from math import log
#========================trans prob=======================       
def file_parser(fname):
    states={}
    with open(fname) as f:
        nstates = 99999
        count = 0
        dfl=[]
        for i, l in enumerate(f):
            if(count <= nstates):
                if(i==0):
                    nstates = int(l.strip())
                else:
                    states[l.strip()] = i-1
            else:
                dfl.append(l.split())
            count+=1
        df = pd.DataFrame(dfl, columns = ['state1', 'state2','frequency']) 
        df= df.apply(pd.to_numeric, errors="ignore")
    return nstates,states,df

#create transition matrix
def transition_matrix(nstates,all_states,df):
    matrx = [[0 for j in range(nstates)] for i in range(nstates)]
    # no transition from end to any other states
    for i in range(nstates-1):
        for j in range(nstates):
            # no transition to begin
            if(all_states['BEGIN'] == j):
                matrx[i][j] = 0
                continue
            if(i in df.state1.values):
                df_t = df.loc[df.state1==i]
                n_i = df_t.frequency.sum()
                # if state2 value in the dataframe
                if(j in df_t.state2.values):
                    n_ij = df_t[df_t[df_t.state1==i].state2==j].frequency.values[0]
                else:
                    n_ij = 0
            else:
                n_i = n_ij = 0
            matrx[i][j]=(n_ij+1)/(n_i+nstates-1)
    return matrx

#========================Emission prob=======================      
# prob that a [symbol] belongs to [state] (both are indices)
def emission_prob(nsymbols, df1, states_dict, state, symbol):
    if(states_dict['BEGIN'] == state or states_dict['END'] == state):
        return 0
    #if state1 value in the dataframe
    if(state in df1.state1.values):
        # df_t: all records starting at [state]
        df_t = df1.loc[df1.state1==state]
        # total number of records related to this state
        n_i = df_t.frequency.sum()
        # if state2 value in the dataframe
        # find the prob of this specific relation: state->symbol
        if(symbol in df_t.state2.values):
            n_ij = df_t[df_t[df_t.state1==state].state2==symbol].frequency.values[0]
        else: # UNK
            n_ij = 0
    else:
        n_i = n_ij = 0

    # emission prob with +1 smoothing 
    emission_prob=(n_ij+1)/(n_i+nsymbols+1)
    return emission_prob     

#=======================tokenize==============================
def tokenize(address):
    aprime = re.split('\s+',address)
    for i in aprime:
        i = i.strip()
    #result = ['BEGIN']
    result=[]
    for i in aprime:
        temp = re.split('(\*|\,|\(|\)|\/|-|&|\")',i)
        for j in temp:
            if(j):
                result.append(j)
    result.append('END')
    return result

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    nstates,all_states,df_state = file_parser(State_File)
    nsymbols,all_symbols,df_symbol = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df_state)


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    nstates,all_states,df_state = file_parser(State_File)
    nsymbols,all_symbols,df_symbol = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df_state)
    result_mat = []
    answer = []

    qf = open(Query_File, 'r')
    # process each line of query
    for l in qf:
        line = tokenize(l)
        col1 = []
        col2 = []
        i_end = all_states['END']

        ### store first column to start the algorithm
        if line[0] in all_symbols:
            symbol_index = all_symbols[line[0]]
        else:
            symbol_index = nsymbols+1  # UNKnown
        for i in range(nstates):
            # get the index of cur symbol. Set a new index to new symbols
            tmp1 = emission_prob(nsymbols, df_symbol, all_states, i, symbol_index)
            tmp2 = tran_matrx[nstates-2][i] # transfer from BEGIN to other states
            col1.append([ tmp1 * tmp2, nstates-2 ]) # [ prob, [BEGIN] ]
        result_mat.append(col1)

        ### loop through each element (symbol) of current query.
        for ele in line[1:]:
            # get the index of cur symbol. Set a new index to new symbols
            if ele in all_symbols:
                symbol_index = all_symbols[ele]
            else:
                symbol_index = nsymbols+1  # UNKnown

            # to get each of current state..
            for i in range(nstates):
                # col2.append(max([tran_matrx[j][i] * col1[j] * emission_prob(nsymbols, df_symbol, st_cur, [ele]) for j in range(nstates)]))
                emit = emission_prob(nsymbols, df_symbol, all_states, i, symbol_index)
                
                if ele == 'END':
                    # set the END col (directly transfer into this state).
                    trans_prob = tran_matrx[i][i_end]   # transfer from other states to END
                    prev_prob = col1[i][0]
                    col2.append([trans_prob*prev_prob , i])
                else:
                    prev_of_max = 0
                    tmp_max = 0
                    tmp_prob = 0
                    # loop through prob of each prev states to get cur state
                    for j in range(nstates):
                        # cur_candidate = prev_prob * transition_prob * emission
                        tmp_prob = col1[j][0] * tran_matrx[j][i] * emit
                        # cur prob is the max amongst transitions from all prev states
                        if tmp_prob > tmp_max:
                            tmp_max = tmp_prob
                            prev_of_max = j
                    col2.append([tmp_max , prev_of_max]) # record the predecessor state that transit to cur state (max prob)

            result_mat.append(col2) # store this column to the result
            col1 = col2 # assign cur col to be the previous col
            col2 = []

        ### find the sequence of states by back tracking
        big = 0
        index_prev = -1
        answer = [i_end]
        try:
            # find the largest prob of the last state and get the index of prev state
            big = max(result_mat[-1], key = lambda x: x[0])
            # l_prob, l_prev = zip(*result_mat[-1])
            # ind = l_prob.index(big[0])
            # index_prev = l_prev[ind]
            index_prev = big[1]
            for col in range(len(result_mat)-1, -1, -1):
                index_prev = result_mat[col][index_prev][1]
                answer.append(index_prev)
        except ValueError:
            print('[*] Error! Empty list. No solution found!')

        print(answer[::-1] + [log(big[0])])
        result_mat = []

    return
# top_k_viterbi('toy_example/State_File', 'toy_example/Symbol_File', 'toy_example/Query_File', 3)
top_k_viterbi('dev_set/State_File', 'dev_set/Symbol_File', 'dev_set/Query_File', 3)



# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...
