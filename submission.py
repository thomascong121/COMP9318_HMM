#Q1
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
    ##no transition from end to any other states
    for i in range(nstates-1):
        for j in range(nstates):
            #no transition to begin
            if(all_states['BEGIN'] == j):
                matrx[i][j] = 0
                continue
            if(i in df.state1.values):
                df_t = df.loc[df.state1==i]
                n_i = df_t.frequency.sum()
                #if state2 value in the dataframe
                if(j in df_t.state2.values):
                    n_ij = df_t[df_t[df_t.state1==i].state2==j].frequency.values[0]
                else:
                    n_ij = 0
            else:
                n_i = n_ij = 0
            matrx[i][j]=(n_ij+1)/(n_i+nstates-1)
    return matrx

#========================Emission prob=======================               

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
def viterbi_algorithm(State_File, Symbol_File, Query_File): 
    nstates,all_states,df_state = file_parser(State_File)
    nsymbols,all_symbols,df_symbol = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df_state)
    f = open(Query_File, "r")
    final=[]
    for query in f:
        query = query.strip()
        tokens = tokenize(query)
        st_symbol_matrx = [[[0,0] for i in range(len(tokens))] for j in range(nstates)]
        #fill in the table column by column
        for t in range(len(tokens)):
            if(tokens[t] in all_symbols):
                symbol_index = all_symbols[tokens[t]]
            else:
                #UNK
                symbol_index = nsymbols+1
            for st in range(nstates-1):
                emis_prob = emission_prob(nsymbols,df_symbol,all_states,st,symbol_index)
                #first column
                if(t == 0):
                    initial = all_states['BEGIN']
                    st_symbol_matrx[st][t][0] = tran_matrx[initial][t]*emis_prob
                    st_symbol_matrx[st][t][1] = nstates-2
                elif(t == len(tokens)-1):
                    end_index = all_states['END']
                    st_symbol_matrx[st][t][0] = tran_matrx[st][end_index]*st_symbol_matrx[st][t-1][0]
                    st_symbol_matrx[st][t][1] = st   
                else:
                    maxm = 0
                    index = 0
                    for prev in range(nstates):
                        cal = st_symbol_matrx[prev][t-1][0]*tran_matrx[prev][st]*emis_prob
                        if(cal > maxm):
                            maxm = cal
                            index = prev
                    st_symbol_matrx[st][t][0] = maxm
                    st_symbol_matrx[st][t][1] = index
        #backtracking
        # find the sequence of states by back tracking
        big = 0
        index_prev = -1
        answer=[nstates-1]
        try:
            # find the largest prob of the last state and get the index of prev state
            b = [i[-1] for i in st_symbol_matrx]
            big = max(b, key = lambda x: x[0])
            index_prev = big[1]
            for col in range(len(tokens)-1, -1, -1):
                index_prev = st_symbol_matrx[index_prev][col][1]
                answer.append(index_prev)
        except ValueError:
            print('[*] Error! Empty list. No solution found!')
        print(answer[::-1] + [log(big[0])])


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...



viterbi_algorithm("./toy_example/State_File","./toy_example/Symbol_File","./toy_example/Query_File")
