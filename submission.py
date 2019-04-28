import re
from math import log
import numpy as np
#extract transition probability from state file A()
#extract emission probability from symbol file B()
#========================trans prob=======================       
def file_parser(fname):
    '''
    input: file name/path
    output: 
    1.total: number of states/symbols
    2.dic: dictionary {'state/symbol name':index}
    '''
    dic={}
    f = open(fname,'r')
    for i, l in enumerate(f):
        if(i == 0):
            total = int(l.strip())  
        elif(i <= total):
            dic[l.strip()] = i-1
    return total,dic

#create transition matrix
def transition_matrix(nstates,all_states,state_file):
    '''
    input: 
    1.nstates: number of states/symbols
    2.all_states: dictionary {'state/symbol name':index}
    3.state_file: state_file
    output: transition_matrix
    '''
    matrx = np.zeros(shape=(nstates,nstates))
    #1st scan of state file
    f = open(state_file,'r')
    for i, line in enumerate(f):
        if(i > nstates):
            line = line.split()
            row_number = int(line[0])
            col_number = int(line[1])
            matrx[row_number][col_number] = int(line[2]) 
    for i in range(len(matrx)-1):
        total = np.sum(matrx[i])
        smoothing = (total*0.02)/(nstates - 1)
        for j in range(len(matrx[i])):
            if(j == all_states['BEGIN']):
                continue
            matrx[i][j] = (matrx[i][j]+1)/(total + 1*(nstates - 1))  
    f.close()
    return matrx

#========================Emission prob=======================               
def emission_matrix(nstates,nsymbols,symbol_file):
    '''
    input:
    1.nstates: number of states
    2.nsymbols: number of symbols
    3.symbol_file: symbol_file
    output: emission_matrix
    '''
    matrx = np.zeros(shape=(nstates,nsymbols+1))
    #1st scan of symbol file
    f = open(symbol_file,'r')
    for i, line in enumerate(f):
        if(i > nsymbols):
            line = line.split()
            row_number = int(line[0])
            col_number = int(line[1])
            matrx[row_number][col_number] = int(line[2]) 
    for i in range(len(matrx)-2):
        total = np.sum(matrx[i])
        #smoothing = (total*0.02)/(nsymbols + 1)
        for j in range(len(matrx[i])):
            matrx[i][j] = (matrx[i][j] + 1)/(total + 1*(nsymbols + 1))
    f.close()
    return matrx
    
def tokenize(address):
    '''
    input: a string
    output: list of tokens of the input string
    '''
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
    #set up initial variables
    nstates,all_states = file_parser(State_File)
    nsymbols,all_symbols = file_parser(Symbol_File)
    trans_matrx = transition_matrix(nstates,all_states,State_File)
    emis_matrx = emission_matrix(nstates,nsymbols,Symbol_File)
    query = open(Query_File, 'r')
    # process each line of query
    final = []
    for l in query:
        line = tokenize(l)
        dp_matrx = np.zeros(shape=(nstates,len(line)))
        track_matrx = np.zeros(shape=(nstates,len(line)))
        begin_state = all_states['BEGIN']
        for e in range(len(line)):
            if line[e] in all_symbols:
                symbol_index = all_symbols[line[e]]
            else:
                symbol_index = nsymbols  # UNKnown
            ### store first column to start the algorithm
            if e == 0:
                begin_trans = trans_matrx[begin_state]
                first_emis = emis_matrx[...,symbol_index]
                dp_matrx[...,0] = np.multiply(begin_trans,first_emis)
                track_matrx[...,0] = np.array([begin_state]*nstates)
            ### deal with last column to end the algorithm
            elif e == len(line) - 1:
                end_trans = trans_matrx[...,-1]
                dp_matrx[...,-1] = np.multiply(dp_matrx[...,-2],end_trans)
                track_matrx[...,-1] = np.arange(nstates)
            ### general case of the algorithm
            else:
                for st in range(nstates):
                    transistion = trans_matrx[...,st]
                    emission = emis_matrx[st][symbol_index]
                    product = np.dot(transistion,emission)
                    temp = np.multiply(dp_matrx[...,e-1],product)#last column*T*E
                    dp_matrx[st][e] = np.max(temp)
                    track_matrx[st][e] = np.argmax(temp)
        ### final look up
        #step1: look for the largest value of the last column in dp_matrx and record its index
        maximum = np.max(dp_matrx[...,-1])
        maximum_index = np.argmax(dp_matrx[...,-1])
        #step2: back track in the track matrx
        state_seq = [all_states['END']]
        for k in range(len(line)-1,-1,-1):
            state_seq.append(int(track_matrx[maximum_index][k]))
            maximum_index = int(track_matrx[maximum_index][k])
        state_seq.reverse()
        log_max = log(maximum)
        ###################################
        state_seq.append(log_max)
        ##################################
        #state_seq_str = [str(i) for i in state_seq]
        #print(' '.join(state_seq_str))
        #print(state_seq)
        final.append(state_seq)
    return final

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    #set up initial variables
    nstates, all_states = file_parser(State_File)
    nsymbols, all_symbols = file_parser(Symbol_File)
    trans_matrx = transition_matrix(nstates, all_states, State_File)
    emis_matrx = emission_matrix(nstates, nsymbols, Symbol_File)
    begin_state = all_states['BEGIN']
    answer_matrix = []
    query = open(Query_File, 'r')
    
    # process each line of query
    for l in query:
        line = tokenize(l)
        # dp_matrx = [[0 for i in range(nstates)] for j in range(len(line))] 
        dp_matrx = np.zeros(shape=(nstates, len(line), k)) # [ [ [ k-bests ], ... n_states ], ... n_symbols ]
        track_matrx = np.zeros(shape=(nstates, len(line), k))

        for e in range(len(line)):
            if line[e] in all_symbols:
                symbol_index = all_symbols[line[e]]
            else:
                symbol_index = nsymbols  # UNKnown

            ### store first column to start the algorithm
            if e == 0:
                begin_trans = trans_matrx[begin_state]
                first_emis = emis_matrx[..., symbol_index]
                dp_matrx[:,0,0] = np.multiply(begin_trans, first_emis)
                for i in range(k):
                    track_matrx[:,0,i] = np.array([begin_state] * nstates)

            else:
                for st in range(nstates):
                    trans = trans_matrx[..., st]
                    prev_col = dp_matrx[:, e-1, :]

                    ### general case of the algorithm
                    if e != (len(line) - 1):
                        emission = emis_matrx[st][symbol_index]
                        transition = np.vstack((trans for _ in range(k))).T
                        product = np.dot(transition, emission)
                        all_candidates = np.multiply(prev_col, product).flatten() # prev_col*T*E

                        # find the k largest values and their indices (not sorted) in O(n) time 
                        tmp = np.partition(-all_candidates, k)
                        k_best = -tmp[:k]
                        tmp = np.argpartition(-all_candidates, k)
                        i_k_best = tmp[:k]

                        # combine the lists and sort them largest to smallest
                        #   [ [probs] , [indices] ]
                        k_candidates = zip(k_best, i_k_best)
                        k_candidates = sorted(k_candidates, key=lambda x:x[1])  # if there is a tie, choose smaller index
                        k_candidates = sorted(k_candidates, key=lambda x:x[0], reverse = 1)
                        k_candidates = np.array(k_candidates)

                        dp_matrx[st,e,:] = k_candidates[:,0]
                        track_matrx[st,e,:] = k_candidates[:,1] # stores FLATTENED indices of k predecessors 

                    ### END column
                    else:
                        all_candidates = np.multiply(prev_col, trans.reshape((nstates,1))) # prev_col*T*E
                        dp_matrx[:,-1,:] = all_candidates
                        predecessors = np.arange(k*nstates).reshape((nstates,k))
                        track_matrx[:,-1,:] = predecessors

        ### find the k best sequences of states by back tracking
        # step1: sort all of the final probabilities and choose k largest values and their indices
        final_probs = list(zip( all_candidates.flatten() , predecessors.flatten() ))
        sorted_probs = sorted(final_probs, key=lambda x:x[1])
        sorted_probs = sorted(sorted_probs, key=lambda x:x[0], reverse = 1)

        # step2: back track in the track matrx
        for i in range(k):
            cur_prob = sorted_probs[i][0]   # i-th largest prob
            cur_index = sorted_probs[i][1]  # start from the END col
            state_seq = [all_states['END']]

            for col in range(len(line)-1, -1, -1):
                # store the BEGIN state as is.
                if col == 0:
                    #                                  n-th state(row), query,  index (m-th largest)   
                    state_seq.append(int(track_matrx[ cur_index // k ,  col , cur_index % k ]))
                else:    
                    #                                  n-th state(row), query, index (m-th largest)   
                    state_seq.append( int(track_matrx[ cur_index // k ,  col , cur_index % k ]) // k )

                cur_index = int(track_matrx[ cur_index//k , col , cur_index%k ])
            state_seq.reverse()

            log_prob = log(cur_prob)
            ##################################
            state_seq.append(log_prob)
            ##################################
            # state_seq_str = [str(i) for i in state_seq]
            # print(' '.join(state_seq_str))
            answer_matrix.append(state_seq)
    return answer_matrix

          
            


# Question 3 + Bonus
def emission_matrix_smoothed(nstates,nsymbols,symbol_file):
    '''
    input:
    1.nstates: number of states
    2.nsymbols: number of symbols
    3.symbol_file: symbol_file
    output: emission_matrix
    '''
    matrx = np.zeros(shape=(nstates,nsymbols+1))
    #1st scan of symbol file
    f = open(symbol_file,'r')
    for i, line in enumerate(f):
        if(i > nsymbols):
            line = line.split()
            row_number = int(line[0])
            col_number = int(line[1])
            matrx[row_number][col_number] = int(line[2]) 
    for i in range(len(matrx)-2):
        total = np.sum(matrx[i])
        smoothing = (total*0.02)/(nsymbols + 1)
        for j in range(len(matrx[i])):
            matrx[i][j] = (matrx[i][j] + smoothing)/(total + smoothing*(nsymbols + 1))
    f.close()
    return matrx

def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    #set up initial variables
    nstates,all_states = file_parser(State_File)
    nsymbols,all_symbols = file_parser(Symbol_File)
    trans_matrx = transition_matrix(nstates,all_states,State_File)
    emis_matrx = emission_matrix_smoothed(nstates,nsymbols,Symbol_File)
    query = open(Query_File, 'r')
    # process each line of query
    final = []
    for l in query:
        line = tokenize(l)
        dp_matrx = np.zeros(shape=(nstates,len(line)))
        track_matrx = np.zeros(shape=(nstates,len(line)))
        begin_state = all_states['BEGIN']
        for e in range(len(line)):
            if line[e] in all_symbols:
                symbol_index = all_symbols[line[e]]
            else:
                symbol_index = nsymbols  # UNKnown
            ### store first column to start the algorithm
            if e == 0:
                begin_trans = trans_matrx[begin_state]
                first_emis = emis_matrx[...,symbol_index]
                dp_matrx[...,0] = np.multiply(begin_trans,first_emis)
                track_matrx[...,0] = np.array([begin_state]*nstates)
            ### deal with last column to end the algorithm
            elif e == len(line) - 1:
                end_trans = trans_matrx[...,-1]
                dp_matrx[...,-1] = np.multiply(dp_matrx[...,-2],end_trans)
                track_matrx[...,-1] = np.arange(nstates)
            ### general case of the algorithm
            else:
                for st in range(nstates):
                    transistion = trans_matrx[...,st]
                    emission = emis_matrx[st][symbol_index]
                    product = np.dot(transistion,emission)
                    temp = np.multiply(dp_matrx[...,e-1],product)#last column*T*E
                    dp_matrx[st][e] = np.max(temp)
                    track_matrx[st][e] = np.argmax(temp)
        ### final look up
        #step1: look for the largest value of the last column in dp_matrx and record its index
        maximum = np.max(dp_matrx[...,-1])
        maximum_index = np.argmax(dp_matrx[...,-1])
        #step2: back track in the track matrx
        state_seq = [all_states['END']]
        for k in range(len(line)-1,-1,-1):
            state_seq.append(int(track_matrx[maximum_index][k]))
            maximum_index = int(track_matrx[maximum_index][k])
        state_seq.reverse()
        log_max = log(maximum)
        ###################################
        state_seq.append(log_max)
        ##################################
        #state_seq_str = [str(i) for i in state_seq]
        #print(' '.join(state_seq_str))
        #print(state_seq)
        final.append(state_seq)
    return final


#perl -pi -e 'chomp if eof' Q1
#a = top_k_viterbi("./toy_example/State_File","./toy_example/Symbol_File","./toy_example/Query_File",4)
# a = viterbi_algorithm("./toy_example/State_File","./toy_example/Symbol_File","./toy_example/Query_File")
# for i in a:
# 	print(i)