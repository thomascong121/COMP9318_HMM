import pandas as pd
import re
def state_smoothing():
    pass
def symbol_smoothing():
    pass
#extract transition probability from state file A()
#extract emission probability from symbol file B()
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
                    states[i] = l.strip()
            else:
                dfl.append(l.split())
            count+=1
        df = pd.DataFrame(dfl, columns = ['state1', 'state2','frequency']) 
        df= df.apply(pd.to_numeric, errors="ignore")
    return nstates,states,df
# nstates,all_states,df = file_parser("./dev_set/State_File")

#create transition matrix
def transition_matrix(nstates,all_states,df):
    matrx = [[0 for j in range(nstates)] for i in range(nstates)]
    ##no transition from end to any other states
    for i in range(nstates-1):
        for j in range(nstates):
            #no transition to begin
            if(all_states[j+1] == 'BEGIN'):
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
# nsymbols,all_symbols,df1 = file_parser("./dev_set/Symbol_File")
def emission_prob(nsymbols,df1,state,symbol):
    #if state1 value in the dataframe
    if(state in df1.state1.values):
        df_t = df1.loc[df1.state1==state]
        n_i = df_t.frequency.sum()
        #if state2 value in the dataframe
        if(symbol in df_t.state2.values):
            n_ij = df_t[df_t[df_t.state1==state].state2==symbol].frequency.values[0]
        else:
            n_ij = 0
    else:
        n_i = n_ij = 0
    emission_prob=(n_ij+1)/(n_i+nsymbols+1)
    return emission_prob     
#=======================tokenize==============================
def tokenize(address):
    aprime = re.split('\s+', address)
    for i in aprime:
        i = i.strip()
    result = []
    for i in aprime:
        temp = re.split('(\*|\,|\(|\)|\/|-|&|\")',i)
        for j in temp:
            if(j):
                result.append(j)
    return result
# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    nstates,all_states,df = file_parser(State_File)
    nsymbols,all_symbols,df1 = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df)

    with open(Query_File) as f:
        for l in f:
            tmp = tokenize(l)
            print(tmp)

# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    nstates,all_states,df = file_parser(State_File)
    nsymbols,all_symbols,df1 = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df)
    result_mat = []

    with open(Query_File) as f:
        # loop through each query
        for l in f:
            line = tokenize(l)
            col1 = []
            col2 = []
            # store first column
            for i in range(nstates):
                col1.append(emission_prob(nsymbols, df1, i, 0) /25)
            print(col1)
            result_mat.append(col1)

            # loop through each char/element of current query.
            for ele in line[1:]:
                # as for each state
                for j in df1.state1.values:
                    # col2.append(max([tran_matrx[i][j] * col1[i] * emission_prob(nsymbols, df1, i, ele) for i in range(nstates)]))
                    tmp = []
                    for i in range(nstates):
                        emit = emission_prob(nsymbols, df1, i, ele)
                        tmp.append()

                    col2.append(max([tran_matrx[i][j] * col1[i] *  ]))

                result_mat.append(col2)
                col1 = col2 # assign new col to be the previous col

            # print(result_mat)
            # find the sequence of states
            for cols in result_mat:
                biggest = max(cols)
                for i,e in enumerate(cols):
                    if e==biggest:
                        ind = i
                        break
                print(ind+1 ' - ', end='')
            print('.')
            result_mat = []

    return
top_k_viterbi('toy_example/State_File', 'toy_example/Symbol_File', 'toy_example/Query_File', 3)
# top_k_viterbi('dev_set/State_File', 'dev_set/Symbol_File', 'dev_set/Query_File', 3)



# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...
