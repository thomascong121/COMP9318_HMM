import pandas as pd
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
nstates,all_states,df = file_parser("./dev_set/State_File")

#create transition matrix
def transition_matrix(nstates,all_states,df):
    matrx = [[0 for j in range(nstates)] for i in range(nstates)]
    ##no transition from end to any other states
    for i in range(nstates-1):
        for j in range(nstates):
            #no transition to begin
            if(all_states[i+1] == 'BEGIN' and all_states[j+1] == 'BEGIN'):
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
nsymbols,all_symbols,df1 = file_parser("./dev_set/Symbol_File")
def emission_prob(nsymbols,df1,state,symbol):
    #if state1 value in the dataframe
    if(state in df.state1.values):
        df_t = df.loc[df.state1==state]
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

# Question 1
def viterbi_algorithm(State_File, Symbol_File, Query_File): # do not change the heading of the function
    nstates,all_states,df = file_parser(State_File)
    nsymbols,all_symbols,df1 = file_parser(Symbol_File)
    tran_matrx = transition_matrix(nstates,all_states,df)


# Question 2
def top_k_viterbi(State_File, Symbol_File, Query_File, k): # do not change the heading of the function
    pass # Replace this line with your implementation...


# Question 3 + Bonus
def advanced_decoding(State_File, Symbol_File, Query_File): # do not change the heading of the function
    pass # Replace this line with your implementation...
