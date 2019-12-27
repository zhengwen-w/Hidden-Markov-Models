import numpy as np

from util import accuracy
from hmm import HMM

def model_training(train_data, tags):
    """
    Train HMM based on training data
    Inputs:
    - train_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - tags: (1*num_tags) a list of POS tags
    Returns:
    - model: an object of HMM class initialized with parameters(pi, A, B, obs_dict, state_dict) you calculated based on train_data
    """
    model = None
    ###################################################
    words = []
    obs_dict = {}
    
    S = len(tags)
    L = len(train_data)
    state_dict = {}
    for i in range(S):
        state_dict[tags[i]] = i
    FTC={}
    for i in range(S):
        FTC[tags[i]] =0
    
    TC ={}
    for i in range(S):
        TC[tags[i]] =0
    
    TTC ={}
    for i in range(S):
        TTC[tags[i]]={tags[j]:0 for j in range(S)}  
    
    TWC ={}
    for i in range(S):
        TWC[tags[i]] ={}
        
    
    
    for line in train_data:
        FTC[line.tags[0]] += 1
        l=line.length

        for index in range(l):
            TC[line.tags[index]] += 1
            TWC[line.tags[index]].setdefault(line.words[index], 0)
            TWC[line.tags[index]][line.words[index]] += 1
            if index < l-1:
                TTC[line.tags[index]][line.tags[index+1]] += 1
            if line.words[index] not in obs_dict:
                obs_dict[line.words[index]] = len(words)
                words.append(line.words[index])      
            

    pi=[]
    for i in tags:
        pi.append(FTC[i])
    pi = np.array(pi)
    pi= pi/len(train_data)
    
    
    model = HMM(pi, np.array([[TTC[s].get(ss, 1e-06) for ss in tags] for s in tags]) /np.array([[TC[t]] for t in tags]), np.array([[TWC[s].get(w, 1e-06) for w in words] for s in tags]) / np.array([[TC[t]] for t in tags]), obs_dict, state_dict)
    ###################################################
    return model

# TODO:
def sentence_tagging(test_data, model, tags):
    """
    Inputs:
    - test_data: (1*num_sentence) a list of sentences, each sentence is an object of line class
    - model: an object of HMM class
    Returns:
    - tagging: (num_sentence*num_tagging) a 2D list of output tagging for each sentences on test_data
    """
    tagging = []
    
    ###################################################
            
    n = len(test_data)
    for i in range(n):
        tmp = test_data[i].words
        for word in tmp:
            if word not in model.obs_dict:
                a = len(model.obs_dict)
                model.obs_dict[word] = a
                lll = [[1] for i in range(model.B.shape[0])]
                lll = 1e-06 *  np.array(lll)
                
                
                
                model.B = np.concatenate((model.B,lll), axis=1)
        tagging.append(model.viterbi(tmp))
    ###################################################
    return tagging
