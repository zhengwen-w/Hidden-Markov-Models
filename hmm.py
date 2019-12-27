from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        ans=[]
        for index in Osequence:
            ans.append(self.obs_dict[index])
        for j in range(S):
            alpha[j,0] = self.pi[j]*self.B[j,ans[0]]

        for t in range(1, L):
            a1=(self.A).T
            a2=alpha[:,t-1]
            alpha[:, t] = self.B[:, ans[t]] * np.dot(a1,a2)
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        ans=[]
        for index in Osequence:
            ans.append(self.obs_dict[index])
        for j in range(S):
            beta[j,L-1] = 1
        
        for i in range(L-2,-1,-1):
            a1 = beta[:,i+1]
            a2=self.B[:,ans[i+1]]
            a3=np.multiply(a1,a2)
            beta[:,i] = np.dot(self.A, a3)
                                      
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        ans=[]
        for index in Osequence:
            ans.append(self.obs_dict[index])
        beta = self.backward(Osequence)
        l=len(self.pi)
        res=[]
        for i in range(l):
            a=beta[i,0]*self.pi[i]*self.B[i,0]
            res.append(a)
        prob= sum(res)
            
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        ans=[]
        for index in Osequence:
            ans.append(self.obs_dict[index])
        alphaBeta = self.forward(Osequence) * self.backward(Osequence)
        O=np.sum(self.forward(Osequence)[:, len(ans)-1])
        prob = alphaBeta / O
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        n=len(Osequence)
        for t in range(L-1):
            for x in range(S):
                for y in range(S):
                    a1=self.forward(Osequence)[x, t] * self.A[x, y]
                    a2=a1* self.B[y, self.obs_dict[Osequence[t+1]]] 
                    a3=a2* self.backward(Osequence)[y, t+1]
                    prob[x, y, t] = a3 / sum(self.forward(Osequence)[:,n- 1])
        return prob
        

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Q3.3 Edit here
        ###################################################
        S = len(self.pi)
        
        
        O= list()
        l=len(self.obs_dict)
        for t in Osequence:
            if t in self.obs_dict:
                O.append(self.obs_dict[t])
            else:
                O.append(l)
                self.obs_dict[t] = l
                li=[0.01]*l
                c=np.array(li)
                nc = np.expand_dims(np.transpose(c), axis=1)
                length=len(self.B)
                res=[]
                for i in range(length):
                    level=[]
                    for k in range(len(self.B[i])):
                        level.append(self.B[i][k])
                    for j in range(len(nc[i])):
                        level.append(self.B[i][j])
                    res.append(level)
                self.B = np.array(res)
        L = len(O)
        delta = np.zeros([len(self.pi), len(O)])
        paths = np.zeros([len(self.pi),len(O)], dtype="int")
        tmp = np.zeros([len(O)], dtype="int")
        
        for j in range(S):
            delta[j, 0] = self.pi[j] * self.B[j, O[0]]
        for t in range(1, L):
            for j in range(S):
                deltas=[]
                for k in range(S):
                    deltas.append(delta[k, t - 1] * self.A[k, j] * self.B[j, O[t]])          
                delta[j, t] = max(deltas) 
                paths[j, t] = np.argmax(deltas)
        tmp[L-1] = np.argmax(delta[:,L-1])
        
        for index in range(L-1, 0,-1):
            tmp[index-1] = paths[tmp[index], index]
        path = list(tmp)
        index = 0
        d={}

        for k,v in self.state_dict.items():
            d[v]=k
        
        for i in path:
            for k in self.state_dict:
                if self.state_dict[k] == i:
                    path[index] = k 
                else:
                    continue
            index+=1
            

        
        return path
