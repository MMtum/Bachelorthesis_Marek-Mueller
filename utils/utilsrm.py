import numpy as np
import pickle


class ddrideshare:
    def __init__(self, loc_lst,price_lst,seed=2,lam=[0.0,0.0], base=True, params={'A1':[],'A2':[],'Ac1':[],'Ac2':[]},maxx=10,tot_rev=0):
        self.loc_lst = loc_lst
        self.price_lst=price_lst
        self.n=2
        self.d=len(loc_lst)
        self.m=1
        self.lam1=lam[0]; 
        self.lam2=lam[1]
        self.l=[-maxx for i in range(self.d)]
        self.u=[maxx for i in range(self.d)]
        self.tot_rev=0
        if base:
            self.params=getparams(loc_lst)
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        else:
            self.params=params
            self.A1=self.params['A1']
            self.A2=self.params['A2']
            self.Ac1=self.params['Ac1']
            self.Ac2=self.params['Ac2']
        self.I=np.eye(self.d)
        self.seed=seed
        np.random.seed(self.seed)
        self.q_dic=get_data_dic()
        self.locations_ = self.q_dic['locations']
        self.dates_ = self.q_dic['dates']
        self.prices_=self.q_dic['prices']
        self.qlbar=self.q_dic['lyft_mean']
        self.qubar=self.q_dic['uber_mean']
        self.verbose=False

    def setup_distribution(self,centered=False):
        if centered:
            self.qu=self.q_dic['uber centered']
            self.ql=self.q_dic['lyft centered']
        else:
            self.qu=self.q_dic['uber']
            self.ql=self.q_dic['lyft']

        self.qlbar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.qubar_=np.zeros((len(self.loc_lst),len(self.price_lst)))
        self.ql_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        self.qu_=np.zeros((len(self.dates_),len(self.loc_lst),len(self.price_lst)))
        for loc in self.loc_lst:
            for p in self.price_lst:
                self.qlbar_[loc,p]=self.qlbar[loc,p]
                self.qubar_[loc,p]=self.qubar[loc,p]
                self.ql_[:,loc,p]=self.ql[:,loc,p]
                self.qu_[:,loc,p]=self.qu[:,loc,p]

    def proj(self,x):
        y=np.zeros(np.shape(x))
        for i in range(self.n):
            for j in range(self.d):
                if x[i][j]<=self.l[j]:
                    y[i][j]=self.l[j]
                elif self.l[j]<x[i][j] and x[i][j]<self.u[j]:
                    y[i][j]=x[i][j]
                else:
                    y[i][j]=self.u[j]
        return y

    def sample_base_demand(self,q_,n=1, batch=1):
        vals=np.random.choice(q_,size=batch)
        return np.mean(vals)

    def D_z(self, q_, locs=None,batch=1):
        z0=[]
        if locs==None:
            for loc_id in self.loc_lst:
                z0.append(self.sample_base_demand(q_[loc_id], batch=batch))
        else:
            for loc_id in locs:
                z0.append(self.sample_base_demand(q_[loc_id], batch=batch))
        return np.asarray(z0).flatten()
    

    def query_env_player(self, x,player, q_, locs=None, batch=1):
        '''
           if perform=true:
              return environment query with the performative effects
        '''
        z_=self.D_z(q_,locs=locs, batch=batch)
        #print(z_)
        #print(np.shape(z_))
        if player==0:
            z=z_+self.A1@x[0]+self.Ac1@x[1]
        if player==1:
            z=z_+self.A2@x[1]+self.Ac2@x[0]
        return z


def get_data_dic(filename='../data/datadic.p'):
    return pickle.load(open(filename,'rb'))


def getparams(loc_lst):
    mu=np.load('../data/mu_est.npy')
    gamma=np.load('../data/gamma_est.npy')
    who=['Lyft values','Uber values']
    A={}
    B={}
    for i in range(2):
        A[who[i]]=[]
        B[who[i]]=[]
        for j in loc_lst:
            B[who[i]].append(gamma[j][i][0,0])
            A[who[i]].append(mu[j][i][0,0])

        B[who[i]]=np.asarray(B[who[i]])
        A[who[i]]=np.asarray(A[who[i]])
    A1=np.diag(A[who[0]])
    A2=np.diag(A[who[1]])
    Ac1=np.diag(B[who[0]])
    Ac2=np.diag(B[who[1]])
    dic={}
    dic['A1']=A1
    dic['A2']=A2
    dic['Ac1']=Ac1
    dic['Ac2']=Ac2
    return dic
