import os, json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bayes_opt import BayesianOptimization
from bayes_opt import UtilityFunction
from matplotlib import gridspec
plt.rcParams['font.family'] = 'Arial'


class BlackBoxFunc:
    def __init__(self, random_state, E_exp):
        self.random_state = random_state
        self.E_exp_normalized = (np.array(E_exp)-min(E_exp))/(max(E_exp)-min(E_exp))
        # self.mu_E = np.random.random()
        # self.mu_L = np.random.random()
        # self.mu_W = np.random.random()
        # self.mu_T = np.random.random()
        # self.mu_P = np.random.random()    
        # self.E_exp = E_exp
        self.mu_E = np.random.randint(0, 13, 1).item()
        self.mu_L = np.random.randint(1000, 3000, 1).item()
        self.mu_W = np.random.randint(200, 1000, 1).item()
        self.mu_T = np.random.randint(50, 200, 1).item()
        self.mu_P = np.random.randint(0, 100, 1).item()
        #↓変えました！！（分散が小さいから最大値に達しにくいのかも）
        # var_list = [25, 50, 100, 1000]
        var_list = [50, 100, 500, 1000]
        self.var_E = 500 # np.random.choice(var_list)
        self.var_L = 500 # np.random.choice(var_list)
        self.var_W = 500 # np.random.choice(var_list)
        self.var_T = 500 # np.random.choice(var_list)
        self.var_P = 500 # np.random.choice(var_list)

        # self.var_E = 0.2 # np.random.choice(var_list)
        # self.var_L = 0.2 # np.random.choice(var_list)
        # self.var_W = 0.2 # np.random.choice(var_list)
        # self.var_T = 0.2 # np.random.choice(var_list)
        # self.var_P = 0.2 # np.random.choice(var_list)
        coef_list = [3] #[0.5, 1, 3]
        self.coef_E = np.random.choice(coef_list)
        self.coef_L = np.random.choice(coef_list)
        self.coef_W = np.random.choice(coef_list)
        self.coef_T = np.random.choice(coef_list)
        self.coef_P = np.random.choice(coef_list)
        
    def discretize(self, E, L, P, T, W):
        E = self.getNearestValue(E)
        L = L
        W = W
        T = T
        P = P
        return E, L, P, T, W
    
    #平均と分散は正規化前のものを使っている
    def black_box_function(self, E, L, P, T, W, space_raw, E_exp):
        y = self.coef_E*np.exp(-1*(E*(max(E_exp) - min(E_exp)) + min(E_exp) - self.mu_E)**2/(2*self.var_E))\
            + self.coef_L*np.exp(-1*((L*(space_raw['L'][1]-space_raw['L'][0])+space_raw['L'][0]) - self.mu_L)**2/(2*self.var_L))\
            + self.coef_P*np.exp(-1*((P*(space_raw['P'][1]-space_raw['P'][0])+space_raw['P'][0]) - self.mu_P)**2/(2*self.var_P))\
            + self.coef_T*np.exp(-1*((T*(space_raw['T'][1]-space_raw['T'][0])+space_raw['T'][0]) - self.mu_T)**2/(2*self.var_T)) \
            + self.coef_W*np.exp(-1*((W*(space_raw['W'][1]-space_raw['W'][0])+space_raw['W'][0]) - self.mu_W)**2/(2*self.var_W))
        return y

    def max_minY(self, space, space_raw, E_exp):
        n = 30
        var = np.zeros(n*len(space)).reshape(-1, len(space))
        for i, key in enumerate(space):
            var[:, i] = np.linspace(space[key][0], space[key][1], n)
        E, L, P, T, W = np.meshgrid(self.E_exp_normalized, var[:,1], var[:,2], var[:,3], var[:,4])
        y = self.black_box_function(E, L, P, T, W, space_raw, E_exp)
        return y.max(), y.min()

    #一旦コメントアウト。実験の時に使用！    
    # def discreteGP(self, E, L, P, T, W):
    #     assert E in self.E_exp
    #     assert type(L) == int
    #     assert type(W) == int
    #     assert type(T) == int
    #     assert type(P) == int
    #     assert P >= 0 and P <= 100
    #     mu = optimizer._gp.predict(E, L, P, T, W)
    #     return mu
    
    def getNearestValue(self, num):
        idx = np.abs(np.asarray(self.E_exp_normalized) - num).argmin()
        return self.E_exp_normalized[idx]
    
    def initsampling(self, space):
        x_rand = []
        for i in range(len(space)):
            rand = np.random.random()
            x_rand.append(rand)
        return x_rand
    
    # def initsampling(self, space):
    #     x_rand = []
    #     for i in range(len(space)):
    #         rand = np.random.randint(space[i][0], space[i][1], 1)
    #         x_rand.append(rand.item())
    #     return x_rand
    
    def save_max_min_mu(self, maxY, minY, results_folder, save_name):
        save_dic={"max":round(maxY,3),"min":round(minY,3),"mu_E":self.mu_E,"mu_L":self.mu_L,"mu_W":self.mu_W,"mu_T":self.mu_T,"mu_P":self.mu_P}
        path_save_dic = f'{results_folder}/{save_name}_max_min_mu.json'
        json_save_dic = open(path_save_dic, mode="w")
        json.dump(save_dic, json_save_dic)
        json_save_dic.close()

    def save_non_normalized_dataset(self, results_folder, save_name, space_raw,E_exp):
        json_file=open(f'{results_folder}/{save_name}.json','r')
        df=pd.read_json(json_file,orient='records', lines=True)
        for i in range(len(df)):
            E_ordinal=df['params'][i]['E']*(max(E_exp)-min(E_exp))+min(E_exp)
            df['params'][i]['E']=E_ordinal
            df['params'][i]['L']=round(df['params'][i]['L']*(space_raw['L'][1]-space_raw['L'][0])+space_raw['L'][0])
            df['params'][i]['P']=round(df['params'][i]['P']*(space_raw['P'][1]-space_raw['P'][0])+space_raw['P'][0])
            df['params'][i]['W']=round(df['params'][i]['W']*(space_raw['W'][1]-space_raw['W'][0])+space_raw['W'][0])
            df['params'][i]['T']=round(df['params'][i]['T']*(space_raw['T'][1]-space_raw['T'][0])+space_raw['T'][0])
        df.to_json(f'{results_folder}/{save_name}_non_normalized.json',orient='records',lines=True)
        
def handinput():
    print('Type Youngs modulus (GPa)')
    E = input('>>')
    print('Type crystal length (μm)')
    L = input('>>')
    print('Type UV intensity (%)')
    P = input('>>')
    print('Type crystal width (μm)')
    W = input('>>')
    print('Type crystal thickness (μm)')
    T = input('>>')
    return float(E), int(L), int(P), int(W), int(T)
    
    
# Raw file (from tact) -> a value
def getmaxY():
    try:
        file = input('>>')
        df = pd.read_excel(file, engine='openpyxl')
        for i in range(len(df)):
            if df.iloc[i+1,1]>=0.2:
                save=i
                break
        maxY = df.iloc[(save+1):len(df),1].max()-df.iloc[i+1,1]
    except:
        print('Max_Y was not found. Please input manually.')
        maxY = input('>>')
        maxY = float(maxY)
    return maxY


def posterior(optimizer, x_obs, y_obs, grid):
    # optimizer._gp.fit(x_obs, y_obs)
    mu, sigma = optimizer._gp.predict(grid, return_std=True)
    return mu, sigma


def plot_gp(optimizer, utility, number, i, key, random_state, space_raw, E_exp, save_folder, function=None, next_candidate=None):
# def plot_gp(optimizer, utility, number, i, key, save_folder, func, space_raw, E_exp, next_candidate=None):
    steps = len(optimizer.space)
    x_range = optimizer.space.bounds[i]
    n_keys = len(optimizer.space.keys)
    n_lin = 1000
    
    x_all = np.zeros(int(n_keys * n_lin)).reshape(-1, n_keys)
    for k in range(n_keys):
        if k == i:
            x = np.linspace(x_range[0], x_range[1], n_lin)
            x_vis = x.reshape(-1, 1)
        else:
            x = np.ones(n_lin)
            x = x*optimizer.res[-1]['params'][optimizer.space.keys[k]]
        x_all[:, k] = x

    # bbf=BlackBoxFunc(random_state, E_exp)
    # y = bbf.black_box_function(E=x_all[:,0], L=x_all[:,1], P=x_all[:,2], T=x_all[:,3], W=x_all[:,4], space_raw=space_raw, E_exp=E_exp)
    if function is not None:
        y = function(x_all[:,0], x_all[:,1], x_all[:,2], x_all[:,3], x_all[:,4], space_raw, E_exp)
    else:
        y = None

    # Observed points
    y_obs_all = np.array([res['target'] for res in optimizer.res])
    y_obs = y_obs_all[-(number + 1):]
    x_obs_all = np.zeros(int(n_keys * steps)).reshape(-1, n_keys)    
    for k in range(n_keys):
        x = np.array([res['params'][optimizer.space.keys[k]] for res in optimizer.res])
        if k == i:
            x_obs_all_vis = x
        x_obs_all[:, k] = x
    x_obs = x_obs_all_vis[-(number + 1):]
    optimizer.local_max = y.max()
    optimizer.local_min = y.min()
    optimizer.local_obs_max = y_obs.max()
    
    # Set figure
    fig = plt.figure(figsize=(8, 5), tight_layout=True)
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    axis = plt.subplot(gs[0])
    acq = plt.subplot(gs[1])

    # Plot setting for observed points & GP
    mu, sigma = posterior(optimizer, x_obs_all, y_obs_all, x_all)
    if y is not None:
        axis.plot(x_vis, y, linewidth=3, label='Target')
    axis.plot(x_obs.flatten(), y_obs, 'D', markersize=8, label='Observations', color='r')
    axis.plot(x_vis, mu, '--', color='k', label='Prediction')
    axis.fill(np.concatenate([x_vis, x_vis[::-1]]), 
              np.concatenate([mu - 1.9600 * sigma, (mu + 1.9600 * sigma)[::-1]]),
              alpha=.6, fc='c', ec='None', label='95% CI')
    axis.set_xlim((x_range[0], x_range[1]))
    axis.set_ylim((0, 15))
    axis.set_yticks(list(np.arange(0, 15, 2.5)))
    axis.set_ylabel('Target', fontdict={'size':15})
    axis.set_xlabel(key, fontdict={'size':15})

    # Plot setting for acquision function
    utility_res = utility.utility(x_all, optimizer._gp, 0)
    acq.plot(x_vis, utility_res, label='Acquisition Func.', color='purple')
    acq.plot(next_candidate[key], np.max(utility_res), '*', markersize=15, 
             label='Next Candidate', markerfacecolor='gold', markeredgecolor='k', 
             markeredgewidth=1)
    acq.set_xlim((x_range[0], x_range[1]))
    acq.set_ylim((0, 25))
    acq.set_yticks(list(np.arange(0, 25, 5)))
    acq.set_ylabel('Acquisition', fontdict={'size':15})
    acq.set_xlabel(key, fontdict={'size':15})

    axis.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    acq.legend(loc=2, bbox_to_anchor=(1.01, 1), borderaxespad=0.)
    os.makedirs(f'results/{save_folder}/', exist_ok=True)
    fig.savefig(f'results/{save_folder}/{key}_{steps}_{number}.png', dpi=150)

    
def optimizer2fig(optimizer, utility, number, space_raw, random_state, E_exp, save_folder=None, function=None, next_candidate=None):
    for i, key in enumerate(optimizer.space.keys):
        if key == 'P':
            plot_gp(optimizer, utility, number, i, key, random_state, space_raw, E_exp, save_folder, function, next_candidate)
    
            
def readjson(path):
    res = []
    decoder = json.JSONDecoder()
    with open(path, 'r') as f:
        line = f.readline()
        while line:
            res.append(decoder.raw_decode(line))
            line = f.readline()
    return res