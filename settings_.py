# Run arguments 
mode = 'simulation' # {'experiment', 'simulation'}
work_name = '20230412_simulation'
repeat = 50
n_trial = 1000
n_trial_P = 20
random_state = 1
init_file = None
results_folder = 'results/'
# -------------------------
# Acquisiiton Function
acquisition = 'ucb'  # {'ucb', 'ei', 'poi'}
kappa =  10          # UCB {1, 2.5, 5, 10, 20}
xi = 0.0             # EI & POI
# -------------------------
# GRP kernel
kernel = 'rbf'    # {'rbf', 'matern'}
length_scale = 1   # rfb, matern {0.1, 1, 10, 100}
nu = 1.5             # matern {1, 1.5, 2, 2.5, 3} 
# -------------------------
# Search space
space = {'E': (0, 1),
         'L': (0, 1),
         'P': (0, 1),
         'T': (0, 1),
         'W': (0, 1)}

E_exp = [2, 3.1, 3.7, 4.3, 4.7, 6, 6.2, 7.3, 7.6, 8.5, 12.9]

space_raw = {'E': (0, 13),
             'L': (1000, 3000),
             'P': (0, 100),
             'T': (50, 200),
             'W': (200, 1000)}
# -------------------------

# 課題メモ (Solved!)
# Sample 1の最初の数回がsuggest条件とexp条件が合わないことが多い
# Sample 2以降でもsuggest条件とexp条件が合わないことたまにある
# kernelにホワイトノイズを加えるのはどうか