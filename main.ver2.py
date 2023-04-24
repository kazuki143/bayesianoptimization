# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os, time, settings_, warnings
from bayes_opt import BayesianOptimization, UtilityFunction
from bayes_opt.logger import JSONLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs
from sklearn.gaussian_process.kernels import Matern, RBF
from utils_ import *
warnings.simplefilter('ignore')


def main(mode, work_name, space, space_raw, acquisition, kappa, xi, kernel, length_scale, nu, 
         E_exp=None, init_file=None, n_trial=100, n_trial_P=20, random_state=0, 
         results_folder=None, repeat=None):

    # Make a function to be optimized
    start_time = time.time()
    
    func = BlackBoxFunc(random_state, E_exp)  ### 仮想的な関数の生成
    
    ################################ Simulation mode ###################################
    if mode == 'simulation':
        for number in range(repeat):
            # func = BlackBoxFunc(random_state, E_exp)  ### 仮想的な関数の生成
            maxY, minY = func.max_minY(space, space_raw, E_exp)
            print(f'maxY: {maxY}')
            print(f'minY: {minY}')

            # Optimizer
            optimizer = BayesianOptimization(
                f = func.black_box_function,
                pbounds = space,
                verbose = 2,
                random_state = random_state,
                allow_duplicate_points=True
            )
            if kernel == 'matern':
                optimizer._gp.kernel = Matern(length_scale=length_scale, nu=nu)
            elif kernel == 'rbf':
                optimizer._gp.kernel = RBF(length_scale=length_scale)

            # Saving progress
            utility = UtilityFunction(kind=acquisition, kappa=kappa, xi=xi)

            # Set logfile for save
            # 名前にmax値とmin値を入れました！！
            # save_name = f'{work_name}{number+1}_{acquisition}-k{kappa}-xi{xi}_{kernel}-l{length_scale}-nu{nu}-repeat{repeat}-n_trial{n_trial}-maxY{maxY:.3f}-minY{minY:.3f}'
            save_name = f'{work_name}{number+1}_{acquisition}-k{kappa}-xi{xi}_{kernel}-l{length_scale}-nu{nu}-repeat{repeat}-n_trial{n_trial}'
            os.makedirs(results_folder, exist_ok=True)
            logger = JSONLogger(path=f'{results_folder}{save_name}.json')
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            
            #最大値と最小値、muの保存
            func.save_max_min_mu(maxY, minY, results_folder, save_name)

            # Initial samples
            n_init_points = 5
            print('Initial samples')
            for i in range(n_init_points):
                init_point = func.initsampling(optimizer.space.bounds)
                init_point = {key: num for key, num in zip(optimizer.space.keys, init_point)}
                init_point = func.discretize(**init_point)
                init_point = {key: num for key, num in zip(optimizer.space.keys, init_point)}
                target = func.black_box_function(**init_point, space_raw=space_raw, E_exp=E_exp)
                print(i+1, init_point, target)
                optimizer.register(
                    params = init_point,
                    target = target,
                )
        
            # Optimization loop for E, L, W, T, P
            for i in range(n_trial):
                print(f'Sample {i+1} start')
                optimizer.set_bounds(new_bounds = space)
                # optimizer.set_bounds(new_bounds = space_new)

                # Sugget a new point & discretize
                next_point_to_probe = optimizer.suggest(utility)
                next_point_to_probe = func.discretize(**next_point_to_probe)
                next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                # Do assumed experiment at the new point
                target = func.black_box_function(**next_point_to_probe, space_raw=space_raw, E_exp=E_exp)
                # Save the experimented point
                optimizer.register(
                    params = next_point_to_probe,
                    target = target
                )

                # Optimization loop for P
                optimizer.set_bounds(new_bounds = {
                    'E': (next_point_to_probe['E'], next_point_to_probe['E']),
                    'L': (next_point_to_probe['L'], next_point_to_probe['L']),
                    'W': (next_point_to_probe['W'], next_point_to_probe['W']),
                    'T': (next_point_to_probe['T'], next_point_to_probe['T']),
                })

                # Sugget a new point & discretize
                next_point_to_probe = optimizer.suggest(utility)
                next_point_to_probe = func.discretize(**next_point_to_probe)
                next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                # Save figure
                optimizer2fig(optimizer, utility, 0, space_raw, random_state, E_exp, save_name, func.black_box_function, next_point_to_probe)
                for j in range(n_trial_P):
                    # Exit from loop for P
                    if (optimizer.local_obs_max - optimizer.local_min) >= (optimizer.local_max - optimizer.local_min)*0.9:
                        print(f'---Local maxima was found at {len(optimizer.res)}---')
                        break

                    # Do assumed experiment at the new point
                    target = func.black_box_function(**next_point_to_probe, space_raw=space_raw, E_exp=E_exp)

                    # Save the experimented point
                    optimizer.register(
                        params = next_point_to_probe,
                        target = target
                    )

                    # Sugget a new point & discretize
                    next_point_to_probe = optimizer.suggest(utility)
                    next_point_to_probe = func.discretize(**next_point_to_probe)
                    next_point_to_probe = {key: num for key, num in zip(optimizer.space.keys, next_point_to_probe)}

                    # Save figure
                    optimizer2fig(
                        optimizer, utility, j+1, space_raw, random_state, E_exp, save_name, func.black_box_function, next_point_to_probe
                    )

                # Exit from E, L, W, T, P
                # ループを抜けるタイミングをチェックしました
                if (optimizer.max['target']-minY) >= (maxY-minY)*0.9:
                    print(f'---Global maxima was found at {len(optimizer.res)}---')
                    break

                #正規化をもとに戻す
                func.save_non_normalized_dataset(results_folder, save_name, space_raw,E_exp)
    
    ################################ Experiment mode ###################################
    elif mode == 'experiment':
        # Optimizer & dummy object
        optimizer = BayesianOptimization(
            f = func.discreteGP,
            pbounds = space,
            verbose = 2,
            random_state = 1
        )
        
        dummy = BayesianOptimization(
            f = func.discreteGP,
            pbounds = space,
            verbose = 2,
            random_state = 1
        )
        
        # Saving progress
        utility = UtilityFunction(kind=acquisition, kappa=kappa, xi=xi)
        
        # Make results folder
        os.makedirs(results_folder, exist_ok=True)

        # Set logfile for experiment points
        logfile = f'{results_folder}log_{work_name}_exp.json'
        if os.path.exists(logfile):
            load_logs(optimizer, logs=[logfile])
            print("Logfile(exp) includes {} points.".format(len(optimizer.space)))
        else:
            logger = JSONLogger(path=logfile)
            optimizer.subscribe(Events.OPTIMIZATION_STEP, logger)
            # Initial samples
            df = pd.read_excel(init_file, index_col=0)
            dict_in_list = df.to_dict('records')
            for i in range(len(dict_in_list)):
                init_point = {'E': dict_in_list[i]['E'],'L': dict_in_list[i]['L'],'P': dict_in_list[i]['P'],'T': dict_in_list[i]['T'],'W': dict_in_list[i]['W']}
                # init_point = dict_in_list[i][optimizer.space.keys]  ### 適切な説明変数のカラム名に修正する必要がある
                target = dict_in_list[i]['target']  ### 適切な目的変数のカラム名に修正する必要がある
                optimizer.register(
                    params = init_point,
                    target = target,
                )
    
        # Set logfile for suggested points
        logfile = f'{results_folder}log_{work_name}_suggest.json'
        if os.path.exists(logfile):
            load_logs(dummy, logs=[logfile])
            print("Logfile(suggest) includes {} points.".format(len(dummy.space)))
        else:
            logger = JSONLogger(path=logfile)
            dummy.subscribe(Events.OPTIMIZATION_STEP, logger)

        # Definition of save_name
        save_name = f'{work_name}_{acquisition}-k{kappa}-xi{xi}_{kernel}-l{length_scale}-nu{nu}-n_trial{n_trial}'

        # Optimization loop for E, L, W, T, P
        for i in range(n_trial):
            if i >= 1:
                optimizer.set_bounds(new_bounds = space)
                
            # Sugget a new point
            next_point_to_probe = optimizer.suggest(utility)
            next_point_to_probe = func.discretize(**next_point_to_probe)
            items = [item for item in next_point_to_probe]
            mu = optimizer._gp.predict(np.array(items).reshape(1,-1))
            mu = mu.item()
            print(f'Suggested next point: {next_point_to_probe}'\
                  f'Predicted value: {mu}')
            
            # Save the suggested point
            dummy.register(
                params=next_point_to_probe,
                target=mu,
            )

            # Do REAL experiment near the new point (IMPORTANT!!!)
            print('-----------------')
            print('Please choose a sample close to the suggested point')
            new_exp_point = handinput()
            new_exp_point = {key: num for key, num in zip(optimizer.space.keys, new_exp_point)}
            print('Please measure and specify the result file')
            target = getmaxY()

            # Save the experimented point
            optimizer.register(
                params = new_exp_point,
                target = target,
            )

            # Optimization loop for P
            optimizer.set_bounds(new_bounds = {
                'E': (new_exp_point['E'], new_exp_point['E']),
                'L': (new_exp_point['L'], new_exp_point['L']),
                'W': (new_exp_point['W'], new_exp_point['W']),
                'T': (new_exp_point['T'], new_exp_point['T']),
            })

            # # Save figure
            # optimizer2fig(optimizer, utility, 0, save_name, next_point_to_probe)

            for j in range(n_trial_P):
                # Sugget a new point
                next_point_to_probe = optimizer.suggest(utility)
                next_point_to_probe = func.discretize(**next_point_to_probe)
                items = [item for item in next_point_to_probe]
                mu = optimizer._gp.predict(np.array(items).reshape(1,-1))
                mu = mu.item()
                print(f'Suggested next point: {next_point_to_probe}'\
                      f'Predicted value: {mu}')
                # Save the suggested point
                dummy.register(
                    params=next_point_to_probe,
                    target=mu,
                )

                # Do REAL experiment near the new point (IMPORTANT!!!)
                power = next_point_to_probe[2]
                print(f'Please confirm the UP power is {power}%. (y or Y)')
                confirm = input('>>')
                if confirm == 'y' or confirm == 'Y':
                    print('Please measure and specify the result file')
                    target = getmaxY()
                ### P最適化ループを抜ける処理(一時的) ###
                elif confirm=='out':
                    break

                # Save the experimented point
                optimizer.register(
                    params = next_point_to_probe,
                    target = target
                )

                # # Save figure
                # optimizer2fig(optimizer, utility, j+1, save_name, next_point_to_probe)
            ### E, L, W, T, P最適化ループを抜ける処理 ###

    else:
        print('Choose valid mode!')
    
    # Finishing
    total = time.time() - start_time
    minute = total//60
    second = total%60
    print(f'--- {int(minute)} min {int(second)} sec elapsed ---')
    

if __name__ == '__main__':
    params = {
        'mode': settings_.mode,
        'work_name': settings_.work_name,
        'init_file': settings_.init_file,
        'n_trial': settings_.n_trial,
        'n_trial_P': settings_.n_trial_P,
        'random_state': settings_.random_state,
        'acquisition': settings_.acquisition,
        'results_folder': settings_.results_folder,
        'space': settings_.space,
        'space_raw': settings_.space_raw,
        'E_exp': settings_.E_exp,
        'kappa': settings_.kappa,
        'xi': settings_.xi,
        'kernel': settings_.kernel,
        'length_scale': settings_.length_scale,
        'nu': settings_.nu,
        'repeat': settings_.repeat
    }
    main(**params)
