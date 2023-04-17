import jax
import numpy as np
import sys
import time

from pyRDDLGym.Planner import JaxConfigManager
from pyRDDLGym.Core.Compiler.RDDLDecompiler import RDDLDecompiler
import Examples.Arm.animation as animation
# from pyRDDLGym.Core.Jax.JaxRDDLModelError import JaxRDDLModelError

an_sizes = [(1, 1), (1, 1), (1, 1)]
shelf_sizes = [(0, 10, 0, 10), (0, 10, 0, 10), (0, 10, 0, 10)]
def print_parameterized_exprs(planner):
    model_params = planner.compiled.model_params
    print(f'model_params = {model_params}')
    ids = planner.compiled.get_ids_of_parameterized_expressions()
    for _id in ids:
        expr = planner.compiled.traced.lookup(_id)
        print(f'\nid = {_id}:\n' + RDDLDecompiler().decompile_expr(expr))
    
    
def slp_train(planner, budget, **train_args):
    step = train_args['step']
    del train_args['step']
    print('\n' + 'training plan:')
    for i, callback in enumerate(planner.optimize(**train_args, step=1)):
        if i == 0:
            elapsed = 0
            starttime = time.time()
        else:
            elapsed = time.time() - starttime
        
        if i % step == 0:
            print('[{:.4f} s] step={} train_return={:.6f} test_return={:.6f}'.format(
                elapsed,
                str(callback['iteration']).rjust(4),
                callback['train_return'],
                callback['test_return']))
        if elapsed >= budget:
            print('ran out of time!')
            break
    params = callback['best_params']
    
    # key = jax.random.PRNGKey(42)
    # error = JaxRDDLModelError(planner.rddl, planner.test_policy, 
    #                           batch_size=64, logic=planner.logic)
    # error.summarize(key, params)
    # error.sensitivity(key, params)
    return params


def slp_no_replan(env, trials, timeout, timeout_ps, save, label):

    can_sizes = [(1, 1), (1, 1), (1, 1)]
    shelf_sizes = [(0, 10, 0, 10), (0, 10, 0, 10), (0, 10, 0, 10)]

    myEnv, planner, train_args, (dom, inst) = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']
    #print(key)

    rewards = np.zeros((myEnv.horizon, trials))
    for trial in range(trials):
        #print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        train_args['key'] = key
        params = slp_train(planner, timeout, **train_args)
        
        for i in params.keys():
            print(f'{i} : {params[i]}')
        
        total_reward = 0
        state = myEnv.reset()
        for step in range(myEnv.horizon):
            #myEnv.render()

            subs = myEnv.sampler.subs
            key, subkey = jax.random.split(key)

            action = planner.get_action(subkey, params, step, subs)
            #print(action)
            #if len(action) > 1:
             #   action={list(action.keys())[0]:action[list(action.keys())[0]]}
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward 
            rewards[step, trial] = reward

            
            animation.parse_state(state, step, can_sizes, shelf_sizes)
            
            print()
            print('step       = {}'.format(step))
            print('state      = {}'.format(state))
            print('action     = {}'.format(action))
            print('next state = {}'.format(next_state))
            print('reward     = {}'.format(reward))
            
            
            state = next_state

            #if step==myEnv.horizon-1:
            #animation.parse_state(state,  label, can_sizes, shelf_sizes)
            
            if done:
                animation.parse_state(state,  label, can_sizes, shelf_sizes)
                #animation.create_video()
                break
        #print(f'episode ended with reward {total_reward}')
        
    myEnv.close()
    if save:
        np.savetxt(f'{dom}_{inst}_slp.csv', rewards, delimiter=',')
    return step


def my_planner(env, trials, timeout, timeout_ps, save):

    myEnv, planner, train_args, (dom, inst) = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']
    #print(key)

    rewards = np.zeros((myEnv.horizon, trials))
    for trial in range(trials):
        #print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        train_args['key'] = key
        params = slp_train(planner, timeout, **train_args)
        """
        for i in params.keys():
            print(f'{i} : {params[i]}')
            c = 0
            for j in params[i]:
                if params[i]==1:
                    c+=1
            print(len(params[i]), c)
        """
        total_reward = 0
        state = myEnv.reset()

        actions = []
        for step in range(myEnv.horizon):
            myEnv.render()

            subs = myEnv.sampler.subs
            key, subkey = jax.random.split(key)

            action = planner.get_action(subkey, params, step, subs)
            for a in action.keys():
                actions.append({a:action[a]})

        
        #print(actions)
        for step in range(len(actions)):
            #if len(action) > 1:
                #   action={list(action.keys())[0]:action[list(action.keys())[0]]}
            action = actions[step]
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward 
            rewards[step, trial] = reward

            #animation.parse_state(state, step, can_sizes, shelf_sizes)
            
            print()
            print('step       = {}'.format(step))
            print('state      = {}'.format(state))
            print('action     = {}'.format(action))
            print('next state = {}'.format(next_state))
            print('reward     = {}'.format(reward))
            state = next_state
            
            if done:
                #animation.parse_state(state, step + 1, can_sizes, shelf_sizes)
                #animation.create_video()
                break
            
        #print(f'episode ended with reward {total_reward}')
        myEnv.close()
        
    if save:
        np.savetxt(f'{dom}_{inst}_slp.csv', rewards, delimiter=',')

    
def slp_replan(env, trials, timeout, timeout_ps, save):
    myEnv, planner, train_args, (dom, inst) = JaxConfigManager.get(f'{env}.cfg')
    key = train_args['key']
    
    rewards = np.zeros((myEnv.horizon, trials))
    for trial in range(trials):
        print('\n' + '*' * 30 + '\n' + f'starting trial {trial + 1}\n' + '*' * 30)
        total_reward = 0
        state = myEnv.reset() 
        starttime = time.time()
        train_args['guess'] = None
        for step in range(10):
            #myEnv.render()
            currtime = time.time()
            elapsed = currtime - starttime
            
            if elapsed < timeout:
                subs = myEnv.sampler.subs
                params = slp_train(planner,
                                   budget=min(timeout - elapsed, timeout_ps),
                                   subs=subs,
                                   **train_args)
                key, subkey = jax.random.split(key)
                action = planner.get_action(subkey, params, 0, subs)
                train_args['guess'] = planner.plan.guess_next_epoch(params)
            else:
                print('ran out of time!')
                action = {}
            
            next_state, reward, done, _ = myEnv.step(action)
            total_reward += reward 
            rewards[step, trial] = reward
            
            print()
            print(f'elapsed    = {elapsed} s')
            print(f'step       = {step}')
            print(f'state      = {state}')
            print(f'action     = {action}')
            print(f'next state = {next_state}')
            print(f'reward     = {reward}')
            state = next_state
            if done: 
                break
        print(f'episode ended with reward {total_reward}')
        
    myEnv.close()
    if save:
        np.savetxt(f'{dom}_{inst}_mpc.csv', rewards, delimiter=',')


def modify_cfg_file(cfg_file_name, weight, learning_rate, epochs):
    # Read in the original file
    with open(cfg_file_name, 'r') as f:
        file_contents = f.read()

    # Replace the learning rate and epochs
    lines = file_contents.split('\n')
    for i, line in enumerate(lines):
        if "learning_rate'" in line:
            lines[i] = f"optimizer_kwargs={{'learning_rate': {learning_rate}}}"
        elif 'epochs' in line:
            lines[i] = f'epochs={epochs}'
        elif "'weight'" in line:
            lines[i] = f"logic_kwargs={{'weight': {weight}}}"
    
    # Join the lines back together and write the modified file back to disk
    file_contents = '\n'.join(lines)
    with open(cfg_file_name, 'w') as f:
        f.write(file_contents)

    
    # Join the lines back together and write the modified file back to disk
    file_contents = '\n'.join(lines)
    with open(cfg_file_name, 'w') as f:
        f.write(file_contents)

def modify_mdp_file(mdp_file_name, new_horizon):
    # Read in the original file
    with open(mdp_file_name, 'r') as f:
        file_contents = f.read()

    # Find the line that specifies the horizon and replace its value
    lines = file_contents.split('\n')
    for i, line in enumerate(lines):
        if 'horizon' in line:
            lines[i] = f'   horizon = {new_horizon};'
            break
    else:
        # If the horizon line isn't found, raise an error
        raise ValueError('Could not find "horizon" field in MDP file')

    # Join the lines back together and write the modified file back to disk
    file_contents = '\n'.join(lines)
    with open(mdp_file_name, 'w') as f:
        f.write(file_contents)



    
def main(env, replan, trials, timeout, timeout_ps, save):
    if replan:
        slp_replan(env, trials, timeout, timeout_ps, save)
    else:
        
        import csv

        weights = [0.5]
        learning_rates = [1]
        epochs = [1000]
        horizon = [100]

        
        # Open the CSV file for writing
        with open('Examples/Arm/results.csv', 'a+', newline='') as f:
            writer = csv.writer(f)

            # Write the header row
            writer.writerow(['Weight', 'Learning rate', 'Epochs', 'Horizon', 'Step'])

            # Loop over all combinations of learning rates, epochs, and horizons
            for i in weights:
                for j in learning_rates:
                    for k in epochs:
                        modify_cfg_file('Planner/Arm.cfg', i, j, k)
                        for z in horizon:
                            label = f'W:{i}, L:{j}, E:{k}, H:{z}'
                            print(label)
                            modify_mdp_file('Examples/Arm/instance0.rddl', z)
                            step = slp_no_replan(env, trials, timeout, timeout_ps, save, label)
                            # Write the results for this combination to the CSV file
                            writer.writerow([i, j, k, z, step+1])
        
        #slp_no_replan(env, trials, timeout, timeout_ps, save)


    
        
if __name__ == "__main__":
    if len(sys.argv) < 6:
<<<<<<< HEAD
        TF_CPP_MIN_LOG_LEVEL = 0
        env, trials, timeout, timeout_ps, save = 'Arm', 1, 60 * 100, 1, False
=======
        env, trials, timeout, timeout_ps, save = 'Wildfire', 1, 60 * 2, 1, False
>>>>>>> 5689cf6101383ef158c78497d6bf83b6191ea80b
    else:
        env, trials, timeout, timeout_ps, save = sys.argv[1:6]
        trials = int(trials)
        timeout = int(timeout)
        timeout_ps = int(timeout_ps)
        save = save == 'True' or save == True
    replan = env.endswith('replan')
    main(env, replan, trials, timeout, timeout_ps, save)
    
