import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
import optax
import time
from flax.linen.initializers import constant, orthogonal
from typing import Sequence, NamedTuple, Any, Dict
from flax.training.train_state import TrainState
import distrax
import gymnax
import functools
from gymnax.environments import spaces
import sys
import chex
import jax.tree_util
if not hasattr(jax, 'tree'):
    jax.tree = jax.tree_util
jax.tree_map = jax.tree.map
sys.path.append('../purejaxrl')
sys.path.append('../AlphaTrade')

import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'AlphaTrade'))
from gymnax.environments import spaces

# from purejaxrl.wrappers import FlattenObservationWrapper, LogWrapper,ClipAction, VecEnv,NormalizeVecObservation,NormalizeVecReward
from gymnax_exchange.jaxen.exec_env import ExecutionEnv

#Code snippet to disable all jitting.
from jax import config
# config.update("jax_disable_jit", False)
#config.update("jax_disable_jit", True)

config.update("jax_check_tracer_leaks",False) #finds a whole assortment of leaks if true... bizarre.

"""
Optional Weights & Biases logging: if WANDB_API_KEY is provided in the
environment, we will login and enable wandb logging in addition to console prints.
"""
_WANDB_KEY = os.environ.get("WANDB_API_KEY")
wandb = None
if _WANDB_KEY:
    try:
        import wandb as _wandb
        _wandb.login(key=_WANDB_KEY)
        wandb = _wandb
    except Exception:
        wandb = None

#instead importing LogWrapper from purejaxrl, we define it here. purejaxrl is not installed in the environment.
class LogWrapper:
    def __init__(self, env):
        self.env = env
    def __getattr__(self, name):
        return getattr(self.env, name)
    def step(self, rng, state, action, params):
        obs, state, reward, done, info = self.env.step(rng, state, action, params)
        if info is None:
            info = {}
        info["timestep"] = getattr(state, "timestep", 0)
        info["returned_episode"] = done
        info["returned_episode_returns"] = reward
        return obs, state, reward, done, info

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

def make_train(config):
    config["NUM_UPDATES"] = (
        config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    env= ExecutionEnv(config["ATFOLDER"],config["TASKSIDE"],config["DEBUG"], config["ACTION_TYPE"])
    env_params = env.default_params
    env = LogWrapper(env)
    
    #FIXME : Uncomment normalisation.
    #if config["NORMALIZE_ENV"]:
         #env = NormalizeVecObservation(env)
         #env = NormalizeVecReward(env, config["GAMMA"])

    def train(rng):
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        # jax.debug.breakpoint()
        obsv, env_state = jax.vmap(env.reset, in_axes=(0, None))(reset_rng, env_params)

        # jax.debug.breakpoint()
        # TRAIN LOOP
        def _update_step(runner_state, unused):
            # COLLECT TRAJECTORIES
            def _env_step(runner_state, unused):
                env_state, last_obs, last_done, rng = runner_state
                rng, _rng = jax.random.split(rng)

                rng_action=jax.random.split(_rng, config["NUM_ENVS"])
                action = jax.vmap(env.action_space().sample, in_axes=(0))(rng_action)
                
                # STEP ENV
                rng, _rng = jax.random.split(rng)
                rng_step = jax.random.split(_rng, config["NUM_ENVS"])
                obsv_step, env_state_step, reward_step, done_step, info_step = jax.vmap(
                    env.step, in_axes=(0, 0, 0, None)
                )(rng_step, env_state, action, env_params)

                transition = Transition(
                    done_step, action, reward_step, last_obs, info_step
                )
                runner_state = (env_state_step, obsv_step, done_step, rng)
                return runner_state,transition

            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, None, config["NUM_STEPS"]
            )
            env_state, last_obs, last_done, rng = runner_state

            # CALCULATE ADVANTAGE
            metric = traj_batch.info

            # UPDATE NETWORK
            if config.get("DEBUG"):

                def callback(info):
                    return_values = info["returned_episode_returns"][
                        info["returned_episode"]
                    ]
                    timesteps = (
                        info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    for t in range(len(timesteps)):
                        print(
                            f"global step={timesteps[t]}, episodic return={return_values[t]}"
                        )
                        if wandb is not None:
                            try:
                                wandb.log({
                                    "global_step": int(timesteps[t]),
                                    "episodic_return": float(return_values[t]),
                                })
                            except Exception:
                                pass

                jax.debug.callback(callback, metric)

            runner_state = (env_state, last_obs, last_done, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metric": metric}

    return train


if __name__ == "__main__":
    import datetime
    import json
    
    try:
        ATFolder = sys.argv[1] 
    except:
        ATFolder = '/home/duser'
    print("AlphaTrade folder:",ATFolder)

    # Setup experiment directory structure
    experiments_dir = "/home/duser/experiments"
    os.makedirs(experiments_dir, exist_ok=True)
    
    # Generate experiment name: exp_<number>_<timestamp>
    existing_exps = [d for d in os.listdir(experiments_dir) if d.startswith("exp_") and os.path.isdir(os.path.join(experiments_dir, d))]
    if existing_exps:
        exp_nums = []
        for exp in existing_exps:
            try:
                num = int(exp.split("_")[1])
                exp_nums.append(num)
            except (ValueError, IndexError):
                pass
        next_exp_num = max(exp_nums) + 1 if exp_nums else 1
    else:
        next_exp_num = 1
    
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = f"exp_{next_exp_num}_{timestamp_str}"
    exp_dir = os.path.join(experiments_dir, exp_name)
    os.makedirs(exp_dir, exist_ok=True)

    config = {
        "NUM_ENVS": 10,
        "NUM_STEPS": 10,
        "TOTAL_TIMESTEPS": 1000,
        "DEBUG": True,
        "ATFOLDER": ATFolder,
        "TASKSIDE":'buy',
        "ACTION_TYPE":"pure"
    }

    # Save config to experiment directory
    config_save_path = os.path.join(exp_dir, "config.json")
    with open(config_save_path, 'w') as f:
        json.dump(config, f, indent=2, default=str)
    
    print(f"Experiment: {exp_name}")
    print(f"Experiment directory: {exp_dir}")

    # Initialize wandb run if available
    if wandb is not None:
        try:
            run = wandb.init(project=os.environ.get("WANDB_PROJECT", "AlphaTradeJAX_Train"), config=config, save_code=False)
        except Exception:
            run = None
    else:
        run = None

    rng = jax.random.PRNGKey(30)
    # jax.debug.breakpoint()
    train_jit = jax.jit(make_train(config))

    if config["DEBUG"]:
        pass
        #chexify the function
        #NOTE: use chex.asserts inside the code, under a if DEBUG. 

    # train = make_train(ppo_config)
    # jax.debug.breakpoint()
    start=time.time()
    out = train_jit(rng)
    print("Time: ", time.time()-start)

    if run is not None:
        try:
            run.finish()
        except Exception:
            pass

