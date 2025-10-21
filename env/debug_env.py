import jax
import jax.numpy as jnp
from jax import lax

# 0: HOLD, 1: MKT, 2: LMT
ACTION_HOLD = jnp.int32(0)

def quick_random_rollout(env, rng_key: jax.Array, num_steps: int = 16):
    """
      - random actions (type, size_frac, price_offset)
      - masks actions after done to keep scan shapes static
      - returns (final_obs, cumulative_reward, final_done, final_info) or None if env is None
    """
    if env is None:
        jax.debug.print("[debug] Env is None — skipping rollout.")
        return None

    obs, info = env.reset(seed=None, options=None)
    done0 = jnp.array(False)
    cum0 = jnp.array(0.0)

    def sample_action(key):
        k1, k2, k3 = jax.random.split(key, 3)
        a_type = jax.random.randint(k1, shape=(), minval=0, maxval=3)
        size_frac = jax.random.uniform(k2, shape=(), minval=0.0, maxval=1.0)
        price_offset = jax.random.normal(k3, shape=()) * 5.0  # ticks
        return {"type": a_type, "size_frac": size_frac, "price_offset": price_offset}

    def body(carry, _):
        rng, obs, done, cum, _info = carry
        rng, k = jax.random.split(rng)
        act = sample_action(k)

        # If already done, send HOLD w/ zero params (keeps shapes/staticness)
        act = {
            "type": jnp.where(done, ACTION_HOLD, act["type"]),
            "size_frac": jnp.where(done, 0.0, act["size_frac"]),
            "price_offset": jnp.where(done, 0.0, act["price_offset"]),
        }

        next_obs, reward, done_step, info = env.step(act)

        # Only add reward before the step that turns done=true
        reward_masked = jnp.where(done, 0.0, reward)
        cum = cum + reward_masked
        done = jnp.logical_or(done, done_step)

        return (rng, next_obs, done, cum, info), None

    (rng_f, obs_f, done_f, cum_f, info_f), _ = lax.scan(body, (rng_key, obs, done0, cum0, info), None, length=num_steps)
    return obs_f, cum_f, done_f, info_f


def main():
    rng = jax.random.PRNGKey(0)
    env = None # PLACEHOLDER

    quick_random_rollout(env=env, rng_key=rng)