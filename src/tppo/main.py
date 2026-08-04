import jax 
import gymnax 
from flax import nnx 

def main(): 
    key = jax.random.key(0) 
    key, key_reset, key_action, key_step = jax.random.split(key, 4)

    env, env_params = gymnax.make(env_id="Pendulum-v1")
    # print(env, env_params)
    # <gymnax.environments.classic_control.pendulum.Pendulum object at 0x119827110> 
    # EnvParams(max_steps_in_episode=200, max_speed=8.0, max_torque=2.0, dt=0.05, g=10.0, m=1.0, l=1.0)

    obs, state = env.reset(key_reset, env_params)
    # print(obs, state)
    # [-0.99895006 -0.04581222 -0.9582176 ] 
    # EnvState(time=Array(0, dtype=int32, weak_type=True), theta=Array(-3.0957644, dtype=float32), theta_dot=Array(-0.9582176, dtype=float32), last_u=Array(0., dtype=float32, weak_type=True))

    action = env.action_space(env_params).sample(key_action)
    # print(action)
    # [1.609798]

    next_obs, next_state, reward, done, _ = env.step(key_step, state, action, env_params)


if __name__ == "__main__": 
    main() 