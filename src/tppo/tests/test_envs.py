from tppo.utils import env_utils 

"""
List of foragax envs to test: 
- ForagaxTwoBiomeLarge-v1
- ForagaxBig-v5
- ForagaxSquareWaveTwoBiome-v11
"""
foragax_env = "ForagaxSquareWaveTwoBiome-v11"

gymnax_env = "Pendulum-v1"


if __name__ == "__main__": 
    env, env_params = env_utils.make_env(
        foragax_env, 
        aperture_size=15, 
        observation_type="color"
        )    

    # env, env_params = envs.make_env(gymnax_env)

    print(env_utils.get_env_specs(env, env_params))

