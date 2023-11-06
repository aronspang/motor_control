import gymnasium as gym

from stable_baselines3 import A2C

env = gym.make("CartPole-v1", render_mode='human')

model = A2C("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10_000)

vec_env = model.get_env()
obs = vec_env.reset() # type: ignore
for i in range(1000):
    action, _state = model.predict(obs, deterministic=True) # type: ignore
    obs, reward, done, info = vec_env.step(action) # type: ignore
    vec_env.render() # type: ignore
    # VecEnv resets automatically
    # if done:
    #   obs = vec_env.reset()
