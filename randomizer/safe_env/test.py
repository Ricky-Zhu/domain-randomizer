import randomizer.safe_env.pendulum
import gym

env = gym.make("SafePendulum-v0")
env.reset()
for i in range(1000):
    env.render()
    s,r,d,_=env.step(env.action_space.sample())
    if d:
        env.reset()