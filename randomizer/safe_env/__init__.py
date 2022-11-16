from randomizer.safe_env.pendulum.safe_pendulum import pendulum_cfg, SafePendulumEnv, SautedPendulumEnv
from randomizer.safe_env.double_pendulum.safe_double_pendulum import double_pendulum_cfg, SafeDoublePendulumEnv, \
    RandomizeSafeDoublePendulumEnv
from gym.envs import register
from randomizer.safe_env.safe_fetch_slide.safe_fetch_slide import SafeFetchSlideEnv

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafePendulum-v0',
    entry_point='randomizer.safe_env.pendulum.safe_pendulum:SafePendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len']
)

register(
    id='SafeDoublePendulum-v0',
    entry_point='randomizer.safe_env.double_pendulum.safe_double_pendulum:SafeDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len']
)

register(
    id='RandomizeSafeDoublePendulum-v0',
    entry_point='randomizer.safe_env.double_pendulum.safe_double_pendulum:RandomizeSafeDoublePendulumEnv',
    max_episode_steps=200
)
register(
    id='SafeFetchSlide-v0',
    entry_point='randomizer.safe_env:SafeFetchSlideEnv',
    max_episode_steps=50
)