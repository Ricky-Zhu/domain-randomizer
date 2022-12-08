from gym.envs.registration import register
import os.path as osp
from randomizer.safe_env.pendulum.safe_pendulum import pendulum_cfg, SafePendulumEnv, SautedPendulumEnv
from randomizer.safe_env.double_pendulum.safe_double_pendulum import SafeDoublePendulumEnv, \
    RandomizeSafeDoublePendulumEnv
from randomizer.safe_env.safe_fetch_slide.safe_fetch_slide import SafeFetchSlideEnv
from randomizer.safe_env.saute_cfgs import double_pendulum_cfg

print('LOADING SAFE ENVIROMENTS')

register(
    id='SafePendulum-v0',
    entry_point='randomizer.safe_env.pendulum.safe_pendulum:SafePendulumEnv',
    max_episode_steps=pendulum_cfg['max_ep_len'],
    # max_episode_steps=200,
)

register(
    id='SafeDoublePendulum-v0',
    entry_point='randomizer.safe_env.double_pendulum.safe_double_pendulum:SafeDoublePendulumEnv',
    max_episode_steps=double_pendulum_cfg['max_ep_len'],
    # max_episode_steps=200,
)

register(
    id='RandomizeSafeDoublePendulum-v0',
    entry_point='randomizer.safe_env.double_pendulum.safe_double_pendulum:RandomizeSafeDoublePendulumEnv',
    max_episode_steps=200,
)


register(
    id='SafeFetchSlide-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.safe_fetch_slide:SafeFetchSlideEnv',
    max_episode_steps=50,
)

register(
    id='SafeFetchSlideWithCostFn-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost:SafeFetchSlideWithCost',
    max_episode_steps=150,
)

register(
    id='RandomizeSafeFetchSlideCostEnv-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost:RandomizeSafeFetchSlideCostEnv',
    max_episode_steps=150,
)


############### Saute env for fetch slide with 4 danger regions and 1 danger region (simple) #####
register(
    id='SauteRandomizeSafeFetchSlide-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost:SauteRandomizableFetchSlide',
    max_episode_steps=150,
)

register(
    id='SauteRandomizeSafeFetchSlideSimple-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost_simple:SauteRandomizableFetchSlideSimple',
    max_episode_steps=100,
)

register(
    id='SauteRandomizeSafeDoublePendulum-v0',
    entry_point='randomizer.safe_env.double_pendulum.safe_double_pendulum:SautedRandomizableDoublePendulumEnv',
    max_episode_steps=200,
)
#########################################################################################################

register(
    id='SafeFetchSlideWithCostFnSimple-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost_simple:SafeFetchSlideWithCostSimple',
    max_episode_steps=100,
)

register(
    id='RandomizeSafeFetchSlideCostSimpleEnv-v0',
    entry_point='randomizer.safe_env.safe_fetch_slide.randomized_safe_fetch_slide_with_cost_simple:RandomizeSafeFetchSlideCostSimpleEnv',
    max_episode_steps=100,
)
