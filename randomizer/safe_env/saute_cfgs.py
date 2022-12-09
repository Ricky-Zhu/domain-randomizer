double_pendulum_cfg = dict(
    unsafe_reward=-200.,
    saute_discount_factor=1.0,
    max_ep_len=200,
    min_rel_budget=1.0,
    max_rel_budget=1.0,
    test_rel_budget=1.0,
    use_reward_shaping=True,
    use_state_augmentation=True

)

safe_fetch_slide_cfg = dict(
    unsafe_reward=-10.,
    safety_budget=0.1,
    saute_discount_factor=1.0,
    min_rel_budget=1.0,
    max_rel_budget=1.0,
    test_rel_budget=1.0,
    use_reward_shaping=True,
    use_state_augmentation=True

)
