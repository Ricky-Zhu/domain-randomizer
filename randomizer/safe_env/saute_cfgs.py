double_pendulum_cfg = dict(
    action_dim=1,
    action_range=[
        -1,
        1],
    unsafe_reward=-200.,
    saute_discount_factor=1.0,
    max_ep_len=200,
    min_rel_budget=1.0,
    max_rel_budget=1.0,
    test_rel_budget=1.0,
    use_reward_shaping=True,
    use_state_augmentation=True

)

