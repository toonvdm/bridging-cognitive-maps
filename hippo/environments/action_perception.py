from tqdm import trange
import numpy as np


def action_perception_loop(
    env,
    agent,
    steps=100,
    stop_when_done=True,
    record_frames=True,
    record_agent_info=False,
    progress_bar=True,
    observable_reward=True,
    callbacks=None,
    observable_light=False,
):

    obs, _ = env.reset()

    agent.reset()

    actions = np.zeros(steps, dtype=np.int8)
    observations = np.zeros((steps, *obs.shape), np.float32)
    rewards = np.zeros(steps, dtype=np.bool_)
    poses = np.zeros((steps, 3), dtype=np.int32)
    rules = np.zeros(steps, dtype=np.int8)
    inbound = np.zeros(steps, dtype=np.bool_)

    frames = []
    if record_frames:
        frame = env.get_frame(agent_pov=False)
        frames = np.zeros((steps, *frame.shape))

    if callbacks is None:
        callbacks = []

    agent_infos = []

    done, reward = False, 0
    if progress_bar:
        progress = trange
    else:
        progress = range

    env_info = {"prev": 0}
    for step in progress(steps):
        observation = [obs]
        if observable_reward:
            observation.append(int(reward > 0))
        if observable_light:
            observation.append(env_info["prev"])

        action, agent_info = agent.act(*observation)
        # At step t=step the obs is obs, and the action predicted is action
        observations[step] = obs
        actions[step] = action
        rewards[step] = reward > 0
        poses[step] = [*env.unwrapped.agent_pos, env.unwrapped.agent_dir]
        rules[step] = env.unwrapped.task_mode.value

        inbound[step] = (
            env.unwrapped.success_positions[0]
            == env.unwrapped.corridor_locations[1]
        )

        if record_frames:
            frames[step] = env.get_frame(agent_pov=False)

        if record_agent_info:
            agent_infos.append(agent_info)

        if reward > 0:
            agent._goals = env.unwrapped.success_positions

        # check for done before taking the step, i.e. so the
        # observation of the previous time step is logged
        if done and stop_when_done:
            break

        if action < 2.5:  # add the rest action in other cases
            obs, reward, done, truncated, env_info = env.step(action)

        for cb in callbacks:
            cb(agent, env_info)

    return {
        "obs": observations[: step + 1],
        "action": actions[: step + 1],
        "reward": rewards[: step + 1],
        "pose": poses[: step + 1],
        "frames": frames[: step + 1],
        "inbound": inbound[: step + 1],
        "agent_info": agent_infos,
        "rules": rules,
    }
