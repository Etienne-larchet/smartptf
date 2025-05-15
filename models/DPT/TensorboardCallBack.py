from stable_baselines3.common.callbacks import BaseCallback

#TODO: Vectorize the environment for speed

class TensorboardCallBack(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_reward = 0
        self.episode_length = 0

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        reward = self.locals['rewards'][0]
        done = self.locals['dones'][0]

        if info['retry_count'] == 0:
            self.episode_reward += reward
            self.episode_length += 1

        next_liab = info.pop('next_liability')
        self.logger.record('custom/next_liability_amount', next_liab[1])
        self.logger.record('custom/next_liability_month', next_liab[0])

        for k, v in info.items():
            if v is not None:
                self.logger.record(f'custom/{k}', v)
        
        if done:
            self.logger.record("rollout/ep_rew_mean", self.episode_reward)
            self.logger.record("rollout/ep_len_mean", self.episode_length)
            self.episode_reward = 0
            self.episode_length = 0

        self.logger.dump(self.num_timesteps)
        return True # Return True to continue training

