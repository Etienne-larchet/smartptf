import os
import sys
import time
from datetime import date
from pathlib import Path

import polars as pl
from stable_baselines3 import PPO, TD3

ROOT_DIR = Path(os.getcwd())
sys.path.append(str(ROOT_DIR))

from models.DPT import DPTEnv, TensorboardCallBack
from utils.polars import TimesSeriesPolars


data = pl.read_csv("data/index_historical/SP500_ohlcv_2004-11-30_to_2024-12-31.csv", schema_overrides={"Date": date})
data_holder = TimesSeriesPolars(data=data, index_ticker='GSPC.INDX')
data_holder.calculate_logR()
full_returns = data_holder.get('logR', include_index=True)

# model_path = models_dir / '1500.zip'

models_dir = f"models/DPT/trained-agents/{int(time.time())}/"
logdir = f"models/DPT/logs/{int(time.time())}/"

if not os.path.exists(models_dir):
	os.makedirs(models_dir)

if not os.path.exists(logdir):
	os.makedirs(logdir)

env = DPTEnv(full_returns, index_ticker='GSPC.INDX')
env.reset()
# model = TD3('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, learning_starts=2000)
model = TD3('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, buffer_size=100_000)
# model = PPO('MultiInputPolicy', env, verbose=1, tensorboard_log=logdir, batch_size=256)
# model = TD3.load(model_path, env)
# model.tensorboard_log = logdir

TIMESTEPS = 1000
for i in range(100):
    model.learn(
        total_timesteps=TIMESTEPS, 
        reset_num_timesteps=False, 
        tb_log_name='TD3-tweak5customcallback', 
        callback=TensorboardCallBack(), 
        progress_bar=True
    )
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")
