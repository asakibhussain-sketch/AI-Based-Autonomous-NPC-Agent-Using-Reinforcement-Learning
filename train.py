"""
train.py  --  Phase 3: PPO training for the NPC agent.

Usage
-----
  python train.py                  # train with default settings
  python train.py --timesteps 1000000
  python train.py --render         # watch one env (slower)

Outputs
-------
  models/npc_ppo_best.zip          # best model (by eval reward)
  models/npc_ppo_final.zip         # model at end of training
  logs/training_log.csv            # per-episode metrics
  logs/ppo_tensorboard/            # TensorBoard logs (optional)
"""

import os
import csv
import time
import argparse
import numpy as np

# ---- Headless SDL (must be before any pygame import) -------------------
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from stable_baselines3 import PPO
from stable_baselines3.common.env_util   import make_vec_env
from stable_baselines3.common.callbacks  import BaseCallback, EvalCallback
from stable_baselines3.common.monitor    import Monitor
from stable_baselines3.common.vec_env    import DummyVecEnv

from env import NPCEnv

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
DEFAULT_TIMESTEPS = 500_000
N_ENVS            = 4
MODEL_DIR         = "models"
LOG_DIR           = "logs"
EVAL_FREQ         = 25_000   # steps between evaluations
N_EVAL_EPS        = 12


# ---------------------------------------------------------------------------
# Metrics Callback
# ---------------------------------------------------------------------------

class TrainingMetricsCallback(BaseCallback):
    """
    Logs per-episode metrics to console (every 10 episodes) and CSV.
    Also tracks win rate (NPC killed player) over a rolling 100-episode window.
    """

    def __init__(self, log_path: str, verbose: int = 1):
        super().__init__(verbose)
        self.log_path    = log_path
        self._ep_rewards = []
        self._ep_lengths = []
        self._wins       = []
        self._episode    = 0
        self._train_start = time.time()
        self._csv_file   = None
        self._writer     = None

    def _on_training_start(self):
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._csv_file = open(self.log_path, "w", newline="", encoding="utf-8")
        self._writer   = csv.writer(self._csv_file)
        self._writer.writerow([
            "timestep", "episode",
            "ep_reward", "ep_length",
            "win_rate_100", "mean_reward_100",
            "elapsed_s",
        ])
        self._train_start = time.time()

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "episode" not in info:
                continue
            ep_info = info["episode"]
            r       = float(ep_info["r"])
            l       = int(ep_info["l"])
            # NPC wins when player dies (player_alive==False at episode end)
            won     = int(not info.get("player_alive", True))

            self._ep_rewards.append(r)
            self._ep_lengths.append(l)
            self._wins.append(won)
            self._episode += 1

            win_rate  = float(np.mean(self._wins[-100:])) * 100
            mean_r    = float(np.mean(self._ep_rewards[-100:]))
            elapsed   = time.time() - self._train_start

            self._writer.writerow([
                self.num_timesteps, self._episode,
                f"{r:.3f}", l,
                f"{win_rate:.1f}", f"{mean_r:.3f}",
                f"{elapsed:.1f}",
            ])
            self._csv_file.flush()

            if self.verbose and self._episode % 10 == 0:
                bar_w  = int(win_rate / 5)           # 0-20 chars
                bar    = "#" * bar_w + "." * (20 - bar_w)
                print(
                    f"[Ep {self._episode:5d} | {self.num_timesteps:>9,} steps]  "
                    f"R: {r:+8.2f}  L: {l:5d}  "
                    f"Win%: [{bar}] {win_rate:5.1f}  "
                    f"MeanR: {mean_r:+8.2f}  "
                    f"t: {elapsed:6.0f}s"
                )
        return True

    def _on_training_end(self):
        if self._csv_file:
            self._csv_file.close()
        if self.verbose:
            total = time.time() - self._train_start
            wr    = float(np.mean(self._wins[-100:])) * 100 if self._wins else 0
            print(f"\nTraining complete in {total:.0f}s  |  "
                  f"Episodes: {self._episode}  |  "
                  f"Final win-rate (last 100): {wr:.1f}%")


# ---------------------------------------------------------------------------
# Env factory
# ---------------------------------------------------------------------------

def make_env_fn(render_mode=None):
    def _init():
        env = NPCEnv(render_mode=render_mode)
        return Monitor(env)
    return _init


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(total_timesteps: int = DEFAULT_TIMESTEPS,
          render: bool = False) -> None:

    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,   exist_ok=True)

    print("=" * 65)
    print("   AI NPC Agent  --  PPO Training  (Phase 3)")
    print("=" * 65)
    print(f"  Total timesteps  : {total_timesteps:,}")
    print(f"  Parallel envs    : {N_ENVS}")
    print(f"  Model output     : {MODEL_DIR}/npc_ppo_best.zip")
    print(f"  Metrics log      : {LOG_DIR}/training_log.csv")
    print("=" * 65 + "\n")

    # One visible env for debugging if requested
    render_mode = "human" if render else None
    vec_env = make_vec_env(
        make_env_fn(render_mode=render_mode),
        n_envs=N_ENVS,
        vec_env_cls=DummyVecEnv,
    )
    eval_env = Monitor(NPCEnv(render_mode=None))

    model = PPO(
        policy           = "MlpPolicy",
        env              = vec_env,
        learning_rate    = 3e-4,
        n_steps          = 2048,
        batch_size       = 64,
        n_epochs         = 10,
        gamma            = 0.99,
        gae_lambda       = 0.95,
        clip_range       = 0.2,
        ent_coef         = 0.01,
        vf_coef          = 0.5,
        max_grad_norm    = 0.5,
        verbose          = 0,
        tensorboard_log  = os.path.join(LOG_DIR, "ppo_tensorboard"),
        policy_kwargs    = dict(net_arch=[128, 128]),
    )

    callbacks = [
        TrainingMetricsCallback(
            log_path = os.path.join(LOG_DIR, "training_log.csv"),
            verbose  = 1,
        ),
        EvalCallback(
            eval_env             = eval_env,
            best_model_save_path = MODEL_DIR,
            log_path             = LOG_DIR,
            eval_freq            = max(EVAL_FREQ // N_ENVS, 1),
            n_eval_episodes      = N_EVAL_EPS,
            deterministic        = True,
            verbose              = 1,
        ),
    ]

    model.learn(
        total_timesteps = total_timesteps,
        callback        = callbacks,
        progress_bar    = True,
    )

    # Save final model alongside best
    final_path = os.path.join(MODEL_DIR, "npc_ppo_final")
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}.zip")

    vec_env.close()
    eval_env.close()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NPC PPO agent")
    parser.add_argument("--timesteps", type=int,  default=DEFAULT_TIMESTEPS,
                        help="Total training timesteps")
    parser.add_argument("--render",    action="store_true",
                        help="Show one live environment window (slower)")
    args = parser.parse_args()

    train(total_timesteps=args.timesteps, render=args.render)
