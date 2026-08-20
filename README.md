# AI-Based Autonomous NPC Agent Using Reinforcement Learning

## Project Overview

A Pygame tactical arena for visualising and training an autonomous NPC agent.
The project includes a playable rule-based NPC, a Gymnasium environment, PPO
training, trained-model playback, and evaluation charts.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

To play against the trained RL policy:

```bash
python play_rl.py
```

Requires Python 3.9+ and no external game assets. Everything is drawn with
Pygame primitives.

## Controls

| Key(s) | Action |
|---|---|
| W / A / S / D | Move player |
| Arrow keys | Move player |
| Space | Attack |
| R | Restart round |
| Esc | Quit |

## Architecture

```text
main.py        Entry point
game.py        Main loop, round state machine, win/lose logic
player.py      Player input, movement, attack, HP, rendering
npc.py         Rule-based NPC state machine and rendering
obstacles.py   Obstacle, SafeZone, Target, particles, collision helpers
hud.py         In-game HUD, telemetry panels, minimap
env.py         Gymnasium NPCEnv wrapper for training
train.py       PPO training with metrics and checkpoints
play_rl.py     Play mode using a trained PPO policy
evaluate.py    Evaluation report and charts
```

## NPC State Machine

```text
PATROL  -> player outside FOV
CHASE   -> player in FOV but outside melee range
ATTACK  -> player inside melee range
DEFEND  -> player is attacking near the NPC
RETREAT -> NPC HP below 30%
```

`NPC.decide_action(player_pos, player_alive, player_is_attacking)` returns
`(move_dir, do_attack, do_defend)`. The play mode can bypass this with a PPO
policy while reusing the same rendering, collision, and round logic.

## Reinforcement Learning

### Observation Space

`Box(15,)`

| Index | Feature |
|---|---|
| 0-1 | NPC position `(x, y)` normalised |
| 2-3 | Player position normalised |
| 4-5 | NPC HP fraction, player HP fraction |
| 6-8 | Distance to player, sin/cos bearing |
| 9-11 | Distance to target, sin/cos bearing |
| 12 | Player alive flag |
| 13 | Target active flag |
| 14 | Player attacking flag |

### Action Space

`MultiDiscrete([9, 3])`

| Axis | Meaning |
|---|---|
| 0 | Movement: stay, N, S, W, E, NW, NE, SW, SE |
| 1 | Combat: idle, attack, defend |

### Train

```bash
python train.py
python train.py --timesteps 100000
python train.py --render
```

Outputs:

| Path | Purpose |
|---|---|
| models/best_model.zip | Best evaluation checkpoint |
| models/npc_ppo_final.zip | Final training checkpoint |
| logs/training_log.csv | Per-episode metrics |
| logs/ppo_tensorboard/ | TensorBoard logs |

### Evaluate

```bash
python evaluate.py
python evaluate.py --episodes 100
python evaluate.py --model models/npc_ppo_final
```

Outputs:

| Path | Purpose |
|---|---|
| results/evaluation.png | Training and policy comparison charts |
| results/report.txt | Written evaluation summary |

## Colour Theme

| Element | Colour |
|---|---|
| Player | Cyan |
| NPC | Magenta |
| Target | Amber |
| Safe zone | Green |
| Arena | Dark navy |
