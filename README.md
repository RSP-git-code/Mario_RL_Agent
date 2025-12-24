# Super Mario Bros Reinforcement Learning Agent (PPO)

This project implements a **Proximal Policy Optimization (PPO)** agent to play *Super Mario Bros* using **Stable-Baselines3** and **Gym Super Mario Bros**.

The goal of this project is to demonstrate correct environment setup, PPO training, checkpointing, and evaluation for a classic reinforcement learning problem.

---

## 🧠 Environment Setup

- **Game**: Super Mario Bros (NES)
- **RL Algorithm**: PPO (Proximal Policy Optimization)
- **Framework**: Stable-Baselines3
- **Observation Space**: Grayscale frames
- **Action Space**: SIMPLE_MOVEMENT
- **Hardware**: GPU is better for faster training of RL model\CPU with i7 processo(but will take longer time for training)

---

## ⚙️ Training Details

- Policy: `CnnPolicy`
- Learning Rate: `2.5e-4`
- Gamma: `0.99`
- GAE Lambda: `0.95`
- Batch Size: `64`
- Steps per rollout: `512`
- Training Timesteps: **~100,000**

> Due to hardware and power limitations on a CPU-only setup, longer training runs were interrupted.  
> The provided model represents an **early-stage trained agent** and demonstrates correct PPO training and evaluation.


---
##  Mario RL 100K models perfomance:
Video: https://drive.google.com/file/d/1FB7zL3TlV5O5OL_Ktke0LZiXAm003ZGk/view?usp=drive_link


