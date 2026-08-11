**Work in progress!**
## Contrastively-trained Structured World Models
This is an implementation of the [C-SWM paper](https://arxiv.org/abs/1911.12247) (Kipf et al., ICLR 2020) adapted from the [official implementation by the authors](https://github.com/tkipf/c-swm). 
### Overview
![c-swm simplified](<./images/c-swm simplified.png>)
*A simplified representation of the C-SWM architecture (only showing the encoder and transition modules)*  

C-SWMs can learn object-factored state representations and state transition models directly from visual observations. Instead of pixel-level reconstruction, it optimizes a contrastive loss to learn relevant latent representations.

### Setup 

```bash
# Clone repo
git clone -b scratch https://github.com/estiftole/c-swm.git
cd c-swm

# Install uv (if you don't already have it)
pip install uv

# Install requirements
uv pip install -r pyproject.toml
```

### Generate Data
Pong
```bash
# Generate pong training and eval data
uv run gen_data.py --env_id ALE/Pong-v5 --fname data/pong_train.h5 --num_episodes {num_train_episodes} --atari --seed 1

uv run gen_data.py --env_id ALE/Pong-v5 --fname data/pong_eval.h5 --num_episodes {num_eval_episodes} --atari --seed 2
```

Breakout
```bash
# Generate breakout training and eval data
uv run gen_data.py --env_id ALE/Breakout-v5 --fname data/breakout_train.h5 --num_episodes {num_train_episodes} --atari --seed 1

uv run gen_data.py --env_id ALE/Breakout-v5 --fname data/breakout_eval.h5 --num_episodes {num_eval_episodes} --atari --seed 2
```

Centipede
```bash
# Generate centipede training and eval data
uv run gen_data.py --env_id ALE/Centipede-v5 --fname data/centipede_train.h5 --num_episodes {num_train_episodes} --atari --seed 1

uv run gen_data.py --env_id ALE/Centipede-v5 --fname data/centipede_eval.h5 --num_episodes {num_eval_episodes} --atari --seed 2
```

### Train
```bash
# Train standard C-SWM
uv run train.py --dataset data/trainset_name.h5 --embedding-dim 4 --action-dim 6 --n
um-slots 3 --batch-size {batch_size} --global-action --epochs {num_epochs} --name experiment_name --seed {seed}

# Train C-SWM with decoder for reconstruction loss (simply add a '--decoder' flag)
uv run train.py --dataset data/trainset_name.h5 --embedding-dim 4 --action-dim 6 --n
um-slots 3 --batch-size {batch_size} --global-action --epochs {num_epochs} --name experiment_name --seed {seed} --decoder
```

### Evaluate
```bash
# Evaluate model
uv run eval.py --dataset data/evalset_name.h5 --save-folder checkpoints/experiment_name --num-steps {n_steps}
```

### Cite
If you make use of this code in your own work, please cite the original paper:
```
@article{kipf2019contrastive,
  title={Contrastive Learning of Structured World Models}, 
  author={Kipf, Thomas and van der Pol, Elise and Welling, Max}, 
  journal={arXiv preprint arXiv:1911.12247}, 
  year={2019} 
}
```
