**Work in progress!**
## Contrastively-trained Structured World Models
This is an implementation of the [C-SWM paper](https://arxiv.org/abs/1911.12247) (Kipf et al., ICLR 2020) adapted from the [official implementation by the authors](https://github.com/tkipf/c-swm). 
### Overview
![c-swm simplified](<./images/c-swm simplified.png>)
*A simplified representation of the C-SWM architecture*  

C-SWMs can learn object-factored state representations and state transition models directly from visual observations. Instead of pixel-level reconstruction, it optimizes a contrastive loss to learn relevant latent representations.
