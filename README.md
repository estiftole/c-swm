**Work in progress!**
## Contrastively-trained Structured World Models
This is an implementation of the [C-SWM](https://arxiv.org/abs/1911.12247) (Kipf et al., ICLR 2020). 
### Overview
#### A simplified diagram of the architecture
![c-swm simplified](<./images/c-swm simplified.png>)
C-SWMs can learn object-factored state representations and state transition models directly from visual observations. Instead of pixel-level reconstruction, it leverages a Contrastive Loss to learn relevant latent representations.
