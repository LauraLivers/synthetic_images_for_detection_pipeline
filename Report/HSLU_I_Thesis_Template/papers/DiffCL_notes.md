## concepts
### Long Tail Classification
imbalanced data distribution with few samples in the tail classes
#### Challenge
capture critical features of target classes (hard because intricate background, occlusion, motion blur)
#### Mitigation
1. learn synthetic images with lower image guidance for tail classes (they enhance diversity and quantity of of data)
2. increase guidance (learn synthetic images closer to the original images)

### hard samples
low quality/quantity data $\rightarrow$ makes the model more prone to the gap between test and training distributions (introduction of of bias and outliers). Model struggles to extract helpful features for target classification
- LT-Classification: difficulty of each sample depends on whether it belongs to tail classes
- low quality data: use loss or confidence on the ground truth to measure and mark as hard samples
#### Identification


## related work
- He et all (GLIDE) can improve zero-shot and few-shot performance 
- Bengio et all (Curriculum Learning): step-by-step learning like humans

## Approach
### Phase 1
Synthetic-to-Real Data Generation $\rightarrow$ syn-tp-real spectrum of interpolated data for hard samples
#### 1. Synthetic Data Generation with Image Guidance aka Stable Diffusion
latent representation of original image $z_{real}$, denoising (backward diffusion) process starts at any step $t$ with initial $z_t$:  
$$
z_t = \sqrt{\tilde{\alpha}_t}z_{real}  + \sqrt{1 - \tilde{\alpha}}\mathbf{\epsilon}, \mathbf{\epsilon} \sim \mathcal{N}(0, \mathbf{I})
$$
remaining denoising steps apply iteratively noise estimation $\hat{\mathbf{\epsilon}}_t$:
$$
\hat{\mathbf{\epsilon}}_t = (1+\omega)\mathbf{\epsilon}_\theta(z_t, t|c) - \omega\mathbf{\epsilon}_\theta(z_t, t)\\
z_{t-1} = \frac{1}{\sqrt{\alpha_t}}\Bigl(z_t - \frac{\beta_t}{\sqrt{1 - \tilde{\alpha_t}}}\hat{\mathbf{\epsilon}}) + \sqrt{\beta_t}\mathbf{\epsilon}', t \leftarrow t - 1
$$
until $t=0$ resulting in synthetic image $z_0$.

$t(\lambda) = \lfloor(1 - \lambda)T\rfloor, \lambda \in \[0, 1\]$
larger $\lambda$ leads to higher fidelity between $z_0$ and $z_{real}$

#### 2. Synthetic-to-Real Spectrum of Generated Images


### Phase II
Generative Curriculum learning based on synthetic data from Phase I
