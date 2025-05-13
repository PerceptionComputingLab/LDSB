# LDSB
[CVPR 2025] Pytorch Implementation of the Paper "Finding Local Diffusion Schr\"odinger Bridge using Kolmogorov-Arnold Network"

The code will be released before June 2025.

The code project is optimized based on the official code of DDIM, and the specific library version and configuration can be referred to https://github.com/ermongroup/ddim.
The parameters of the KAN corresponding to the results in the paper have been given in the repository .\exp\logs\.
## Running the Experiments
### Train a model
1. Training KAN with initial paths
```
python main.py --exp {PROJECT_PATH} --config mydataset.yml --doc {MODEL_NAME} --timesteps {STEPS} --ni
```

2. Optimizing paths with LDSB
```
python main.py --exp {PROJECT_PATH} --config mydataset.yml --doc {MODEL_NAME} --timesteps {STEPS} --ni --resume_training
```
where
- `STEPS` controls how many timesteps used in the process.

### Sampling from the model
```
python main.py --config {DATASET}.yml --exp {PROJECT_PATH} --doc {MODEL_NAME} --use_pretrained --sample --fid --timesteps {STEPS} --eta 0 --ni --image_folder
```
where 
- `STEPS` controls how many timesteps used in the process.
- `MODEL_NAME` finds the pre-trained checkpoint according to its inferred path.
- `image_folder` is the path where the output image is saved.

### Limitations and Futures
In this paper, we explore for the first time how to find diffusion Schrödinger bridges in the path subspace. The proposed LDSB method typically reaches the optimal value within 3 iterations during path optimization. However, continued optimization may lead to divergence due to a mismatch between the optimized paths and the training distribution of the pre-trained denoising network. Specifically, as the paths and their corresponding time-step data deviate from the training data, the denoising network produces large and uncontrollable errors. Future work will focus on enhancing the stability of the optimization process and further improving the quality of the generated samples.

- ## References and Acknowledgements
```
@article{qiu2025finding,
  title={Finding Local Diffusion Schr$\backslash$" odinger Bridge using Kolmogorov-Arnold Network},
  author={Qiu, Xingyu and Yang, Mengying and Ma, Xinghua and Li, Fanding and Liang, Dong and Luo, Gongning and Wang, Wei and Wang, Kuanquan and Li, Shuo},
  journal={arXiv preprint arXiv:2502.19754},
  year={2025}
}
```


