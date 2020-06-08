# GRAC
implementation of our self-guided and self-regularized actor-critic algorithm

## Requirement
```
python >= 3.6;
mujoco >= 2.0;
mujoco-py;
```

## Run the code
```
python main.py --policy GRAC --env Ant-v2 --max_timesteps 3000000 --which_cuda 0 --seed 0 --exp_name exp_Ant
```
