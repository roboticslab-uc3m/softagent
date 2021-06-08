# SoftAgent

The main goal of this fork is to add [Ravens](https://github.com/google-research/ravens) support to SoftGym.

## Installation 

1. Install SoftGym by following the instructions in [SoftGym](https://github.com/roboticslab-uc3m/softgym) repository. Then, copy the softgym code to the SoftAgent root directory so we have the following file structure:
    ```
    softagent
    ├── cem
    ├── ...
    ├── softgym
    ```
2. For dependency installation see original repository: https://github.com/Xingyu-Lin/softagent

3. Install Ravens by following the instructions in [Ravens](https://github.com/roboticslab-uc3m/ravens) repository. After this, file structure should look like:
    ```
    softagent
    ├── cem
    ├── ravens
    ├── ...
    ├── softgym
    ``` 

## Usage

### Database generation
First step is to generate a database from an existing trajectory following ravens format. A test trajectory is provided in `./cem/trajs/cem_traj_test.pkl`, but you can create your own by running `python experiments/run_cem.py` (see [SoftAgent](https://github.com/Xingyu-Lin/softagent)).

Translate `./cem/trajs/cem_traj_test.pkl` to ravens db format with:

```
python cem/run_cem.py --to_ravens_db True --env_name ClothFoldPPP
```
Ravens db is written by default to `./data/ravens_sg` (see --demo_dir). 

### Training 

In order to train an agent with the previous db:

```
cd ~/softagent/
mkdir ravens/soft-fold-train
cp -r data/ravens_sg/* ravens/soft-fold-train/
python ravens/train.py --task=soft-fold --agent=transporter --n_demos=1
```
You can stop it at 1000 iterations (first tensorflow checkpoint). And continue with testing

### Testing 
```
cd ~/softagent/
mkdir ravens/soft-fold-test
cp -r data/ravens_sg/* ravens/soft-fold-test/
python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=soft-fold --agent=transporter --n_demos=1 --n_steps=1000
```
