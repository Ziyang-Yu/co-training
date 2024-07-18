# P2G: Pipeline training towards GNN 

## Environment setup

Build the conda env from the `env.yaml` file:
```bash
conda env create -f env.yaml
```
Install the `p2g` package and `deepspeed`:
```bash
cd DeepSpeed && python setup.py develop && cd .. 
python setup.py develop
```

## Ckpt download

Use the scripts under `scripts/save/` to download the checkpoints of llama2-7b and opt-1.3b and turn them into the p2g format. 

## Run

Refer to `scripts/run_p2g.sh`.
