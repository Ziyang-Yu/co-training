#! /bin/bash 

export PYTHONPATH=$PYTHONPATH:./DeepSpeed/:./co-training/
DeepSpeed/bin/deepspeed --num_gpus=4 pipe_modeling.py 2>&1 | tee log
