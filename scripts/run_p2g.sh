#! /bin/bash 

deepspeed --num_gpus 4 --module p2g --config configs/p2g_cfg/opt1.3b_cora.yaml 2>&1 | tee log 
