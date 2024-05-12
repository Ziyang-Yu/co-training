python main.py \
  --dataset reddit \
  --dropout 0.5 \
  --lr 0.01 \
  --n-partitions 2 \
  --n-epochs 3000 \
  --model graphsage \
  --n-layers 4 \
  --n-hidden 256 \
  --log-every 10 \
  --inductive \
  --enable-pipeline \
  --use-pp