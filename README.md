<h1 align="center">Co-training of GNN and LLM</h1>


<h2>Experiments Log</h2>

- 2024.04.29 根据TAPE的代码和数据集写了数据处理py文件，使用bert-base-cased处理文本，前向传播可以在cpu上运行，反向传播未知。
- 2024.05.11 跑通了PipeGCN在Reddit数据集的代码：需要放弃老版cuda版本的Pytorch和DGL
- 2024.05.16 实现了Deberta+GraphSage的训练代码（.ipynb）commit a9ae54f
- 2024.05.16 Fix bug: load_data order is different from dgl

<h2>Experimental Caution</h2>

- 服务器最好在中国大陆外（方便连接huggingface）
