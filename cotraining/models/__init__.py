from .deberta import deberta
from .bert import bert
from .graphsage import graphsage
from .vanilla_gcn import GCN as vanilla_gcn

def gnn_registry(model_name, model_kwargs):
    if model_name == "graphsage":
        return graphsage(**model_kwargs)
    elif model_name == 'vanilla_gcn':
        return vanilla_gcn(**model_kwargs)
    else:
        raise ValueError(f'not supported gnn {model_name}')