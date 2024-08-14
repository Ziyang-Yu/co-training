from vllm import LLM
llm = LLM("meta-llama/Llama-2-7b-hf", tensor_parallel_size=4)
output = llm.generate("San Franciso is a")
