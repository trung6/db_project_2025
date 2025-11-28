import vllm
from vllm import EngineArgs, SamplingParams, LLM

llm = LLM(model="meta-llama/Llama-2-7b-chat-hf")

llm.llm_engine.model_executor.driver_worker