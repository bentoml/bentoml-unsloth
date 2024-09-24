from __future__ import annotations

import logging, pathlib, fastapi, yaml, bentoml, vllm.entrypoints.openai.api_server as vllm_api_server

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Load the constants from the yaml file
CONSTANT_YAML = pathlib.Path(__file__).parent/'service_config.yaml'
if not CONSTANT_YAML.exists(): raise FileNotFoundError(f'service_config.yaml not found in {CONSTANT_YAML.parent}')
with CONSTANT_YAML.open('r') as f:
  CONSTANTS = yaml.safe_load(f)

openai_api_app = fastapi.FastAPI()
for route, endpoint, methods in [
  ('/chat/completions', vllm_api_server.create_chat_completion, ['POST']),
  ('/completions', vllm_api_server.create_completion, ['POST']),
  ('/models', vllm_api_server.show_available_models, ['GET']),
]: openai_api_app.add_api_route(path=route, endpoint=endpoint, methods=methods, include_in_schema=True)


@bentoml.mount_asgi_app(openai_api_app, path='/v1')
@bentoml.service(**CONSTANTS['service_config'])
class UnslothVLLM:
  bentomodel = bentoml.models.get(CONSTANTS['model_tag'])

  def __init__(self) -> None:
    from transformers import AutoTokenizer
    from vllm import AsyncEngineArgs, AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion

    self.engine = AsyncLLMEngine.from_engine_args(
      AsyncEngineArgs(model=self.bentomodel.path, enable_prefix_caching=True, **CONSTANTS['engine_config'])
    )
    self.tokenizer = AutoTokenizer.from_pretrained(self.bentomodel.path)
    model_config = self.engine.engine.get_model_config()
    # inject the engine into the openai serving chat and completion
    vllm_api_server.openai_serving_chat = OpenAIServingChat(
      self.engine,
      served_model_names=[self.bentomodel.path],
      chat_template=None,
      response_role='assistant',
      model_config=model_config,
      lora_modules=None,
      prompt_adapters=None,
      request_logger=None,
    )
    vllm_api_server.openai_serving_completion = OpenAIServingCompletion(
      self.engine,
      served_model_names=[self.bentomodel.path],
      model_config=model_config,
      lora_modules=None,
      prompt_adapters=None,
      request_logger=None,
    )
