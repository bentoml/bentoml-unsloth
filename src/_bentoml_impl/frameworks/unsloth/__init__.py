from __future__ import annotations

import logging, math, os, pathlib, shutil, subprocess, importlib.util, sys, yaml, tempfile, typing as t, yaml, bentoml

from deepmerge.merger import Merger

from .mapping import RUNTIME_MAPPING as MAPPINGS, get_extras

if importlib.util.find_spec("unsloth") is None:
  raise bentoml.exceptions.MissingDependencyException("'unsloth' is required in order to use module 'bentoml.unsloth', install unsloth with 'pip install bentoml[unsloth]'.")

__all__ = ['build_bento']

if t.TYPE_CHECKING:
  from _bentoml_sdk.service.config import ServiceConfig
  from transformers import PreTrainedModel, PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

merger = Merger([(dict, 'merge'), (list, 'append')], ['override'], ['override'])

def replace_tag(tag: str) -> str: return tag.lower().replace('/', '--')

ModelType = t.Literal['llama', 'mistral', 'gemma', 'gemma2', 'qwen2']

SPEC = {
  'nvidia-tesla-t4': 16.0, 'nvidia-tesla-v100': 16.0,
  'nvidia-l4': 24.0, 'nvidia-tesla-l4': 24.0, 'nvidia-tesla-a10g': 24.0,
  'nvidia-tesla-a100': 40.0,
  'nvidia-a100-80gb': 80.0,
}

def calculate_recommended_gpu_type(model) -> str:
  # ceiling the number of parameters to the nearest billion
  num_params = math.ceil(sum(p.numel() for p in model.parameters()) / 1e9)
  gpu_type = next((k for k, v in SPEC.items() if num_params <= v / 2), None)
  # If no suitable GPU is found, return the one with the highest memory
  return gpu_type or max(SPEC, key=SPEC.get)

def build_bento(
  model: PreTrainedModel, tokenizer: PreTrainedTokenizerFast,
  /,
  model_name: str | None = None,
  *,
  save_method: t.Literal['merged_16bit', 'merged_4bit'] = 'merged_16bit',
  service_config: ServiceConfig | None = None,
  engine_config: t.Dict[str, t.Any] | None = None,  # arguments to pass to AsyncEngineArgs
) -> bentoml.Model:
  # this model is local then model_name must specified, otherwise derived from model_id
  if (getattr(model.config, '_commit_hash', None) is None) and model_name is None:
    raise bentoml.exceptions.BentoMLException('Fine-tune from a local checkpoint requires specifying "model_name".')

  model_name = model_name or replace_tag(model.config._name_or_path)
  model_type = t.cast(ModelType, model.config.model_type)

  if service_config is None: service_config = {}
  if engine_config is None: engine_config = {}

  merger.merge(
    (local_items := MAPPINGS[model_type]['service_config']),
    {'resources': {'gpu': 1, 'gpu_type': calculate_recommended_gpu_type(model)}},
  )
  service_config.update(local_items)
  engine_config.update(MAPPINGS[model_type]['engine_config'])

  if (
    (quantization := engine_config.get('quantization')) is not None
    and (load_format := engine_config.get('load_format')) is not None
    and quantization != load_format
  ):
    raise bentoml.exceptions.BentoMLException(f"'load_format' and 'quantization' must be the same, got ({load_format} and {quantization} respectively)")

  with bentoml.models.create(model_name) as bentomodel: model.save_pretrained_merged(bentomodel.path, tokenizer, save_method=save_method)

  build_opts = dict(
    python=dict(
      packages=['pyyaml', 'vllm==0.6.1.post2', 'unsloth[huggingface]>=2024.9.post2'],
      lock_packages=True,
    ),
    envs=[{'name': 'HF_TOKEN'}],
  )
  merger.merge(build_opts, get_extras().get(model_type, {}))

  logger.info('Building bentos for %s, model_id=%s', model_type, model.config._name_or_path)

  with tempfile.TemporaryDirectory() as tempdir:
    tempdir = pathlib.Path(tempdir)
    shutil.copytree(pathlib.Path(__file__).parent/'template', tempdir, dirs_exist_ok=True)
    with (tempdir/'service_config.yaml').open('w') as f:
      f.write(
        yaml.safe_dump({
          'model_tag': str(bentomodel.tag),
          'engine_config': engine_config,
          'service_config': service_config,
        })
      )
    with (tempdir / 'bentofile.yaml').open('w') as f:
      yaml.dump(dict(
        name=f"{model_name.replace('.', '-')}-service",
        service='service:UnslothVLLM',
        include=['*.py', '*.yaml'],
        docker=dict(python_version='3.11', system_packages=['git']),
        models=[{"tag": str(bentomodel.tag)}],
        description='API Service for running Unsloth models, powered with BentoML and vLLM.',
        **build_opts,
      ), f)

    subprocess.run([sys.executable, '-m', 'bentoml', 'build', str(tempdir)], check=True, cwd=tempdir, env=os.environ)
