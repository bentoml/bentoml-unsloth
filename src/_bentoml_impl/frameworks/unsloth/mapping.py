RUNTIME_MAPPING = {
  'llama': {
    'service_config': {'traffic': {'timeout': 300}, 'resources': {'gpu': 1, 'gpu_type': 'nvidia-l4'}},
    'engine_config': {'max_model_len': 2048},
  },
  'mistral': {
    'service_config': {'traffic': {'timeout': 300}, 'resources': {'gpu': 1, 'gpu_type': 'nvidia-l4'}},
    'engine_config': {'max_model_len': 2048},
  },
  'gemma': {
    'service_config': {'traffic': {'timeout': 300}, 'resources': {'gpu': 1, 'gpu_type': 'nvidia-l4'}},
    'engine_config': {'max_model_len': 2048},
  },
  'gemma2': {
    'service_config': {'traffic': {'timeout': 300}, 'resources': {'gpu': 1, 'gpu_type': 'nvidia-l4'}},
    'engine_config': {'max_model_len': 2048},
  },
  'qwen2': {
    'service_config': {'traffic': {'timeout': 300}, 'resources': {'gpu': 1, 'gpu_type': 'nvidia-l4'}},
    'engine_config': {'max_model_len': 2048},
  },
}


def get_extras():
  return {
    'gemma2': {
      'envs': [{'name': 'VLLM_ATTENTION_BACKEND', 'value': 'FLASHINFER'}],
      'python': {
        'extra_index_url': ['https://flashinfer.ai/whl/cu121/torch2.3'],
        'packages': ['flashinfer==0.1.2+cu121torch2.3'],
      },
    }
  }
