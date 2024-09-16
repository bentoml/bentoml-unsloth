def prep_dataset(tokenizer):
  alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""

  EOS_TOKEN = tokenizer.eos_token  # Must add EOS_TOKEN

  def formatting_prompts_func(examples):
    instructions = examples['instruction']
    inputs = examples['input']
    outputs = examples['output']
    texts = []
    for instruction, input, output in zip(instructions, inputs, outputs):
      # Must add EOS_TOKEN, otherwise your generation will go on forever!
      text = alpaca_prompt.format(instruction, input, output) + EOS_TOKEN
      texts.append(text)
    return {'text': texts}

  from datasets import load_dataset

  dataset = load_dataset('yahma/alpaca-cleaned', split='train')
  dataset = dataset.map(
    formatting_prompts_func,
    batched=True,
  )
  return dataset


def llama31_bnb(max_seq_length:int=8196,max_steps:int=100)->int:
  import unsloth, trl, transformers
  from _bentoml_impl.frameworks.unsloth import build_bento

  model, tokenizer = unsloth.FastLanguageModel.from_pretrained('unsloth/Meta-Llama-3.1-8B-bnb-4bit', max_seq_length=max_seq_length, load_in_4bit=True)
  # alpaca chat templates
  tokenizer.chat_template="{% if messages[0]['role'] == 'system' %}{% set loop_messages = messages[1:] %}{% set system_message = messages[0]['content'].strip() + '\n\n' %}{% else %}{% set loop_messages = messages %}{% set system_message = '' %}{% endif %}{{ bos_token + system_message }}{% for message in loop_messages %}{% if (message['role'] == 'user') != (loop.index0 % 2 == 0) %}{{ raise_exception('Conversation roles must alternate user/assistant/user/assistant/...') }}{% endif %}{% if message['role'] == 'user' %}{{ '### Instruction:\n' + message['content'].strip() + '\n\n' }}{% elif message['role'] == 'assistant' %}{{ '### Response:\n' + message['content'].strip() + eos_token + '\n\n' }}{% endif %}{% if loop.last and message['role'] == 'user' and add_generation_prompt %}{{ '### Instruction:\n' }}{% endif %}{% endfor %}"
  model = unsloth.FastLanguageModel.get_peft_model(
    model, target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    r=16, lora_alpha=16, lora_dropout=0, bias='none', random_state=3407,
    use_gradient_checkpointing='unsloth',  # True or 'unsloth' for very long context
  )
  trl.SFTTrainer(
    model=model, tokenizer=tokenizer,
    train_dataset=prep_dataset(tokenizer),
    dataset_text_field='text', max_seq_length=max_seq_length,
    dataset_num_proc=2, packing=False,  # Can make training 5x faster for short sequences.
    args=transformers.TrainingArguments(
      per_device_train_batch_size=2, gradient_accumulation_steps=4,
      warmup_steps=5, max_steps=max_steps, learning_rate=2e-4,
      weight_decay=0.01, seed=3407, optim='adamw_8bit',
      fp16=not unsloth.is_bfloat16_supported(), bf16=unsloth.is_bfloat16_supported(),
      logging_steps=1,
      lr_scheduler_type='linear',
      output_dir='outputs',
    ),
  ).train()

  build_bento(model, tokenizer, engine_config={'quantization': 'bitsandbytes', 'load_format': 'bitsandbytes'})
  return 0

if __name__ == '__main__':
  import argparse

  parser = argparse.ArgumentParser(description='Run language model training')
  parser.add_argument('--model', type=str, default='llama-3.1', choices=['llama-3.1'], help='Model to use')
  parser.add_argument('--max_steps', type=int, default=100, help='Max steps to train for')
  parser.add_argument('--max_seq_length', type=int, default=8196, help='Max sequence length to train for')
  args = parser.parse_args()

  if args.model.lower() == 'llama-3.1': raise SystemExit(llama31_bnb(max_steps=args.max_steps,max_seq_length=args.max_seq_length))
  else:
    parser.print_help()
    raise SystemExit(1)
