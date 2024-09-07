## BentoML Unsloth integrations

## Installation

```bash
pip install "bentoml[unsloth]"
```

## Examples.

See [train.py](https://github.com/bentoml/bentoml-unsloth/blob/main/train.py)

## API

To use this integration, one can use `bentoml.unsloth.build_bento`:

```python
bentoml.unsloth.build_bento(model, tokenizer)
```

If you model is continued froma fine-tuned checkpoint, then `model_name` must be passed as well:

```python
bentoml.unsloth.build_bento(model, tokenizer, model_name="llama-3-continued-from-checkpoint")
```

> [!important]
>
> Make sure to save the chat templates to tokenizer instance to make sure generations are correct based on how you setup your data pipeline.
> See [example](https://github.com/bentoml/bentoml-unsloth/blob/da52d51366ea3217a3ee644f80042b1f425e00c6/train.py#L42) and [documentation](https://huggingface.co/docs/transformers/main/en/chat_templating#advanced-adding-and-editing-chat-templates) for more information.
