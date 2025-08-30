from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
#from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset
import torch
import os
import wandb
import warnings

os.environ["USE_FLASH_ATTENTION"]="1"
token = "secret_token"

model_id = 'meta-llama/Llama-3.1-8B-Instruct'

tokenizer = AutoTokenizer.from_pretrained(model_id, token=token)

quantize_config = BitsAndBytesConfig(
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
  model_id,
  device_map="cuda:0",
  torch_dtype=torch.bfloat16,
  quantization_config=quantize_config,
  token=token
)

warnings.filterwarnings("ignore")

def generate(inputs, length=100):
  input_ids = tokenizer.apply_chat_template(
          inputs,
          add_generation_prompt=True,
          return_tensors="pt"
      ).to(model.device)

  output_ids = model.generate(
          input_ids,
          max_new_tokens=length,
          eos_token_id=model.config.eos_token_id,
          do_sample=True,
          temperature=0.5
      )[0][input_ids.shape[-1]:]
  return tokenizer.decode(output_ids, skip_special_tokens=True)


def explain(review, cat):
  sys_msg_gen = [{"role": "system",
               "content": f"The following review was classified as the category \"{cat}\". Please give an appropriate explanation in two sentences.\n\nReview:\n\"{review}\""}]
  return generate(sys_msg_gen)