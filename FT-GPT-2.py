# -*- coding: utf-8 -*-

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch

#%% Cargar el tokenizador y el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device = ", device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
base_model.config.pad_token_id = tokenizer.pad_token_id

#%% Cargar el dataset
dataset = load_dataset('gopalkalpande/bbc-news-summary')

dataset_barajado = dataset["train"].shuffle(seed=42)
indices = int(0.8 * len(dataset_barajado))

train_dataset = dataset_barajado.select(range(indices))    
eval_dataset = dataset_barajado.select(range(indices, len(dataset_barajado)))

#%%
# Cargamos el modelo y el tokenizadpor si es necesario

FT_model = AutoModelForCausalLM.from_pretrained("C:/Users/pablo/ModelosLLM/GPT2-FT").to(device)
FT_tokenizer = AutoTokenizer.from_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorGPT2-FT")

# %%

def generate_summary(text,model, tokenizer, device):
    articles = "Summarize this article:\n"+text+"Summary:\n"
    inputs = tokenizer(
        articles,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024
        ).to(device)
    
    input_length = inputs["input_ids"].shape[1]
    
    summary_ids = model.generate(
        **inputs,
        max_new_tokens=250,
        num_return_sequences=1,
        do_sample=True,
        top_k=5
    )
    generated_ids = summary_ids[:, input_length:]
    
    summary = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
    return summary
# %%


article = "Climate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change."
print(generate_summary(article,base_model,tokenizer, device))
print("_"*100)
print(generate_summary(article,FT_model,FT_tokenizer, device))
print(article)

