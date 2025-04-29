# -*- coding: utf-8 -*-
from transformers import Trainer, TrainingArguments,AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
import torch
from peft import PeftModel
from evaluate import load  
#%%
device = "cuda" if torch.cuda.is_available() else "cpu"

print("Devise = ",device)

model_name = "openai-community/gpt2"

FT_model_path="C:/Users/pablo/ModelosLLM/GPT2-FT"
tokenizer_FT_path="C:/Users/pablo/ModelosLLM/tokenizadorGPT2-FT"

lora_model_path="C:/Users/pablo/ModelosLLM/GPT2-Lora-FT"
tokenizer_Lora_path="C:/Users/pablo/ModelosLLM/tokenizadorGPT2-Lora-FT"


#%% Cargar los tokenizadores y los modelos

base_model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
base_tokenizer = AutoTokenizer.from_pretrained(model_name)

#Definimos un token para el padding
base_tokenizer.pad_token = base_tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
base_model.config.pad_token_id = base_tokenizer.pad_token_id


FT_model = AutoModelForCausalLM.from_pretrained(FT_model_path).to(device)
FT_tokenizer = AutoTokenizer.from_pretrained(tokenizer_FT_path)


##lora_model = PeftModel.from_pretrained(base_model, lora_model_path).to(device)
lora_tokenizer = AutoTokenizer.from_pretrained(tokenizer_Lora_path)

#%%
lora_model = PeftModel.from_pretrained(base_model, lora_model_path)
lora_model = lora_model.merge_and_unload()  # Merge adapters into the base model
lora_model.to(device)

#%% Cargar el dataset
dataset = load_dataset('gopalkalpande/bbc-news-summary')

dataset_barajado = dataset["train"].shuffle(seed=42)
indices = int(0.8 * len(dataset_barajado))

train_dataset = dataset_barajado.select(range(indices))    
eval_dataset = dataset_barajado.select(range(indices, len(dataset_barajado)))

#%% 

def compute_predictions_gpt2(data, model, tokenizer, device):
    predictions = []
    model.eval()
    
    for i in range(0, len(data)):
        articles = "Summarize this article:\n"+data["Articles"][i]+ "Summary:\n"

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
        predictions.append(summary)
    
    return predictions

#%%
def compute_predictions_bart(data, model, tokenizer, device):
    predictions = []
    model.eval()
    for i in range(0, len(data)):
        articles = "Sumarize this article:\n"+data["Articles"][i]+ "Summary:\n"

        inputs = tokenizer(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(device)

        summary_ids = model.generate(
            **inputs,
            max_new_tokens=250,
            num_return_sequences=1,
            do_sample=True,
            top_k=5
        )
        
        summaries = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        predictions.append(summaries)
    
    return predictions

#%%

rouge = load("rouge",token=True)

def rouge_evaluate(data, model, tokenizer, device,compute_predictions,references):  
    predictions = compute_predictions(data,model, tokenizer, device)
    rouge_results = rouge.compute(predictions=predictions, references=references)
    return rouge_results

#%%

data = eval_dataset.select(range(6))

references = [example["Summaries"] for example in data]  

rouge_base = rouge_evaluate(data,base_model,base_tokenizer,device,compute_predictions_gpt2,references)
rouge_FT = rouge_evaluate(data,FT_model,FT_tokenizer,device,compute_predictions_gpt2,references)
rouge_lora = rouge_evaluate(data,lora_model,lora_tokenizer,device,compute_predictions_gpt2,references)

#%%
print("_"*100)
print(f"Resultados modelo base: ", rouge_base)
print("_"*100)
print(f"Resultados modelo FullFineTuning: ",rouge_FT)
print("_"*100)
print(f"Resultados modelo Lora: ", rouge_lora)
print("_"*100)

#%%

data = eval_dataset.select(range(30))

references = [example["Summaries"] for example in data]  

rouge_base = rouge_evaluate(data,base_model,base_tokenizer,device,compute_predictions_bart,references)
rouge_FT = rouge_evaluate(data,FT_model,FT_tokenizer,device,compute_predictions_bart,references)
rouge_lora = rouge_evaluate(data,lora_model,lora_tokenizer,device,compute_predictions_bart,references)
  
#%%
print("_"*100)
print(f"Resultados modelo base: ", rouge_base)
print("_"*100)
print(f"Resultados modelo FullFineTuning: ",rouge_FT)
print("_"*100)
print(f"Resultados modelo Lora: ", rouge_lora)
print("_"*100)

#%%  




