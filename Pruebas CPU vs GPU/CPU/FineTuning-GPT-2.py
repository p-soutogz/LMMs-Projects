# -*- coding: utf-8 -*-

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
import time

#%% Cargar el tokenizador y el modelo
device = "cpu"

print("Device = ", device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
base_model.config.pad_token_id = tokenizer.pad_token_id

#%% Cargar el dataset
dataset = load_dataset('gopalkalpande/bbc-news-summary')

split_dataset = dataset["train"].train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=42
)

train_dataset = split_dataset["train"].select(range(int(0.1 * len(split_dataset["train"]))))
eval_dataset = split_dataset["test"].select(range(int(0.1 * len(split_dataset["test"]))))


#%% Función de preprocesamiento
def preprocess_function(examples):
    inputs = [
        "Summarize this article:\n" + article + "\nSummary:\n" + summary
        for article, summary in zip(examples["Articles"], examples["Summaries"])
    ]
    model_inputs = tokenizer(
        inputs,
        truncation=True,
        max_length=1024,
        padding="max_length",
        return_tensors="pt",
    )
    model_inputs["labels"] = model_inputs["input_ids"].clone()
    for i, input_text in enumerate(inputs):
           summary_start = input_text.find("\nSummary:\n") + len("\nSummary:\n")
           prompt_length = len(tokenizer(input_text[:summary_start], return_tensors="pt")["input_ids"][0])
           model_inputs["labels"][i, :prompt_length] = -100
    return model_inputs

#%% Aplicar preprocesamiento
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["Summaries", "Articles",'File_path'])

#%% Configurar los argumentos de entrenamiento

training_args = TrainingArguments(      
    evaluation_strategy="epoch",    
    learning_rate=3e-5,               
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,  
    num_train_epochs=1,             
    weight_decay=0.01,              
    save_total_limit=2,             
    fp16=False,
    no_cuda=True
)

#%%
trainer = Trainer(
    model=base_model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

#%% Entrenar el modelo
start_time = time.time() 
trainer.train()
end_time = time.time()  

training_duration = end_time - start_time

with open("resultados_entrenamiento_GPT2_FFT_CPU.txt", "w") as archivo:
    archivo.write(f"\nTiempo total de entrenamiento: {training_duration:.2f} segundos\n")
    archivo.write(f"Tiempo por época: {training_duration/training_args.num_train_epochs:.2f} segundos\n")


#%% Guardar el modelo fine-tuned(Modifica tu la ruta)
trainer.save_model("C:/Users/pablo/ModelosLLM/GPT2-FT")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorGPT2-FT")

