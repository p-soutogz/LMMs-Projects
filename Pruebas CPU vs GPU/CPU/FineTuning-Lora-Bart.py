# -*- coding: utf-8 -*-

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType
import time

#%% Cargar el tokenizador y el modelo
model_name = "facebook/bart-large-cnn"

device = "cpu"

print("Device = ", device)

tokenizer = AutoTokenizer.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)


#%% Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,  
    inference_mode=False,             
    r=8,                            
    lora_alpha=32,                   
    lora_dropout=0.1,                
)

#%% Aplicar LoRA al modelo
peft_model = get_peft_model(base_model, peft_config).to(device)
peft_model.print_trainable_parameters()  

#%% Cargar el dataset
dataset = load_dataset('gopalkalpande/bbc-news-summary')

dataset = load_dataset('gopalkalpande/bbc-news-summary')

split_dataset = dataset["train"].train_test_split(
    test_size=0.2,
    shuffle=True,
    seed=42
)

train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# %%

def preprocess_function(examples):
    # Combinar el artículo y los highlights con un token especial
    inputs = ["Summarize this article:\n "+ article + "Summary:\n" for article in examples["Articles"]]

    # Tokenizar los inputs y los targets
    model_inputs = tokenizer(inputs, truncation=True,max_length=1024, padding="max_length",return_tensors="pt")
    labels = tokenizer(examples["Summaries"], truncation=True,max_length=250, padding="max_length",return_tensors="pt")

    
    # Asignar los labels al diccionario de inputs
    model_inputs["labels"] = labels["input_ids"]
    
    return model_inputs

# Aplicar la función de preprocesamiento al dataset
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["File_path", "Summaries","Articles"])


#%% Configurar los argumentos de entrenamiento

training_args = TrainingArguments(      
    evaluation_strategy="epoch",    
    learning_rate=3e-5,               
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,  
    num_train_epochs=2,             
    weight_decay=0.01,              
    save_total_limit=2,             
    fp16=False,
    no_cuda=True
)

#%%
trainer = Trainer(
    model=peft_model,
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

with open("resultados_entrenamiento_Bart_Lora_CPU.txt", "w") as archivo:
    archivo.write(f"\nTiempo total de entrenamiento: {training_duration:.2f} segundos\n")
    archivo.write(f"Tiempo por época: {training_duration/training_args.num_train_epochs:.2f} segundos\n")

#%% Guardar el modelo lora-fine-tuned(Modifica tu la ruta)
trainer.save_model("C:/Users/pablo/ModelosLLM/Bart-Lora-FT")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart-Lora-FT")




