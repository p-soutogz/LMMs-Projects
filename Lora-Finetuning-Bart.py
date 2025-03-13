# -*- coding: utf-8 -*-

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

#%% Cargar el tokenizador y el modelo
model_name = "facebook/bart-large-cnn"
device = "cuda" if torch.cuda.is_available() else "cpu"
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
peft_model = get_peft_model(base_model, peft_config)
peft_model.print_trainable_parameters()  

#%% Cargar el dataset
dataset = load_dataset('abisee/cnn_dailymail', '3.0.0')
train_dataset = dataset["train"].select(range(int(0.01 * len(dataset["train"]))))
test_dataset = dataset["test"].select(range(int(0.01 * len(dataset["test"]))))
eval_dataset = dataset["validation"].select(range(int(0.01 * len(dataset["validation"]))))

#%% Función de preprocesamiento
def preprocess_function(examples):
    inputs = ["Summarize this article:\n " + article + "Summary:\n" for article in examples["article"]]
    model_inputs = tokenizer(inputs, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["highlights"], truncation=True, max_length=250, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

#%% Aplicar preprocesamiento
train_dataset = train_dataset.map(preprocess_function, batched=True)
eval_dataset = eval_dataset.map(preprocess_function, batched=True)
test_dataset = test_dataset.map(preprocess_function, batched=True)

train_dataset = train_dataset.remove_columns(["id", "highlights", "article"])

#%% Configurar los argumentos de entrenamiento

training_args = TrainingArguments(      
    evaluation_strategy="epoch",    
    learning_rate=3e-5,               
    per_device_train_batch_size=4,   
    per_device_eval_batch_size=4,  
    num_train_epochs=1,             
    weight_decay=0.01,              
    save_total_limit=2,             
    fp16=torch.cuda.is_available(),  
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
trainer.train()

#%% Guardar el modelo fine-tuned
#trainer.save_model("C:/Users/pablo/ModelosLLM/Bart-Lora-FT")
#tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorBart-Lora-FT")




