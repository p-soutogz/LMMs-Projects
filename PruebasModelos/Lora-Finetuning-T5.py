# -*- coding: utf-8 -*-

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

#%% Cargar el tokenizador y el modelo
model_name = "google/flan-t5-small"
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
dataset = load_dataset('gopalkalpande/bbc-news-summary')

dataset_barajado = dataset["train"].shuffle(seed=42)
indices = int(0.7 * len(dataset_barajado))

train_dataset = dataset_barajado.select(range(indices))    
eval_dataset = dataset_barajado.select(range(indices, len(dataset_barajado)))

#%% Función de preprocesamiento
def preprocess_function(examples):
    inputs = ["Summarize this article:\n " + article + "Summary:\n" for article in examples["Articles"]]
    model_inputs = tokenizer(inputs, truncation=True, max_length=1024, padding="max_length", return_tensors="pt")
    labels = tokenizer(examples["Summaries"], truncation=True, max_length=250, padding="max_length", return_tensors="pt")
    model_inputs["labels"] = labels["input_ids"]
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
    fp16=torch.cuda.is_available(),  
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
trainer.train()

#%% Guardar el modelo fine-tuned
trainer.save_model("C:/Users/pablo/ModelosLLM/T5-Lora-FT")
tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorT5-Lora-FT")

# %%
## Vamos a compararlos usando la metrica ROUGE

# Cargamos el modelo y el tokenizadpor si es necesario

peft_model = AutoModelForSeq2SeqLM.from_pretrained("C:/Users/pablo/ModelosLLM/T5-Lora-FT").to(device)
peft_tokenizer = AutoTokenizer.from_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorT5-Lora-FT")


# %%


from evaluate import load

rouge = load("rouge",token=True)

references = [example["Summaries"] for example in eval_dataset]  

# %%
def compute_predictions(data, model, tokenizer, device):
    predictions = []
    
    for i in range(0, len(data)):
        articles = "Sumarize this article:\n"+data["Articles"][i]+ "Summary:\n"

        inputs = tokenizer(
            articles,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=1024
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

# %%

predictions_lora = compute_predictions(eval_dataset,peft_model, peft_tokenizer, device)
rouge_lora = rouge.compute(predictions=predictions_lora, references=references)
predictions_base = compute_predictions(eval_dataset,base_model, tokenizer, device)
rouge_base = rouge.compute(predictions=predictions_base, references=references)
print("ROUGE del Lora_model:", rouge_lora)
print("ROUGE del base_model:", rouge_base)

# %%