# -*- coding: utf-8 -*-

from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments,AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import torch
from peft import get_peft_model, LoraConfig, TaskType

#%% Cargar el tokenizador y el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device = ", device)

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
base_model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)
#Definimos un token para el padding
tokenizer.pad_token = tokenizer.eos_token

# Actualizar la configuración del modelo para que use el mismo token de padding
base_model.config.pad_token_id = tokenizer.pad_token_id

#%% Configurar LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,  
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
indices = int(0.8 * len(dataset_barajado))

train_dataset = dataset_barajado.select(range(indices))    
eval_dataset = dataset_barajado.select(range(indices, len(dataset_barajado)))

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
    num_train_epochs=2,             
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
##trainer.save_model("C:/Users/pablo/ModelosLLM/GPT2-Lora-FT")
##tokenizer.save_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorGPT2-Lora-FT")

# %%
## Vamos a compararlos usando la metrica ROUGE

# Cargamos el modelo y el tokenizadpor si es necesario

peft_model = AutoModelForCausalLM.from_pretrained("C:/Users/pablo/ModelosLLM/GPT2-Lora-FT").to(device)
perf_tokenizer = AutoTokenizer.from_pretrained("C:/Users/pablo/ModelosLLM/tokenizadorGPT2-Lora-FT")



# %%

def generate_summary(text,model, tokenizer, device):
    articles = "Summarize this article:\n"+text+ "Summary:\n"
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


article = "Summarize this article:\nClimate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change. The time to act is now, as delaying action will only make the problem more difficult and expensive to solve.Summary:\n"
article2= "Climate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also key in the fight against climate change."
print(generate_summary(article,base_model,tokenizer, device))
print("_"*50)
print(generate_summary(article,peft_model,perf_tokenizer, device))


# %%


from evaluate import load

rouge = load("rouge",token=True)

references = [example["Summaries"] for example in eval_dataset]  

# %%
def compute_predictions(data, model, tokenizer, device):
    predictions = []
    
    for i in range(len(data)):
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
        
        summaries = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        predictions.append(summaries)
    
    return predictions

# %%

predictions_lora = compute_predictions(eval_dataset,peft_model, tokenizer, device)
rouge_lora = rouge.compute(predictions=predictions_lora, references=references)
predictions_base = compute_predictions(eval_dataset,base_model, tokenizer, device)
rouge_base = rouge.compute(predictions=predictions_base, references=references)
print("ROUGE del Lora_model:", rouge_lora)
print("ROUGE del base_model:", rouge_base)

# %%
article = "Sumarize this article:\nClimate change is one of the most pressing issues facing humanity today. Scientists have warned that rising global temperatures, caused by increased greenhouse gas emissions, could lead to catastrophic consequences such as extreme weather events, rising sea levels, and the loss of biodiversity. To combat this, countries around the world are adopting measures like transitioning to renewable energy sources, improving energy efficiency, and protecting forests. However, experts emphasize that individual actions, such as reducing waste and using public transportation, are also crucial in the fight against climate change. The time to act is now, as delaying action will only make the problem more difficult and expensive to solve.Summary:\n"

print(generate_summary(eval_dataset[0]["Articles"],base_model))
print("_"*100)
print(generate_summary(eval_dataset[0]["Articles"],peft_model))
