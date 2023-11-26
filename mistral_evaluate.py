import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from tqdm import tqdm
import wandb
from transformers import BartModel, BartTokenizer, BartForConditionalGeneration, TrainingArguments, AdamW
from datasets import load_metric
from dataset import PositiveDataset
from config import config
from transformers import AutoTokenizer
import nltk
nltk.download('punkt')

# mistral
from transformers import AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, PeftConfig, get_peft_model

'''
output_dir = '/home/jimin/PycharmProjects/positive-frames-main/content/positive-frames/epoch0-1/mistral-rw-7b/'
model = AutoPeftModelForCausalLM.from_pretrained(output_dir+"/output/reframer", load_in_4bit=True)
config = PeftConfig.from_pretrained(output_dir+"/output/reframer")
model = prepare_model_for_kbit_training(model)
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.bos_token = "<|startoftext|>"

peft_config = LoraConfig(
    r=16,
    lora_alpha=16,
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )
model = get_peft_model(model, peft_config).to('cuda')
model.eval()

# dataset
dataset = PositiveDataset("/home/jimin/PycharmProjects/positive-frames-main/data", phase='test', tokenizer=tokenizer)
dataloader = DataLoader(dataset, batch_size=1) # test batch size=1

metric = load_metric("rouge")
metric2 = load_metric('sacrebleu')

rouge_preds = []
rouge_gts = []
bleu_preds = []
bleu_gts = []

f1 = open('./mistral_pred.txt', 'w')
f2 = open('./mistral_gt.txt', 'w')
for i, batch in tqdm(enumerate(dataloader)):
    input_text = batch['original_text']
    label_text = batch['reframed_text']
    strategy = batch['strategy']

    # input_tokens = tokenizer(input_text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to('cuda')
    input_tokens = tokenizer(input_text, return_tensors='pt', truncation=True,max_length=512, padding="max_length")
    input_ids = input_tokens['input_ids'].to('cuda')

    # output_ids = model.generate(input_ids, num_beams=peft_config.num_beams, min_length=0, max_length=128)
    output_ids = model.generate(**input_tokens, max_new_tokens=512)
    output_text = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    # extract only 'reframe' from output_text
    lines = output_text.split('\n')
    c = True
    for line in lines:
        if 'reframed' in line and c == True:
            pred = line.replace("reframed: ", '')
            pred = pred.replace("<endoftext>", '')
            c = False
    f1.write(pred + '\n')
    f2.write(label_text[0] + '\n')
    print("pred: ", pred, ", gt: ", label_text)
    bleu_preds.append([pred])
    bleu_gts.append(label_text if len(label_text)==1 else label_text)

    ##### rouge #####
    rouge_preds.append(pred)
    rouge_gts.append(label_text[0])

    
    # rouge test
    #decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in rouge_preds]
    #decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in rouge_gts]

    #result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
    #print("result: ", result)
    # bleu test
    #bleu_scores = metric2.compute(predictions=bleu_preds, references=bleu_gts)['score']
    #print("bleu: ", bleu_scores)
    
f1.close()
f2.close()
'''
##### Rouge Score #####
import evaluate
f1 = open('mistral_pred.txt', 'r')
f2 = open('mistral_gt.txt', 'r')

preds = []
gts = []

lines = f1.readlines()
for line in lines:
    preds.append(line)

lines2 = f2.readlines()
for line in lines2:
    gts.append(line)

rouge = evaluate.load('rouge')
# metric2 = load_metric('sacrebleu')

f3 = open('./mistral_score3.txt', 'w')

# Rouge expects a newline after each sentence
# decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in rouge_preds]
# decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in rouge_gts]
# result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)

rouge_scores = rouge.compute(predictions=preds, references=gts)
print("Mistral7B Rouge: ", rouge_scores)
f3.write(str(rouge_scores) + '\n')

###### BLUE Score #####
# bleu_scores = metric2.compute(predictions=bleu_preds, references=bleu_gts)['score']

sacrebleu = evaluate.load("sacrebleu")
sacrebleu_scores = sacrebleu.compute(predictions=preds, references=gts)
print("Mistral7B SacreBleu: ", sacrebleu_scores)
f3.write("Mistral7B SacreBleu: "+ str(sacrebleu_scores))

bleu = evaluate.load("bleu")
bleu_scores = bleu.compute(predictions=preds, references=gts)
print("Mistral7B Bleu: ", bleu_scores)
f3.write("Mistral7B Bleu: "+ str(bleu_scores))

f1.close()
f2.close()
f3.close()