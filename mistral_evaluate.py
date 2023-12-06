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
import pandas as pd
from textblob import TextBlob

# mistral
from transformers import AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM, prepare_model_for_kbit_training, PeftConfig, get_peft_model

output_dir = './content/positive-frames/epoch0-1/mistral-rw-7b/'
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
dataset = PositiveDataset("./data", phase='test', tokenizer=tokenizer)
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

    input_tokens = tokenizer(input_text, return_tensors='pt', truncation=True, max_length=512, padding="max_length")
    input_ids = input_tokens['input_ids'].to('cuda')

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
    
    bleu_preds.append([pred])
    bleu_gts.append(label_text if len(label_text)==1 else label_text)

    ##### rouge #####
    rouge_preds.append(pred)
    rouge_gts.append(label_text[0])
  
f1.close()
f2.close()

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

f3 = open('./mistral_score3.txt', 'w')

rouge_scores = rouge.compute(predictions=preds, references=gts)
print("Mistral7B Rouge: ", rouge_scores)
f3.write(str(rouge_scores) + '\n')

###### BLUE Score #####
sacrebleu = evaluate.load("sacrebleu")
sacrebleu_scores = sacrebleu.compute(predictions=preds, references=gts)
print("Mistral7B SacreBleu: ", sacrebleu_scores)
f3.write("Mistral7B SacreBleu: "+ str(sacrebleu_scores))

###### bertscore #####
bertscore = evaluate.load("bertscore")
bertscore_scores = bertscore.compute(predictions=preds, references=gts, lang='en')['f1']
print("Mistral7B bertscore: ", bertscore_scores)
f3.write("Mistral7B bertscore: "+ str(bertscore_scores))

###### textblob #####
data = pd.read_csv('./data/wholetest.csv')
inputs = list(data['original_text'].values)

pred_sentiment_scores = []
input_sentiment_scores = []
assert len(preds) == len(gts)
for i in range(len(preds)):
    blob = TextBlob(preds[i])
    pred_sentiment_scores.append(blob.sentences[0].sentiment.polarity)
    blob = TextBlob(inputs[i])
    input_sentiment_scores.append(blob.sentences[0].sentiment.polarity)
pred_sent_scores = np.array(pred_sentiment_scores)
input_sent_scores = np.array(input_sentiment_scores)
deltas = pred_sent_scores - input_sent_scores
avg_delta = deltas.mean()
print("Mistral7B textblob delta score: ", avg_delta)
f3.write("Mistral7B textblob delta score: "+ str(avg_delta))

f1.close()
f2.close()
f3.close()
