import argparse
import pandas as pd
import numpy as np
import torch
import random
from transformers import DataCollatorForLanguageModeling
import os
from transformers import Trainer, pipeline, set_seed, AutoModelForSeq2SeqLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainingArguments
import nltk
nltk.download('punkt')
import csv
import transformers
from datasets import load_dataset, load_metric
from peft import LoraConfig,PeftConfig,get_peft_model,prepare_model_for_kbit_training
from transformers import AutoConfig,AutoModelForCausalLM,AutoTokenizer,BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM

from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM
from transformers import TextDataset

# Dataset Load
from datasets import load_dataset
root = "./content/positive-frames/"
train_path = root + "./data/wholetrain_gpt.txt"
dev_path =  root+"./data/wholedev.csv"
dev_dataset = load_dataset('csv', data_files=dev_path)
test_path =  root + "./data/wholetest.csv"
test_dataset = load_dataset('csv', data_files=test_path)

def tokenize(prompt):
    result = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        padding="max_length",
    )
    result["labels"] = result["input_ids"].copy()
    return result

def generate_and_tokenize_prompt(data_point):
    full_prompt =f"""Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

    ### Target sentence:
    {data_point["target"]}

    ### Meaning representation:
    {data_point["meaning_representation"]}
    """
    return tokenize(full_prompt)

def load_dataset(train_path,test_path,tokenizer):
    train_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=train_path,
        block_size=50)
    test_dataset = TextDataset(
        tokenizer=tokenizer,
        file_path=test_path,
        block_size=50)
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False,
        )
    return train_dataset,test_dataset,data_collator

def run_mistral_unconstrained(name="s"):
    model_name = base_model_id = "mistralai/Mistral-7B-v0.1"
    output_dir = root+"mistral-rw-7b"
    f_name = "mistral_rw_7b_predict.txt"

    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      load_4bit_use_double_quant=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.bfloat16, # float16 -> bfloat16
    )

    ##### Tokenizer #####
    tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        model_max_length=512,
        padding_side="left",
        add_eos_token=True)

    tokenizer.pad_token = tokenizer.eos_token

    ##### model #####
    model =AutoModelForCausalLM.from_pretrained(
      base_model_id, quantization_config=bnb_config
    )
    model = prepare_model_for_kbit_training(model)
    config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=[
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, config)


    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    from transformers import TextDataset

    train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)

    batch_size = 3
    optim = "paged_adamw_32bit"
    per_device_train_batch_size = batch_size
    gradient_accumulation_steps = batch_size
    # save_steps = 10
    logging_steps = 10
    learning_rate = 2e-4
    max_grad_norm = 0.3
    max_steps = 5
    warmup_ratio = 0.03
    lr_scheduler_type = "constant"
    args = transformers.TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=per_device_train_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        optim=optim,
        learning_rate=learning_rate,
        evaluation_strategy="epoch",
        do_train=True,
        do_eval=True,
        logging_steps=1024,
        save_steps=2048,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        lr_scheduler_type=lr_scheduler_type,
        num_train_epochs = 5,
        overwrite_output_dir=True,
        save_total_limit=5,
        fp16=True,
    )


    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Rouge expects a newline after each sentence
        decoded_preds_joined = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels_joined = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

        result = metric.compute(predictions=decoded_preds_joined, references=decoded_labels_joined, use_stemmer=True)
        # Extract a few results
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

        # Add mean generated length
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in predictions]
        result["gen_len"] = np.mean(prediction_lens)

        decoded_labels_expanded = [[x] for x in decoded_labels]
        result2 = metric2.compute(predictions=decoded_preds, references=decoded_labels_expanded)

        # print(result2)
        result['sacrebleu'] = round(result2["score"], 1)

        return {k: round(v, 4) for k, v in result.items()}

    trainer = transformers.Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,

    )
    model.config.use_cache = False
    trainer.train()

    trainer.evaluate()
    
    # save model
    trainer.save_model(output_dir+"/output/reframer")

    model = AutoPeftModelForCausalLM.from_pretrained(output_dir+"/output/reframer", load_in_4bit=True)
    config = PeftConfig.from_pretrained(output_dir+"/output/reframer")
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|startoftext|>"

    reframer = pipeline('text-generation', model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id,  pad_token_id=tokenizer.eos_token_id)
    
    import csv
    with open (test_path, newline="") as data:
      annotations = csv.DictReader(data, delimiter=',', quotechar='"')
      annotations_list = list(annotations)
      reframed_phrases = []
      answer_phrases = []
      for i in range(len(annotations_list)):
          prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
          gen_text = reframer(prefix, max_length=100)[0]['generated_text']
          reframed_phrases.append(gen_text)
          answer_phrases.append(annotations_list[i]['reframed_text'])

    with open(os.path.join(root, f_name), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

    with open(os.path.join(root, "total_reframe.txt"),'w') as f:
      for item in answer_phrases:
        f.write("%s\n"%item)

if __name__=='__main__':
    ##### Train #####
    print("Train Mistral start!")
    run_mistral_unconstrained()
