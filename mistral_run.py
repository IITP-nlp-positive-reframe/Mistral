import argparse
import pandas as pd
import numpy as np
import torch
import random
# from sentence_transformers import SentenceTransformer, util
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
root = "/content/positive-frames/"
train_path = root + "data/wholetrain_gpt.txt"
# train_path = root + "data/wholetrain.csv"
# train_dataset = load_dataset('csv', data_files=train_path)
dev_path =  root+"data/wholedev.csv"
dev_dataset = load_dataset('csv', data_files=dev_path)
test_path =  root + "data/wholetest.csv"
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

    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.bos_token = "<|startoftext|>"

    metric = load_metric("rouge")
    metric2 = load_metric("sacrebleu")

    from transformers import TextDataset

    # def generate_tokenized_text(examples):
    #   print(examples)
    #   tok_full_prompt = tokenizer(examples, padding="max_length", max_length=1024, truncation=True)
    #   return tok_full_prompt
    # train_dataset = np.read_txt(train_path)
    # train_dataset = train_dataset.map(generate_tokenized_text, batched=True)
    # test_dataset = load_dataset("csv",dataset_files=test_path)

    train_dataset,test_dataset,data_collator = load_dataset(train_path,test_path,tokenizer)
    # def process_test_dataset(examples):
    #   inputs =  "<|startoftext|> " + examples['original_text']+ "\nreframed:"
    #   model_inputs = tokenizer(inputs, padding="max_length",truncation=True, max_length = 1024)
    #   with tokenizer.as_target_tokenizer():
    #     labels = tokenizer(examples["reframed_text"], padding = "max_length",max_length = 1024, truncation=True) #, max_length=max_target_length, truncation=True)
    #   model_inputs["labels"] = labels["input_ids"]
    #   return model_inputs



    # def preprocess_function(examples):
    #     inputs = examples["original_text"]
    #     model_inputs = tokenizer(inputs, padding = "max_length", truncation=True) # max_length=max_input_length, truncation=True)
    #     # Setup the tokenizer for targets
    #     with tokenizer.as_target_tokenizer():
    #         labels = tokenizer(examples["reframed_text"], padding = "max_length", truncation=True) #, max_length=max_target_length, truncation=True)
    #     model_inputs["labels"] = labels["input_ids"]
    #     return model_inputs
    # tokenized_train_datasets = train_dataset.map(preprocess_function, batched=True)
    # # tokenized_dev_datasets = dev_dataset.map(preprocess_function, batched=True)
    # tokenized_test_datasets = test_dataset.map(preprocess_function, batched=True)
    # data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

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
        # per_device_train_batch_size=batch_size,
        # per_device_eval_batch_size=batch_size,
        evaluation_strategy="epoch",    # no for no testing
        do_train=True,
        do_eval=True,
        logging_steps=1024,
        save_steps=2048,
        # warmup_steps=1024,
        max_grad_norm=max_grad_norm,
        warmup_ratio=warmup_ratio,
        # group_by_length=True,
        lr_scheduler_type=lr_scheduler_type,
        #max_steps=1500, # delete for full training
        num_train_epochs = 5, #TRAIN_EPOCHS
        overwrite_output_dir=True,
        save_total_limit=5,
        fp16=True,
        # bf16=True
        ###### below is original for peft training
      # auto_find_batch_size=True,
      # num_train_epochs=4,
      # learning_rate=2e-4,
      # bf16=True,
      # save_total_limit=4,
      # logging_steps=10,
      # output_dir=OUTPUT_DIR,
      # save_strategy='epoch',
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

    # from trl import SFTTrainer
    # # Why not working?
    # trainer = SFTTrainer(
    #     model=model,
    #     train_dataset=train_dataset,
    #     # test_dataset = test_dataset,
    #     peft_config=config,
    #     data_collator=data_collator,
    #     # dataset_text_field="text",
    #     # max_seq_length=max_seq_length,
    #     tokenizer=tokenizer,
    #     args=args,
    # )
    trainer = transformers.Trainer(
        model=model,
        args=args,
        data_collator=data_collator,
        # train_dataset=tokenized_train_datasets["train"],
        # eval_dataset=tokenized_test_datasets["train"],
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        # compute_metrics=compute_metrics

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
    # #prediction
    # model = AutoPeftModelForCausalLM.from_pretrained(output_dir+"/output/reframer", load_in_4bit=True)
    # model = prepare_model_for_kbit_training(model)

    # tokenizer = AutoTokenizer.from_pretrained(output_dir+"/output/reframer")
    # reframer = pipeline('text-generation', model=model, tokenizer=tokenizer,eos_token_id=tokenizer.eos_token_id)
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

    # test = pd.read_csv(test_path)
    # texts = test['original_text'].to_list()
    # reframed_phrases = [reframer(phrase)[0]['generated_text'] for phrase in texts]

    with open(os.path.join(root, f_name), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

    with open(os.path.join(root, "total_reframe.txt"),'w') as f:
      for item in answer_phrases:
        f.write("%s\n"%item)

if __name__=='__main__':
    ##### Train #####
    # print("Train Mistral start!")
    # run_mistral_unconstrained()

    ##### Falcon Inference #####
    '''
    print("Inference Falcon start!")
    # # is it okay to use this?
    
    # tokenizer = AutoTokenizer.from_pretrained('Rocketknight1/falcon-rw-1b')
    model = AutoPeftModelForCausalLM.from_pretrained("/content/positive-frames/falcon-rw-1b/output/reframer",
                                                     load_in_4bit=True)
    config = PeftConfig.from_pretrained("/content/positive-frames/falcon-rw-1b/output/reframer")
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|startoftext|>"

    reframer = pipeline('text-generation', model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id)
    '''

    ##### Mistral Inference #####
    print("Inference Mistral start!")

    # model = AutoPeftModelForCausalLM.from_pretrained("/content/positive-frames/falcon-rw-1b/output/reframer",
                                                     load_in_4bit=True)
    config = PeftConfig.from_pretrained("/content/positive-frames/mistral-rw-7b/output/reframer")
    model = prepare_model_for_kbit_training(model)
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.bos_token = "<|startoftext|>"

    reframer = pipeline('text-generation', model=model, tokenizer=tokenizer, eos_token_id=tokenizer.eos_token_id,
                        pad_token_id=tokenizer.eos_token_id)

    '''
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

    base_model_id = "mistralai/Mistral-7B-v0.1"
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_id,  # Mistral, same as before
        quantization_config=bnb_config,  # Same quantization config as before
        device_map="auto",
        trust_remote_code=True,
        use_auth_token=True
    )
    
    eval_tokenizer = AutoTokenizer.from_pretrained(
        base_model_id,
        add_bos_token=True,
        trust_remote_code=True,
    )

    eval_prompt = """Given a target sentence construct the underlying meaning representation of the input sentence as a single function with attributes and attribute values.
    This function should describe the target string accurately and the function must be one of the following ['inform', 'request', 'give_opinion', 'confirm', 'verify_attribute', 'suggest', 'request_explanation', 'recommend', 'request_attribute'].
    The attributes must be one of the following: ['name', 'exp_release_date', 'release_year', 'developer', 'esrb', 'rating', 'genres', 'player_perspective', 'has_multiplayer', 'platforms', 'available_on_steam', 'has_linux_release', 'has_mac_release', 'specifier']

    ### Target sentence:
    Earlier, you stated that you didn't have strong feelings about PlayStation's Little Big Adventure. Is your opinion true for all games which don't have multiplayer?

    ### Meaning representation:
    """

    model_input = tokenizer(eval_prompt, return_tensors="pt").to("cuda")

    ft_model.eval()
    with torch.no_grad():
        print(eval_tokenizer.decode(ft_model.generate(**model_input, max_new_tokens=100)[0], skip_special_tokens=True))
    '''
    import csv

    with open(test_path, newline="") as data:
        annotations = csv.DictReader(data, delimiter=',', quotechar='"')
        print(annotations)
        # for item in annotations:
        #   print(item)
        annotations_list = list(annotations)
        reframed_phrases = []
        answer_phrases = []
        for i in range(len(annotations_list)):
            prefix = "<|startoftext|> " + annotations_list[i]['original_text'] + "\nreframed:"
            gen_text = reframer(prefix, max_length=100)[0]['generated_text']
            reframed_phrases.append(gen_text)
            answer_phrases.append(annotations_list[i]['reframed_text'])

    # test = pd.read_csv(test_path)
    # texts = test['original_text'].to_list()
    # reframed_phrases = [reframer(phrase)[0]['generated_text'] for phrase in texts]

    with open(os.path.join(root, "output_temp.txt"), 'w') as f:
        for item in reframed_phrases:
            f.write("%s\n" % item)

    with open(os.path.join(root, "total_reframe.txt"), 'w') as f:
        for item in answer_phrases:
            f.write("%s\n" % item)