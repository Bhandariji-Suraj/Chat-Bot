# train_model.py
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, Trainer, TrainingArguments
import pandas as pd
from preprocess import load_and_clean_data

def prepare_data(file_path):
    # Load and clean data
    conversations = load_and_clean_data(file_path)
    dialogues = [f"{row['User']} {tokenizer.eos_token} {row['Response']}" for _, row in conversations.iterrows()]
    return dialogues

class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, tokenizer, dialogues, max_length=512):
        self.tokenizer = tokenizer
        self.input_ids = [tokenizer.encode(d, return_tensors="pt", max_length=max_length, truncation=True) for d in dialogues]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {'input_ids': self.input_ids[idx].squeeze()}

def fine_tune_gpt(file_path):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    dialogues = prepare_data(file_path)
    dataset = CustomDataset(tokenizer, dialogues)

    # Training parameters
    training_args = TrainingArguments(
        output_dir="./gpt2_finetuned",
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=2,
        save_steps=10_000,
        save_total_limit=2,
        logging_dir='./logs',
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
    )

    trainer.train()
    model.save_pretrained("./gpt2_finetuned")
    tokenizer.save_pretrained("./gpt2_finetuned")

if __name__ == "__main__":
    file_path = 'data/dialogs.txt'
    fine_tune_gpt(file_path)
