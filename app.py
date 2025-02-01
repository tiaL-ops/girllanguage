import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments

# Enable MPS fallback for unsupported operations on Apple Silicon (if using MPS)
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

# Choose device: if MPS is available, use it; otherwise, use CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using MPS device")
else:
    device = torch.device("cpu")
    print("Using CPU device")


df = pd.read_csv("reddit_data.csv")
df = df.dropna(subset=["post_text", "comment"])


if "post_text" not in df.columns or "comment" not in df.columns:
    raise ValueError("CSV file must contain 'post_text' (question) and 'comment' (answer) columns.")


tokenizer = T5Tokenizer.from_pretrained("t5-small")

class RedditQADataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
       
        self.questions = dataframe["post_text"].tolist()  
        self.answers = dataframe["comment"].tolist()        
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        # Tokenize the question
        encoding = self.tokenizer(
            self.questions[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        # Tokenize the answer (target)
        target_encoding = self.tokenizer(
            self.answers[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "labels": target_encoding["input_ids"].squeeze(0),
        }

# Split the dataset (using 80/20 split)
train_size = int(0.8 * len(df))
train_df = df.iloc[:train_size]
eval_df = df.iloc[train_size:]

train_dataset = RedditQADataset(train_df, tokenizer)
eval_dataset = RedditQADataset(eval_df, tokenizer)

model = T5ForConditionalGeneration.from_pretrained("t5-small").to(device)

training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    num_train_epochs=3,
    logging_dir="./logs",
    eval_strategy="epoch",
    save_strategy="epoch",
   
)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)


trainer.train()

def generate_answer(question):
  
    inputs = tokenizer(question, return_tensors="pt", truncation=True, max_length=128).to(device)
   
    outputs = model.generate(inputs.input_ids)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

sample_question = "My boyfriend is acting distant, what should I do?"
print("Generated Answer:", generate_answer(sample_question))
