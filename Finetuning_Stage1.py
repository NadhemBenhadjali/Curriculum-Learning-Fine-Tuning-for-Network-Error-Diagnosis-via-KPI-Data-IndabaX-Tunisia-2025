# !pip install evaluate
# !pip install unsloth
# -------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import (
    standardize_sharegpt,
    get_chat_template,
    train_on_responses_only,
)
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import torch
import evaluate
from tqdm import tqdm

# -------------------------------------------------------------
# 2. Model & Tokenizer
# -------------------------------------------------------------
max_seq_length = 2048
dtype           = None
load_in_4bit    = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = "unsloth/Llama-3.2-3B-Instruct",
    max_seq_length = max_seq_length,
    dtype          = dtype,
    load_in_4bit   = load_in_4bit,
    token          = "your_token_here"  # Replace with your token
)

model = FastLanguageModel.get_peft_model(
    model,
    r                       = 16,
    target_modules          = ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    lora_alpha              = 16,
    lora_dropout            = 0,
    bias                    = "none",
    use_gradient_checkpointing = "unsloth",
    random_state            = 3407,
)

# -------------------------------------------------------------
# 3. Load & prepare data
# -------------------------------------------------------------
df = pd.read_csv("/kaggle/input/indabax-tunisia-2025-anomaly-solver-challenge-2/Train.csv")

def to_chat(row):
    return {
        "conversations": [
            { "role": "user", "content": f"Explain how to solve these errors: {row['network_labels']}" },
            { "role": "assistant", "content": row['improvement_solutions'] }
        ]
    }

all_data = list(df.apply(to_chat, axis=1))

# Split: 80% train, 20% test — no validation
train_raw, test_raw = train_test_split(all_data, test_size=0.1, shuffle=True, random_state=42)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def add_template(examples):
    txts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
            for x in examples["conversations"]]
    return {"text": txts}

train_ds = Dataset.from_list(train_raw)
test_ds  = Dataset.from_list(test_raw)

train_ds = standardize_sharegpt(train_ds).map(add_template, batched=True, num_proc=1)
test_ds  = standardize_sharegpt(test_ds).map(add_template, batched=True, num_proc=1)

# -------------------------------------------------------------
# 4. Trainer
# -------------------------------------------------------------
trainer = SFTTrainer(
    model = model,
    tokenizer = tokenizer,
    train_dataset = train_ds,
    dataset_text_field = "text",
    max_seq_length = 2048,
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer),
    dataset_num_proc = 2,
    packing = False,
    args = TrainingArguments(
        per_device_train_batch_size = 4,
        gradient_accumulation_steps = 4,
        warmup_steps = 5,
        num_train_epochs=4,
        learning_rate = 2e-4,
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        logging_steps = 20,
        optim = "adamw_8bit",
        weight_decay = 0.01,
        lr_scheduler_type = "linear",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
    response_part    = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    num_proc         = 1,
)

trainer.train()
save_path = "final_model"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)

print("✅ Training finished.")

# -------------------------------------------------------------
# 5. Simple 5-Fold ROUGE Evaluation (on test set only)
# -------------------------------------------------------------
from math import ceil

FastLanguageModel.for_inference(model)
rouge = evaluate.load("rouge")

test_items = list(test_ds)
torch.manual_seed(42)
test_items = [test_items[i] for i in torch.randperm(len(test_items))]  # Shuffle

# Split test set into 5 equal chunks
fold_size = ceil(len(test_items) / 5)
folds = [test_items[i:i + fold_size] for i in range(0, len(test_items), fold_size)]

all_fold_scores = { "rouge1": [], "rouge2": [], "rougeL": [], "rougeLsum": [] }

for fold_idx, fold_samples in enumerate(folds, start=1):
    print(f"\n🔎 Fold {fold_idx}/5 — {len(fold_samples)} samples")

    predictions, references = [], []

    for sample in tqdm(fold_samples):
        messages = sample["conversations"]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_tensors="pt"
        )
        input_ids = inputs.to("cuda")
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=128,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_beams=1,
            use_cache=True,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = decoded.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
        reference = messages[1]["content"].strip()

        predictions.append(prediction)
        references.append(reference)

        if len(predictions) <= 3:
            print("👤", messages[0]["content"][:80], "…")
            print("🤖", prediction)
            print("✅", reference)
            print("-" * 60)

    scores = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    for key in all_fold_scores:
        all_fold_scores[key].append(scores[key])

    print(f"\n📊 ROUGE Scores for Fold {fold_idx}:")
    for key, val in scores.items():
        print(f"  {key}: {val:.4f}")

# -------------------------------------------------------------
# 6. Mean ROUGE Score Across 5 Folds
# -------------------------------------------------------------
print("\n📈 Mean ROUGE Scores Across 5 Folds:")
for key, values in all_fold_scores.items():
    mean_score = sum(values) / len(values)
    print(f"  {key}: {mean_score:.4f}")
print("\n✅ Evaluation finished.")