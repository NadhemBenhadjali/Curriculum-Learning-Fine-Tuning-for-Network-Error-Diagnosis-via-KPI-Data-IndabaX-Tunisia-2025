# -------------------------------------------------------------
# 1. Imports
# -------------------------------------------------------------
import re
import pickle
import pandas as pd
import torch
import evaluate
from tqdm import tqdm
from math import ceil
from unsloth import FastLanguageModel, is_bfloat16_supported
from unsloth.chat_templates import (
    standardize_sharegpt,
    get_chat_template,
    train_on_responses_only,
)
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
from sklearn.model_selection import train_test_split

# -------------------------------------------------------------
# 2. Load Model & Tokenizer
# -------------------------------------------------------------
max_seq_length = 2048
dtype = None
load_in_4bit = True

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="final_model",
    max_seq_length=max_seq_length,
    dtype=dtype,
    load_in_4bit=load_in_4bit,
    token="your_token_here"  # Replace with your token
)

# -------------------------------------------------------------
# 3. Load & Prepare Training Data
# -------------------------------------------------------------
df = pd.read_csv("train.csv")

def tokenize(text: str) -> list[str]:
    return re.sub(r'[^a-z0-9]', ' ', str(text).lower()).split()

def learn_label_feature_map(df, solution_col="improvement_solutions", label_col="network_labels", pct_threshold=0.7):
    feature_cols = [c for c in df.columns if c not in {solution_col, label_col}]
    sol_tokens = df[solution_col].fillna("").map(tokenize)

    presence = pd.DataFrame(index=df.index, columns=feature_cols, dtype=int)
    for col in feature_cols:
        col_tokens = df[col].astype(str).fillna("").map(tokenize)
        presence[col] = [
            int(bool(set(ft) & set(st)))
            for ft, st in zip(col_tokens, sol_tokens)
        ]

    lab_exp = presence.assign(label=df[label_col].str.split(',').fillna('')).explode('label')
    lab_exp['label'] = lab_exp['label'].str.strip()
    lab_exp = lab_exp.query("label != ''")

    pct = lab_exp.groupby('label')[feature_cols].mean().T

    label_to_feats = {
        lab: pct.index[pct[lab] >= pct_threshold].tolist()
        for lab in pct.columns
    }

    global_top = presence.sum().sort_values(ascending=False).head(5).index.tolist()

    return label_to_feats, global_top

LABEL_TO_FEATS, GLOBAL_TOP_KPIS = learn_label_feature_map(df)

# ✅ Save mapping for test.py
with open("kpi_mappings.pkl", "wb") as f:
    pickle.dump((LABEL_TO_FEATS, GLOBAL_TOP_KPIS), f)

#-------------------------------------------------------------
# 4. Prompt Building
# -------------------------------------------------------------
def to_chat(row: pd.Series) -> dict:
    labels = [lab.strip() for lab in str(row["network_labels"]).split(',') if lab.strip()]
    kpi_cols = {col for lab in labels for col in LABEL_TO_FEATS.get(lab, [])}
    if not kpi_cols:
        kpi_cols = set(GLOBAL_TOP_KPIS)

    kpi_parts = [
        f"{col}: {row[col]}"
        for col in kpi_cols
        if pd.notna(row[col]) and row[col] != ""
    ]
    kpi_str = "; ".join(kpi_parts) if kpi_parts else "No KPI data provided"

    user_prompt = (
        f"Current KPI values — {kpi_str}. "
        f"Explain how to solve these errors: {row['network_labels']}"
    )

    return {
        "conversations": [
            {"role": "user", "content": user_prompt},
            {"role": "assistant", "content": row["improvement_solutions"]}
        ]
    }

# -------------------------------------------------------------
# 5. Dataset Construction
# -------------------------------------------------------------
all_data = list(df.apply(to_chat, axis=1))
train_raw, test_raw = train_test_split(all_data, test_size=0.01, shuffle=True, random_state=42)

tokenizer = get_chat_template(tokenizer, chat_template="llama-3.1")

def add_template(examples):
    txts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=False)
            for x in examples["conversations"]]
    return {"text": txts}

train_ds = Dataset.from_list(train_raw)
test_ds = Dataset.from_list(test_raw)

train_ds = standardize_sharegpt(train_ds).map(add_template, batched=True)
test_ds  = standardize_sharegpt(test_ds).map(add_template, batched=True)

# -------------------------------------------------------------
# 6. Training Setup
# -------------------------------------------------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=train_ds,
    dataset_text_field="text",
    max_seq_length=2048,
    data_collator=DataCollatorForSeq2Seq(tokenizer),
    dataset_num_proc=2,
    packing=False,
    args=TrainingArguments(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        warmup_steps=5,
        num_train_epochs=4,
        learning_rate=1e-7,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=2,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="cosine",
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)

trainer = train_on_responses_only(
    trainer,
    instruction_part="<|start_header_id|>user<|end_header_id|>\n\n",
    response_part="<|start_header_id|>assistant<|end_header_id|>\n\n",
    num_proc=1,
)

trainer.train()

# -------------------------------------------------------------
# 7. Save Model
# -------------------------------------------------------------
save_path = "final_modelA"
model.save_pretrained(save_path)
tokenizer.save_pretrained(save_path)
print("✅ Training finished and saved to:", save_path)

# -------------------------------------------------------------
# 8. 5-Fold ROUGE Evaluation
# -------------------------------------------------------------
FastLanguageModel.for_inference(model)
rouge = evaluate.load("rouge")

test_items = list(test_ds)
torch.manual_seed(42)
test_items = [test_items[i] for i in torch.randperm(len(test_items))]

fold_size = ceil(len(test_items) / 5)
folds = [test_items[i:i + fold_size] for i in range(0, len(test_items), fold_size)]
all_fold_scores = {k: [] for k in ["rouge1", "rouge2", "rougeL", "rougeLsum"]}

for fold_idx, fold_samples in enumerate(folds, start=1):
    print(f"\n🔎 Fold {fold_idx}/5 — {len(fold_samples)} samples")
    predictions, references = [], []

    for sample in tqdm(fold_samples):
        messages = sample["conversations"]
        inputs = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        input_ids = inputs.to("cuda")
        attention_mask = (input_ids != tokenizer.pad_token_id).long()

        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=True,
            top_p=0.9,
            temperature=0.8,
            num_beams=1,
            use_cache=True,
        )

        decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
        prediction = re.split(r"<\|start_header_id\|>assistant<\|end_header_id\|>\s*", decoded)[-1].strip()
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
# 9. Mean ROUGE
# -------------------------------------------------------------
print("\n📈 Mean ROUGE Scores Across 5 Folds:")
for key, values in all_fold_scores.items():
    mean_score = sum(values) / len(values)
    print(f"  {key}: {mean_score:.4f}")
print("\n✅ Evaluation finished.")
