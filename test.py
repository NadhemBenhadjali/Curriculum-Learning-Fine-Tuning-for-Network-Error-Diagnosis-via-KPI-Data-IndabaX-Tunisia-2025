from unsloth import FastLanguageModel
from transformers import AutoTokenizer
from tqdm import tqdm
import pandas as pd
import torch
import re
import pickle


# -----------------------------------------------------------------
# 1. Model & Tokenizer
# -----------------------------------------------------------------
save_path = "final_modelA"
model, _ = FastLanguageModel.from_pretrained(save_path, load_in_4bit=True)
tokenizer = AutoTokenizer.from_pretrained(save_path)
FastLanguageModel.for_inference(model)

# -----------------------------------------------------------------
# 2. Data
# -----------------------------------------------------------------
df = pd.read_csv("merged_data_with_network_labels.csv")   # needs ID, network_labels, KPI columns

# -----------------------------------------------------------------
# 3. Load Feature Map (from training step)
# -----------------------------------------------------------------
with open("kpi_mappings.pkl", "rb") as f:
    LABEL_TO_FEATS, GLOBAL_TOP_KPIS = pickle.load(f)

# -----------------------------------------------------------------
# 4. Utilities for Prompt Building
# -----------------------------------------------------------------
def tokenize(text: str) -> list[str]:
    return re.sub(r'[^a-z0-9]', ' ', str(text).lower()).split()

def build_user_prompt(row, label_to_feats, global_top):
    labels = [lab.strip() for lab in str(row["network_labels"]).split(',') if lab.strip()]
    kpi_cols = {col for lab in labels for col in label_to_feats.get(lab, [])}
    if not kpi_cols:
        kpi_cols = set(global_top)

    kpi_parts = [
        f"{col}: {row[col]}"
        for col in kpi_cols
        if pd.notna(row[col]) and row[col] != ""
    ]
    kpi_str = "; ".join(kpi_parts) if kpi_parts else "No KPI data provided"

    return (
        f"Current KPI values — {kpi_str}. "
        f"Explain how to solve these errors: {row['network_labels']}"
    )

# -----------------------------------------------------------------
# 5. Generation Loop
# -----------------------------------------------------------------
batch_size = 16
targets = []

for start in tqdm(range(0, len(df), batch_size), desc="Generating"):
    batch_df = df.iloc[start:start + batch_size]

    # 5a) Build prompts per row
    messages_batch = [
        [{"role": "user",
          "content": build_user_prompt(row, LABEL_TO_FEATS, GLOBAL_TOP_KPIS)}]
        for _, row in batch_df.iterrows()
    ]

    # 5b) Tokenize input prompts with chat template
    id_lists = [
        tokenizer.apply_chat_template(
            m, tokenize=True, add_generation_prompt=True
        )
        for m in messages_batch
    ]

    # 5c) Pad and move inputs to GPU
    inputs = tokenizer.pad(
        {"input_ids": id_lists},
        padding=True,
        return_tensors="pt"
    )
    input_ids = inputs["input_ids"].to("cuda")
    attention_mask = (input_ids != tokenizer.pad_token_id).long()
    # 5d) Generate responses
    outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=256,
            do_sample=False,
            top_p=0.9,
            temperature=0.8,
            num_beams=1,
            use_cache=True,
        )

    # 5e) Decode and extract assistant response
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    for txt in decoded:
        answer = txt.split("assistant", 1)[-1].strip()
        answer = re.sub(r"^\s*\d+[\.\)]\s*", "", answer)
        targets.append(answer)

# -----------------------------------------------------------------
# 6. Save CSV
# -----------------------------------------------------------------
pd.DataFrame({"ID": df["ID"], "Target": targets}).to_csv(
    "final_submission.csv", index=False
)

print("✅  Submission file saved as final_submission.csv")
print("✅  Evaluation finished.")
