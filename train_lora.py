import os
import json
import pandas as pd
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model

# ===============================
# 0. 환경설정
# ===============================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

MODEL_NAME = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"
DATA_FILE = "./external_data.csv"   # question,answer 형식 CSV
OUTPUT_DIR = "./lora-output"

# ===============================
# 1. CSV → JSONL 변환 (UTF-8, JSONL 포맷 준수)
# ===============================
df = pd.read_csv(DATA_FILE, encoding="utf-8")  # CSV UTF-8 강제

jsonl_path = "train_sft.jsonl"
with open(jsonl_path, "w", encoding="utf-8") as f:
    for _, row in df.iterrows():
        prompt = (
            "당신은 금융보안 전문가입니다.\n"
            "다음 질문에 대해 정확하고 자세히 설명하세요.\n\n"
            f"질문: {row['question']}\n\n답변:"
        )
        output = row["answer"]
        json.dump({"instruction": prompt, "output": output}, f, ensure_ascii=False)
        f.write("\n")

print(f"✅ JSONL 생성 완료: {jsonl_path}, 총 {len(df)}개 샘플")

# ===============================
# 2. 데이터셋 로드
# ===============================
dataset = load_dataset("json", data_files=jsonl_path)

# ===============================
# 3. 모델 & 토크나이저 로드 (4bit)
# ===============================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # causal LM padding

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)

# ===============================
# 4. LoRA 구성
# ===============================
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","v_proj"],  # attention 부분만 LoRA 적용
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ===============================
# 5. 토크나이징 (CausalLM용 마스킹 적용)
# ===============================
IGNORE_INDEX = -100

def preprocess(batch):
    input_texts = []
    labels = []
    for instr, output in zip(batch["instruction"], batch["output"]):
        # 1. full input = 질문 + 답변
        full_text = f"{instr} {output}"
        tokenized = tokenizer(full_text, max_length=1024, truncation=True)

        # 2. 질문 부분만 -100 처리
        instr_ids = tokenizer(instr, max_length=1024, truncation=True)["input_ids"]
        label_ids = tokenized["input_ids"].copy()
        label_ids[:len(instr_ids)] = [IGNORE_INDEX] * len(instr_ids)

        input_texts.append(tokenized["input_ids"])
        labels.append(label_ids)

    return {"input_ids": input_texts, "labels": labels}

tokenized = dataset.map(preprocess, batched=True, remove_columns=["instruction","output"])

# ===============================
# 6. Data Collator 추가 (padding 자동)
# ===============================
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True)

# ===============================
# 7. 학습 설정
# ===============================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=1,       # GPU 한 대당 한 번에 처리하는 데이터 샘플 수
    gradient_accumulation_steps=8,       # 실제 업데이트 전에 8번 미니배치의 그래디언트를 누적
    learning_rate=2e-4,                  # 안정적인 LR (이전 실험에서 가장 정확도 좋음)
    num_train_epochs=5,                  # 전체 데이터셋을 5회 반복 학습
    lr_scheduler_type="cosine",          # cosine scheduler로 학습률 점진 감소
    warmup_ratio=0.03,                   # 학습 초반 3% 구간은 학습률 천천히 증가
    fp16=True,                           # 16-bit 연산
    save_total_limit=2,                  # 저장할 체크포인트 최대 개수
    logging_steps=10,                    # 로그 출력 간격
    save_steps=100,                      # 체크포인트 저장 간격
    report_to="none"                     # wandb/logging 비활성화
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    data_collator=data_collator
)

# ===============================
# 8. 학습 시작
# ===============================
trainer.train()
model.save_pretrained(OUTPUT_DIR)
print("✅ LoRA 학습 완료 & 모델 저장됨:", OUTPUT_DIR)