import re
import os
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
from peft import PeftModel  # 누락 임포트 추가

# CUDA 디버그 옵션
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# ===========================
# 데이터 로드
# ===========================
test = pd.read_csv('./test.csv')

# ===========================
# 유틸 함수
# ===========================
def is_multiple_choice(question_text: str) -> bool:
    """객관식 여부 판단"""
    lines = question_text.strip().split("\n")
    option_count = sum(bool(re.match(r"^\s*[1-9][0-9]?\s", line)) for line in lines)
    return option_count >= 2

def extract_question_and_choices(full_text: str):
    """질문과 선택지 분리"""
    lines = full_text.strip().split("\n")
    q_lines, options = [], []
    for line in lines:
        if re.match(r"^\s*[1-9][0-9]?\s", line):
            options.append(line.strip())
        else:
            q_lines.append(line.strip())
    return " ".join(q_lines), options

def make_prompt_auto(text: str) -> str:
    """프롬프트 생성"""
    if is_multiple_choice(text):
        question, options = extract_question_and_choices(text)
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요. **개인정보보호법이랑 전자금융거래법을 참조하세요.**\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 **정확하고 자세하며 끊기지 않게 끝까지** 설명하세요. **개인정보보호법이랑 전자금융거래법을 참조하세요.**\n\n"
            f"질문: {text}\n\n"
            "답변:"
        )

def extract_answer_only(generated_text: str, original_question: str) -> str:
    """후처리: 답변만 추출"""
    text = generated_text.split("답변:")[-1].strip() if "답변:" in generated_text else generated_text.strip()
    if not text:
        return "미응답"
    if is_multiple_choice(original_question):
        match = re.match(r"\D*([1-9][0-9]?)", text)
        return match.group(1) if match else "0"
    return text

# ===========================
# 모델 로드 (4bit)
# ===========================
MODEL_NAME = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"
LORA_DIR = "./lora-output"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16"
)

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    quantization_config=bnb_config
)
model = PeftModel.from_pretrained(base_model, LORA_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device_map="auto"
)

# ===========================
# 추론 (배치 처리)
# ===========================
preds = []
for q in tqdm(test["Question"]):
    prompt = make_prompt_auto(q)
    out = pipe(prompt, max_new_tokens=128, temperature=0.2, do_sample=False)[0]["generated_text"]
    answer = extract_answer_only(out, q)
    preds.append(answer)

sample_submission = pd.read_csv("./sample_submission.csv")
sample_submission["Answer"] = preds
sample_submission.to_csv("./finetuned2_submission.csv", index=False, encoding="utf-8-sig")
print("✅ 추론 완료: finetuned2_submission.csv")
