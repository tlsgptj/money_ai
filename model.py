import re
import os
import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig

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
            "아래 질문에 대해 적절한 **정답 선택지 번호만 출력**하세요.\n\n"
            f"질문: {question}\n"
            "선택지:\n"
            f"{chr(10).join(options)}\n\n"
            "답변:"
        )
    else:
        return (
            "당신은 금융보안 전문가입니다.\n"
            "아래 주관식 질문에 대해 **정확하고 자세하며 끊기지 않게 끝까지** 설명하세요.\n\n"
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
# 모델 로드 (4bit + CPU 오프로딩)
# ===========================
model_name = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"

quant_config = BitsAndBytesConfig(
    load_in_4bit=True,  # 8GB VRAM 환경에서는 4bit 권장
    bnb_4bit_compute_dtype=torch.float16,
    llm_int8_enable_fp32_cpu_offload=True
)

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    quantization_config=quant_config,
    torch_dtype=torch.float16
)

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
batch_size = 2  # 배치 2~4 권장 (VRAM 8GB 기준)

for start_idx in tqdm(range(0, len(test), batch_size), desc="Inference"):
    batch_questions = test['Question'][start_idx:start_idx + batch_size]
    prompts = [make_prompt_auto(q) for q in batch_questions]

    outputs = pipe(
        prompts,
        max_new_tokens=128,  # 128 → 64로 단축 (속도 개선)
        temperature=0.2,
        top_p=0.9,
        do_sample=False     # deterministic하게
    )

    for q, out in zip(batch_questions, outputs):
        preds.append(extract_answer_only(out[0]["generated_text"], original_question=q))

# ===========================
# 제출 파일 생성
# ===========================
sample_submission = pd.read_csv('./sample_submission.csv')
sample_submission['Answer'] = preds
sample_submission.to_csv('./improve_submission.csv', index=False, encoding='utf-8-sig')

print("✅ improve_submission.csv 저장 완료")
