from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import torch
import pandas as pd
from tqdm import tqdm

# 모델 로드
MODEL_NAME = "yanolja/EEVE-Korean-Instruct-7B-v2.0-Preview"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

# 파이프라인 생성
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)


# 프롬프트 템플릿
def build_prompt(question, answer, num_aug=5):
    return f"""다음은 하나의 퀴즈 문제와 정답입니다.

Q: {question}
A: {answer}

위 문제와 정답이 동일한 상황에서, 다른 방식으로 질문을 표현해 주세요.
총 {num_aug}개의 질문을 다양한 표현으로 만들어 주세요.
다만 정답은 동일하게 유지되어야 합니다.

질문들:
1."""


# 질문 다양화 함수
def generate_augmented_questions(question, answer, num_aug=5, max_tokens=512):
    prompt = build_prompt(question, answer, num_aug)
    output = generator(prompt, max_new_tokens=max_tokens, do_sample=True, temperature=0.7)[0]['generated_text']

    # 결과 파싱 (숫자 형태의 목록 추출)
    lines = output.split('\n')
    augmented = []
    for line in lines:
        if line.strip().startswith(tuple(str(i) + '.' for i in range(1, num_aug + 1))):
            q = line.split('.', 1)[-1].strip()
            if len(q) > 5:
                augmented.append(q)
    return augmented


# 예시 CSV 파일 불러오기 (형식: question,answer)
df = pd.read_csv("external_data.csv")  # CSV 파일: 'question','answer' 컬럼 포함

# 결과 저장용
augmented_data = []

# 증강 수행
for idx, row in tqdm(df.iterrows(), total=len(df)):
    question = row['question']
    answer = row['answer']
    augmented = generate_augmented_questions(question, answer, num_aug=5)
    for new_q in augmented:
        augmented_data.append({'question': new_q, 'answer': answer})

# DataFrame으로 저장
aug_df = pd.DataFrame(augmented_data)
aug_df.to_csv("augmented_qa.csv", index=False, encoding='utf-8-sig')
print("✅ 증강 완료! 'augmented_qa.csv'로 저장됨.")
