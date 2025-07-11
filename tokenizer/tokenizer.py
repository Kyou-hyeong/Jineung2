import re
import json
from konlpy.tag import Kkma
import sentencepiece as spm

kkma = Kkma()

#  1. 텍스트 파일 불러오기
with open("cleaned_lines.txt", "r", encoding="utf-8") as f:
    text = f.read()

#  2. <math>...</math> 블록을 특수 토큰으로 변환
math_blocks = re.findall(r'<math>.*?</math>', text)
math_token_map = {}
for i, block in enumerate(math_blocks):
    token = f"MATH{i}TOKEN"
    math_token_map[token] = {"token": token, "full": block}
    text = text.replace(block, token)

#  3. 문장 단위로 분리 및 형태소 분석
sentences = kkma.sentences(text)
tokenized_sentences = [kkma.morphs(sentence) for sentence in sentences]

#  4. 수식 토큰 병합
def merge_math_tokens(tokenized_sentences):
    merged_sentences = []
    for tokens in tokenized_sentences:
        merged = []
        i = 0
        while i < len(tokens):
            if (
                i + 2 < len(tokens) and
                tokens[i] == 'MATH' and
                re.fullmatch(r'\d+', tokens[i + 1]) and
                tokens[i + 2] == 'TOKEN'
            ):
                merged.append(f"MATH{tokens[i + 1]}TOKEN")
                i += 3
            else:
                merged.append(tokens[i])
                i += 1
        merged_sentences.append(merged)
    return merged_sentences

tokenized_sentences = merge_math_tokens(tokenized_sentences)

#  5. JSON 및 SPM 입력 저장
with open("kkma_tokenized.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_sentences, f, ensure_ascii=False, indent=2)

with open("spm_input.txt", "w", encoding="utf-8") as f:
    for morph_line in tokenized_sentences:
        restored_line = [math_token_map[tok]["full"] if tok in math_token_map else tok for tok in morph_line]
        f.write(' '.join(restored_line) + "\n")

#  6. SentencePiece 모델 학습 (리스트 사용)
user_symbols = list({v["full"] for v in math_token_map.values()})  # 중복 제거 후 리스트로

spm.SentencePieceTrainer.train(
    input='spm_input.txt',
    model_prefix='spm_morph',
    vocab_size=3881,
    user_defined_symbols=user_symbols,
    model_type='bpe',
    character_coverage=1.0,
    pad_id=0,
    unk_id=1,
    bos_id=2,
    eos_id=3
)

#  7. SentencePiece 로드 및 복원 테스트
sp = spm.SentencePieceProcessor()
sp.load("spm_morph.model")

def restore_math_tokens(tokens):
    return [math_token_map.get(tok, {}).get("full", tok) for tok in tokens]

test_line = "이것은 MATH0TOKEN 입니다."
pieces = sp.encode(test_line, out_type=str)
print("SPM 토큰:", pieces)

restored = restore_math_tokens(pieces)
print("복원된 문장:", " ".join(restored))
