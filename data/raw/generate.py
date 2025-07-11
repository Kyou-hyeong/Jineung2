import json
import os

JSON_PATH = "qna_dataset.json"

def load_existing_data():
    """기존 JSON 파일이 있으면 불러오고, 없으면 빈 리스트 반환"""
    if os.path.exists(JSON_PATH):
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []

def save_data(data):
    """리스트 형태의 데이터를 JSON 파일로 저장"""
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"✅ 저장 완료: {JSON_PATH}")

def main():
    print("질문과 답변을 입력하세요. 끝내려면 'exit'을 입력하세요.")
    dataset = load_existing_data()

    while True:
        question = input("Q: ").strip()
        if question.lower() == "exit":
            break

        answer = input("A: ").strip()
        if answer.lower() == "exit":
            break

        dataset.append({
            "question": question,
            "answer": answer
        })
        print("👉 추가됨.")

    save_data(dataset)

if __name__ == "__main__":
    main()
