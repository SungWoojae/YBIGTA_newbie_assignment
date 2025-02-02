import re
from datasets import load_dataset

def load_corpus() -> list[str]:
    """
    Hugging Face의 Poem Sentiment 데이터셋을 불러와 전처리한 후,
    Word2Vec 학습에 적합한 형태로 변환하여 반환합니다.

    Returns:
        list[str]: 정제된 문장들의 리스트
    """
    corpus: list[str] = []

    # 1. 데이터셋 로드
    dataset = load_dataset("google-research-datasets/poem_sentiment", split="train")

    # 2. 텍스트 전처리
    for sample in dataset:
        text = sample["verse_text"]

        # (1) 소문자 변환
        text = text.lower()

        # (2) 특수 문자 및 숫자 제거
        text = re.sub(r"[^a-z\s]", "", text)

        # (3) 불필요한 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        # (4) 문장이 비어있지 않으면 corpus에 추가
        if text:
            corpus.append(text)

    return corpus