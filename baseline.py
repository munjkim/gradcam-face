"""
baseline.py: 학습하지 않은 ImageNet ResNet-18로 LFW 평가 쌍의 EER을 측정한다.
"""

import torch
from torchvision import models
from data import eval_pairs, transform
from PIL import Image
from sklearn.metrics import roc_curve
import numpy as np
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
RESULTS_DIR = Path(__file__).parent / "results"

def make_model():
    # resnet 18 불러오고, 마지막 fc층을 변환해서 최종 출력을 512차원으로
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device) # 모델을 gpu로 보냄
    return model

# 사진 한 장을 512차원 벡터로 바꾸는 함수
@torch.no_grad()
def embed(model, path):
    """이미지 경로를 통해 이미지를 받아와서 512차원 임베딩하는 함수

    transform 후 배치 추가 후 모델을 통과한다.
    """
    # 입력을 GPU로 보냄.
    x = transform(Image.open(path).convert("RGB")).unsqueeze(0).to(device)
    return model(x)
# print(embed(eval_pairs()[0][0]).shape)

# 평가 쌍 1,000개 각각에 대해 cosine 유사도 계산
def get_scores(model):
    """평가 쌍 1,000개에 대한 유사도 계산 후

    return scores, labels
    """
    scores, labels = [], []
    for p1, p2, label in eval_pairs():
        v1_embed, v2_embed = embed(model, p1), embed(model, p2)
        similarity = torch.nn.functional.cosine_similarity(v1_embed, v2_embed).item()

        scores.append(similarity)
        labels.append(label)

    return scores, labels

# EER 계산
def compute_eer(scores, labels):
    """scores와 labels를 바탕으로 eer 계산

    return: eer, threshold[idx]
    """
    fpr, tpr, threshold = roc_curve(labels, scores)
    fnr = 1 - tpr
    idx = np.argmin(np.abs(fpr-fnr))
    eer = np.mean([fpr[idx], fnr[idx]])

    return eer, threshold[idx]

if __name__ == "__main__":
    #  baseline 모델 준비
    model = make_model()
    model.eval()

    scores, labels = get_scores(model)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / "baseline.npz", scores=scores, labels=labels)

    eer, threshold = compute_eer(scores, labels)
    print(f"EER: {eer*100:.2f}%, threshold: {threshold:.4f}")