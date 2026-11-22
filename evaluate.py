from baseline import make_model, get_scores, compute_eer
from baseline import device, RESULTS_DIR
from train import CKPT_DIR

import torch
import numpy as np

def load_trained_model():
    """학습한 triplet 모델을 불러와서 평가 모드로 반환"""
    model = make_model()
    model.load_state_dict(torch.load(CKPT_DIR / 'triplet.pt', map_location=device))
    model.eval()
    return model

if __name__ == "__main__":
    model = load_trained_model()

    scores, labels = get_scores(model)
    
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / "trained.npz", scores=scores, labels=labels)

    eer, threshold = compute_eer(scores, labels)
    print(f"EER: {eer*100:.2f}%, threshold: {threshold:.4f}")