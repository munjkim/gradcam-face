# [0] import: gradcam.py의 gradcam, evaluate.py의 load_trained_model,
#             baseline.py의 make_model, data.py의 eval_pairs와 transform
from gradcam import gradcam
from evaluate import load_trained_model
from baseline import make_model, device
from data import eval_pairs, transform

import torch
import torch.nn.functional as F
from torchvision import transforms

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).parent / "figures"

# [1] score_fn: out(2, 512) → cos(out[0], out[1])  (F.cosine_similarity, 숫자 하나로)
def cos_score(out):
    return F.cosine_similarity(out[0:1], out[1:2])[0]  # (2, 512) → (1,)

# [2] 쌍 하나를 설명하는 함수 explain_pair(model, path_a, path_b)
#   - 두 사진을 transform → torch.cat 또는 torch.stack으로 (2, 3, 224, 224)
#   - gradcam(model, model.layer4, x, score_fn) → cam (2, 1, 7, 7)
#   - 224×224로 키우고, 사진마다 따로 0~1 정규화
#   - 유사도 y 값도 함께 반환
def explain_pair(model, path_a, path_b):
    img_a = transform(Image.open(path_a).convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
    img_b = transform(Image.open(path_b).convert("RGB")).unsqueeze(0)  # (1, 3, 224, 224)
    torch_cat = torch.cat([img_a, img_b], dim=0).to(device)  # (2, 3, 224, 224)

    cam, y = gradcam(model, model.layer4, torch_cat, cos_score)  # (2, 1, 7, 7), y 값
    cam_resized = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)  # (2, 1, 224, 224
    cam_resized = cam_resized.squeeze(1)  # (2, 224, 224)

    lo = cam_resized.amin(dim=(1, 2), keepdim=True)  # (2, 1, 1)
    hi = cam_resized.amax(dim=(1, 2), keepdim=True)  # (2, 1, 1)
    cam_resized = (cam_resized - lo) / (hi - lo + 1e-8)  # (2, 224, 224) 정규화

    return cam_resized[0].cpu().numpy(), cam_resized[1].cpu().numpy(), y  # a 히트맵 (224, 224), b 히트맵 (224, 224), y 값 반환

# [3] 실행부: 몇 쌍을 골라 baseline / triplet 모델로 각각 그리기
#   - 칸 구성 예: [a 원본 | a 히트맵 | b 히트맵 | b 원본], 제목에 y 값
#   - figures/face_gradcam_*.png
if __name__ == "__main__":
    baseline_model = make_model().eval()
    triplet_model = load_trained_model()

    RUNS = [("Baseline", baseline_model), ("Triplet", triplet_model)]

    pairs = eval_pairs()  # (path_a, path_b, label) 리스트
    selected_pairs = [
        pairs[0],  # same
        pairs[1],  # same
        pairs[500],  # diff
        pairs[501],  # diff
    ]

    for i, (path_a, path_b, label) in enumerate(selected_pairs):
        fig, axes = plt.subplots(len(RUNS), 4, figsize=(12, 6))

        crop = transforms.Compose([transforms.CenterCrop(150), transforms.Resize((224, 224))])  # 원본 사진을 224×224로 자르기
        img_a = crop(Image.open(path_a).convert("RGB"))
        img_b = crop(Image.open(path_b).convert("RGB"))

        for row, (name, model) in enumerate(RUNS):
            cam_a, cam_b, y = explain_pair(model, path_a, path_b)

            axes[row, 0].imshow(img_a)
            axes[row, 0].set_title(f"{name} - A")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(img_a)
            axes[row, 1].imshow(cam_a, cmap="jet", alpha=0.5)
            axes[row, 1].set_title(f"{name} - A CAM")
            axes[row, 1].axis("off")

            axes[row, 2].imshow(img_b)
            axes[row, 2].imshow(cam_b, cmap="jet", alpha=0.5)
            axes[row, 2].set_title(f"{name} - B CAM")
            axes[row, 2].axis("off")

            axes[row, 3].imshow(img_b)
            axes[row, 3].set_title(f"{name} - B\nCosine similarity: {y:.4f}")
            axes[row, 3].axis("off")

        fig.suptitle("Same person" if label == 1 else "Different person")
        fig.tight_layout()
        fig.savefig(FIG_DIR / f"face_gradcam_{i}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
        