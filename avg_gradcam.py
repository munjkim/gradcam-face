"""avg_gradcam.py: 경우(TP/TN/FP/FN)별 평균 Grad-CAM 히트맵과 중앙 집중도.

이 파일은 추가 분석을 위해 AI에게 코드 작성을 요청하였고, 직접 검수하였습니다.

- 테스트 1,000쌍 전부에 대해 explain_pair(face_gradcam.py)로 두 얼굴의 히트맵을 구한다.
- 경우는 모델마다 자기 EER 기준값으로 나눈다 (점수 >= 기준값 → 같은 사람으로 판단).
- 평균 히트맵: 경우에 속한 쌍의 A, B 히트맵(각각 0~1 정규화)을 모두 평균.
  LFW deep-funneled는 얼굴이 정렬되어 있어 위치별 평균이 의미를 가진다.
- 중앙 집중도: 히트맵 합 중 가운데 112×112(전체 면적의 25%) 안에 있는 비율. 균일하면 0.25.
- 결과: results/avg_gradcam.npz, figures/avg_gradcam.png
"""
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from torchvision import transforms

from baseline import make_model, compute_eer, RESULTS_DIR
from evaluate import load_trained_model
from data import eval_pairs
from face_gradcam import explain_pair

FIG_DIR = Path(__file__).parent / "figures"
CASES = ["TP", "TN", "FP", "FN"]
CENTER = slice(56, 168)  # 224×224 안의 가운데 112×112


def case_masks(scores, labels, threshold):
    """경우 이름 → 쌍 1,000개에 대한 True/False 배열."""
    accept = scores >= threshold
    return {
        "TP": (labels == 1) & accept,
        "TN": (labels == 0) & ~accept,
        "FP": (labels == 0) & accept,
        "FN": (labels == 1) & ~accept,
    }


def center_ratio(cam):
    """히트맵 합 중 가운데 영역 안의 비율."""
    return cam[CENTER, CENTER].sum() / (cam.sum() + 1e-8)


def run(model, npz_name, pairs):
    """한 모델에 대해 경우별 평균 히트맵, 중앙 집중도 평균, 개수를 계산."""
    with np.load(RESULTS_DIR / npz_name) as d:
        scores, labels = d["scores"], d["labels"]
    _, threshold = compute_eer(scores, labels)
    masks = case_masks(scores, labels, threshold)

    case_of = np.empty(len(pairs), dtype=object)  # 쌍 번호 → 경우 이름
    for case, m in masks.items():
        case_of[m] = case

    sums = {c: np.zeros((224, 224)) for c in CASES}
    ratios = {c: [] for c in CASES}
    for i, (path_a, path_b, _) in enumerate(pairs):
        ha, hb, _ = explain_pair(model, path_a, path_b)
        c = case_of[i]
        sums[c] += ha + hb
        ratios[c] += [center_ratio(ha), center_ratio(hb)]
        if (i + 1) % 100 == 0:
            print(f"  {npz_name}: {i + 1}/{len(pairs)}")

    counts = {c: int(masks[c].sum()) for c in CASES}
    means = {c: sums[c] / max(2 * counts[c], 1) for c in CASES}  # 쌍마다 히트맵 2장
    ratio = {c: float(np.mean(ratios[c])) if ratios[c] else float("nan") for c in CASES}
    return means, ratio, counts, threshold


def mean_face(pairs):
    """모든 테스트 얼굴(잘린 224×224)의 평균 — 히트맵 배경용."""
    crop = transforms.Compose([transforms.CenterCrop(150), transforms.Resize(224)])
    paths = {p for a, b, _ in pairs for p in (a, b)}
    acc = np.zeros((224, 224, 3))
    for p in paths:
        acc += np.asarray(crop(Image.open(p).convert("RGB")), dtype=float)
    return (acc / len(paths)).astype(np.uint8)


if __name__ == "__main__":
    pairs = eval_pairs()
    models = [
        ("Baseline (ImageNet)", make_model().eval(), "baseline.npz"),
        ("Triplet fine-tuned", load_trained_model(), "trained.npz"),
    ]

    results = {}
    for name, model, npz_name in models:
        print(name)
        results[name] = run(model, npz_name, pairs)

    # 숫자 저장 + 출력
    save = {}
    print("\ncenter ratio (uniform = 0.25), count")
    for name, (means, ratio, counts, th) in results.items():
        key = "baseline" if name.startswith("Baseline") else "triplet"
        print(f"{name} (threshold {th:.4f})")
        for c in CASES:
            print(f"  {c}: {ratio[c]:.3f}  (n={counts[c]})")
            save[f"{key}_{c}_mean"] = means[c]
            save[f"{key}_{c}_ratio"] = ratio[c]
            save[f"{key}_{c}_n"] = counts[c]
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / "avg_gradcam.npz", **save)

    # 그림: 행 = 모델, 열 = 경우. 색 범위는 모든 칸 공통(0~1)이라 칸끼리 세기 비교 가능
    face = mean_face(pairs)
    # layout="constrained": 제목·colorbar 공간까지 자동 배치 (tight_layout은 colorbar(ax=axes)와 겹침)
    fig, axes = plt.subplots(2, 4, figsize=(12, 6.6), layout="constrained")
    for row, (name, (means, ratio, counts, th)) in enumerate(results.items()):
        for col, c in enumerate(CASES):
            ax = axes[row, col]
            ax.imshow(face)
            im = ax.imshow(means[c], cmap="jet", alpha=0.5, vmin=0, vmax=1)
            ax.add_patch(plt.Rectangle((56, 56), 112, 112, fill=False, ls="--", lw=1, color="white"))
            ax.set_title(f"{c} (n={counts[c]})\ncenter ratio {ratio[c]:.2f}", fontsize=10)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(name, fontsize=11)
    fig.colorbar(im, ax=axes, shrink=0.8, label="mean Grad-CAM (0–1)")
    fig.suptitle("Average Grad-CAM of cosine similarity per decision case")
    FIG_DIR.mkdir(exist_ok=True)
    fig.savefig(FIG_DIR / "avg_gradcam.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved {FIG_DIR / 'avg_gradcam.png'}")
