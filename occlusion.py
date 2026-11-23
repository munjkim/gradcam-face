"""occlusion.py: 가리기 실험 — 모델이 얼굴 가운데와 테두리 중 어디에 기대는가.

이 파일은 추가 분석을 위해 AI에게 코드 작성을 요청하였고, 직접 검수하였습니다.

조건 (모델 입력 224×224 기준, avg_gradcam.py의 점선 상자와 같은 영역):
- full        : 가리지 않음 (= baseline.py / evaluate.py 결과와 같아야 함)
- center_only : 가운데 112×112만 남기고 테두리를 가림
- border_only : 가운데 112×112를 가리고 테두리만 남김
가린 곳은 정규화 공간에서 0 = ImageNet 평균 색(회색)으로 채운다.
조건마다 테스트 1,000쌍의 EER을 다시 잰다. 결과: results/occlusion.npz
"""
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from baseline import make_model, compute_eer, device, RESULTS_DIR
from evaluate import load_trained_model
from data import eval_pairs, transform

CENTER = slice(56, 168)
CONDITIONS = ["full", "center_only", "border_only"]


def masked(x, condition):
    """x: (3, 224, 224) 정규화된 텐서. 조건에 맞게 가린 복사본."""
    x = x.clone()
    if condition == "center_only":
        keep = x[:, CENTER, CENTER].clone()
        x.zero_()
        x[:, CENTER, CENTER] = keep
    elif condition == "border_only":
        x[:, CENTER, CENTER] = 0
    return x


@torch.no_grad()
def embed_all(model, paths, condition, batch=64):
    """경로 목록 → {경로: 512차원 특징}. 사진마다 한 번만 계산."""
    feats = {}
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        x = torch.stack([masked(transform(Image.open(p).convert("RGB")), condition) for p in chunk])
        out = model(x.to(device)).cpu()
        feats.update(zip(chunk, out))
    return feats


def eer_for(model, pairs, condition):
    paths = sorted({p for a, b, _ in pairs for p in (a, b)})
    f = embed_all(model, paths, condition)
    scores = np.array([F.cosine_similarity(f[a][None], f[b][None]).item() for a, b, _ in pairs])
    labels = np.array([l for _, _, l in pairs])
    eer, _ = compute_eer(scores, labels)
    return eer, scores


def plot_occlusion(example_path):
    """results/occlusion.npz → figures/occlusion.png (계산 없이 그리기만).
    왼쪽: 예시 얼굴의 세 조건, 오른쪽: 조건별 EER 막대그래프."""
    import matplotlib.pyplot as plt
    from pathlib import Path
    from torchvision import transforms

    with np.load(RESULTS_DIR / "occlusion.npz") as d:
        eers = {m: [float(d[f"{m}_{c}_eer"]) * 100 for c in CONDITIONS] for m in ("baseline", "triplet")}

    # 예시 얼굴: 모델 입력과 같은 자르기, 가린 곳은 ImageNet 평균 색
    crop = transforms.Compose([transforms.CenterCrop(150), transforms.Resize(224)])
    face = np.asarray(crop(Image.open(example_path).convert("RGB"))).copy()
    gray = np.array([124, 116, 104], dtype=np.uint8)  # ImageNet 평균 (0.485, 0.456, 0.406) × 255
    center_only, border_only = np.empty_like(face), face.copy()
    center_only[:] = gray
    center_only[CENTER, CENTER] = face[CENTER, CENTER]
    border_only[CENTER, CENTER] = gray
    examples = [face, center_only, border_only]
    titles = ["Full", "Center only", "Border only"]

    fig, axd = plt.subplot_mosaic([["f", "c", "b", "bar", "bar"]], figsize=(14, 3.4), layout="constrained")
    for key, img, t in zip("fcb", examples, titles):
        axd[key].imshow(img)
        axd[key].set_title(t)
        axd[key].axis("off")

    ax = axd["bar"]
    x = np.arange(len(CONDITIONS))
    for i, (m, label, color) in enumerate([("baseline", "Baseline (ImageNet)", "tab:gray"),
                                           ("triplet", "Triplet fine-tuned", "tab:red")]):
        bars = ax.bar(x + (i - 0.5) * 0.38, eers[m], width=0.38, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f%%", fontsize=9)
    ax.set_xticks(x, titles)
    ax.set_ylabel("Test EER (%)")
    ax.set_ylim(0, max(max(v) for v in eers.values()) * 1.2)
    ax.legend(loc="upper left")
    ax.set_title("EER under occlusion (lower is better)")

    out = Path(__file__).parent / "figures" / "occlusion.png"
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out}")


if __name__ == "__main__":
    pairs = eval_pairs()
    models = [("baseline", make_model().eval()), ("triplet", load_trained_model())]

    save = {}
    print(f"{'model':10s}" + "".join(f"{c:>14s}" for c in CONDITIONS))
    for name, model in models:
        row = []
        for c in CONDITIONS:
            eer, scores = eer_for(model, pairs, c)
            row.append(eer)
            save[f"{name}_{c}_scores"] = scores
            save[f"{name}_{c}_eer"] = eer
        print(f"{name:10s}" + "".join(f"{e * 100:13.2f}%" for e in row))

    save["labels"] = np.array([l for _, _, l in pairs])
    RESULTS_DIR.mkdir(exist_ok=True)
    np.savez(RESULTS_DIR / "occlusion.npz", **save)
    print(f"saved {RESULTS_DIR / 'occlusion.npz'}")

    plot_occlusion(pairs[0][0])
