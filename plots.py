"""plots.py: baseline과 학습 모델의 점수(.npz)로 ppt용 figure 제작하는 코드"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from scipy.stats import norm
from sklearn.metrics import roc_curve, det_curve, auc, DetCurveDisplay

from baseline import compute_eer, RESULTS_DIR


# 상수 설정
FIG_DIR = Path(__file__).parent / "figures"
RUNS = [("Baseline (ImageNet)", "baseline.npz", "tab:gray"), ("Triplet fine-tuned", "trained.npz", "tab:red")]

# 결과 불러오는 함수
#   npz를 np.load로 열어 scores, labels를 numpy 배열로 반환
def load_npz(fname):
    with np.load(RESULTS_DIR / fname) as data:
        return data["scores"], data["labels"]

# 히스토그램 + KDE
def plot_hist():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    bins = np.linspace(0.3, 1.0, 30)

    for ax, (name, fname, color) in zip(axes, RUNS):
        scores, labels = load_npz(fname)
        eer, threshold = compute_eer(scores, labels)

        labels = np.where(labels == 1, "Same", "Diff")

        sns.histplot(
            x=scores,
            hue=labels,
            stat="density",
            kde=True,
            common_norm=False, ax=ax,
            bins=bins
            )

        # threshold 세로선 (빨간 점선)
        line = ax.axvline(threshold, color="red", linestyle="--", label=f"threshold {threshold:.3f}")

        # seaborn이 만든 Same/Diff 범례에 threshold 선을 합쳐 다시 그림
        # claude opus 5.5 의 도움을 받음.
        legend = ax.get_legend()
        handles = list(legend.legend_handles) + [line]
        texts = [t.get_text() for t in legend.get_texts()] + [line.get_label()]
        ax.legend(handles, texts)

        ax.set_title(f"{name} (EER {eer*100:.2f}%)", color=color)
        ax.set_xlabel("cosine similarity")

    # fig.supxlabel("cosine similarity")
    fig.suptitle("Cosine similarity of same / different pairs")
    
    fig.tight_layout()
    fig.savefig(FIG_DIR / "hist_kde.png", dpi=300)

    plt.close(fig)

def plot_roc():
    fig, ax = plt.subplots(figsize=(5, 5))

    for name, fname, color in RUNS:
        scores, labels = load_npz(fname)
        eer, threshold = compute_eer(scores, labels)        
        
        fpr, tpr, _ = roc_curve(labels, scores)
        roc_auc = auc(fpr, tpr)

        ax.plot(fpr, tpr, label=f"{name} (AUC {roc_auc:.3f}, EER {eer*100:.2f}%)", color=color)
        ax.set_xlabel("FPR")
        ax.set_ylabel("TPR")
        
    ax.plot([0, 1], [0, 1], "k--", label="Chance")
    ax.legend(loc="lower right")

    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "roc.png", dpi=300)

    plt.close(fig)

# DET 커브
def plot_det():
    fig, ax = plt.subplots(figsize=(5,5))

    for name, fname, color in RUNS:
        scores, labels = load_npz(fname)
        eer, _ = compute_eer(scores, labels)

        disp = DetCurveDisplay.from_predictions(labels, scores, ax=ax, name=f"{name} (EER {eer*100:.2f}%)", color=color)
        color = disp.line_.get_color()

        ax.plot(norm.ppf(eer), norm.ppf(eer), "o", color=color)

    # y=x 그리기
    ax.plot(ax.get_xlim(), ax.get_xlim(), "k--")

    ax.set_title("DET Curve")
    ax.set_xlabel("FPR")
    ax.set_ylabel("FNR")
    ax.legend()

    fig.tight_layout()
    fig.savefig(FIG_DIR / "det.png", dpi=300)
    plt.close(fig)

if __name__ == "__main__":
    FIG_DIR.mkdir(exist_ok=True)
    plot_hist()
    plot_roc()
    plot_det()