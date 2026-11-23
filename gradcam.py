import torch
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

IMG_DIR = Path(__file__).parent / "img"
FIG_DIR = Path(__file__).parent / "figures"

# Grad-CAM 함수: gradcam(model, layer, x, score_fn)
def gradcam(model, layer, x, score_fn):
    saved = {}

    # layer.register_forward_hook 으로 A를 저장 
    def hook_forward(module, input, output):
        saved["A"] = output
        output.retain_grad()  # output에 대한 기울기 계산 가능하도록 설정

    # layer에 forward hook 걸길
    handle = layer.register_forward_hook(hook_forward)
    
    out = model(x)
    y = score_fn(out)

    model.zero_grad()
    y.backward()

    handle.remove()

    
    A = saved["A"].detach()  # feature map
    G = saved["A"].grad.detach()  # gradient

    # α = 기울기의 공간 평균 (dim=(2, 3))
    alpha = G.mean(dim=(2, 3), keepdim=True)
    # cam = ReLU(Σ α·A)
    cam = F.relu((alpha * A).sum(dim=1, keepdim=True))

    return cam, y.item()  # (B, 1, 7, 7), y 값 반환
        

# cat_dog 이미지에 대해 Grad-CAM을 적용하여 확인하는 main 함수
if __name__ == "__main__":
    # 모델 불러오기
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.eval()

    # 전처리 정의
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
    ])

    # 전처리: Resize((224, 224)) → ToTensor → Normalize
    img = Image.open(IMG_DIR / "cat_dog.jpg").convert("RGB")
    x = preprocess(img).unsqueeze(0)  # 배치 차원 추가

    # tiger cat(282), boxer(242) 두 클래스로 각각 gradcam

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))

    axes[0].imshow(img)
    axes[0].set_title("Original Image")
    axes[0].axis("off")

    for ax, cls, name in zip(axes[1:], [282, 242], ["Cat", "Dog"]):
        # score_fn 정의: 특정 클래스 점수 반환
        score_fn = lambda out: out[0, cls]

        # gradcam 호출
        cam, _ = gradcam(model, model.layer4, x, score_fn)

        # cam을 원본 크기로 키워 사진 위에 겹쳐 그림
        cam_resized = F.interpolate(cam, size=img.size[::-1], mode="bilinear", align_corners=False)
        cam_resized = cam_resized.squeeze().numpy()
        cam_resized = (cam_resized - cam_resized.min()) / (cam_resized.max() - cam_resized.min() + 1e-8)  # 정규화

        ax.imshow(img)
        ax.imshow(cam_resized, cmap="jet", alpha=0.5)
        ax.set_title(f"Grad-CAM: {name}")
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(FIG_DIR / "gradcam_cat_dog.png", dpi=200, bbox_inches="tight")
    plt.close(fig)
