"""
train.py: triplet loss로 ResNet-18을 LFW 얼굴 구별용으로 fine-tuning한다.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models

import time
import random
from pathlib import Path
from PIL import Image
from data import train_images, transform

# [1] 상수 설정
device = "cuda" if torch.cuda.is_available() else "cpu"
EPOCHS = 1
BATCH_SIZE = 32
LR = 1e-4
MARGIN = 0.2
NUM_WORKERS = 4
CKPT_DIR = Path(__file__).parent / "checkpoints"

# 재현성을 위해 random과 torch의 seed를 고정
random.seed(404)
torch.manual_seed(404)

# [2] Triplet Dataset 클래스 (Dataset을 상속)
class TripletDataset(Dataset):
    def __init__(self):
        super().__init__()
        self.items = train_images()

        # '사람번호 → 그 사람의 사진 경로 리스트' 딕셔너리
        self.path_by_pid = {}
        for path, pid in self.items:
            self.path_by_pid.setdefault(pid, []).append(path)


    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        """
        anchor: i번째 사진과 그 사람번호
        positive: 같은 사람의 사진 중 anchor가 아닌 것을 무작위로 하나
        negative: 다른 사람번호를 무작위로 고르고, 그 사람의 사진을 무작위로 하나
        """

        anchor_path, anchor_pid = self.items[i]

        positive_path = random.choice([p for p in self.path_by_pid[anchor_pid] if p != anchor_path])
        # print(anchor_path, positive_path)

        negative_pid = random.randrange(len(self.path_by_pid))
        while negative_pid == anchor_pid:
            negative_pid = random.randrange(len(self.path_by_pid))
        negative_path = random.choice(self.path_by_pid[negative_pid])


        # 세 장 모두 열어서 RGB 변환 → transform → 텐서 3개를 반환
        anchor_tensor = transform(Image.open(anchor_path).convert("RGB"))
        positive_tensor = transform(Image.open(positive_path).convert("RGB"))
        negative_tensor = transform(Image.open(negative_path).convert("RGB"))

        return anchor_tensor, positive_tensor, negative_tensor

# [3] 모델 만드는 함수
def make_model():
    # baseline.py와 동일하지만, eval() 제외
    # resnet 18 불러오고, 마지막 fc층을 변환해서 최종 출력을 512차원으로
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    model.fc = torch.nn.Identity()
    model.to(device) # 모델을 gpu로 보냄
    return model
    
# [4] 학습 함수
def train():
    model = make_model()
    loader = DataLoader(TripletDataset(),
                        batch_size=BATCH_SIZE,
                        shuffle=True,
                        num_workers=NUM_WORKERS,
                        persistent_workers=True
                        )
    loss_fn = torch.nn.TripletMarginLoss(margin=MARGIN)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    # print(len(loader)) # 출력값 = 6671 / 32 = 208.xxx = 209

    start_time = time.time()
    for epoch in range(EPOCHS):
        start_time_epoch = time.time()

        model.train()
        total_loss = 0 # epoch 마다 loss는 0으로 다시 시작

        for anchor, positive, negative in loader:
            # 세 텐서를 device로 이동
            anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

            # 각각 모델에 통과 → 벡터 3개 (배치, 512)
            # 벡터를 F.normalize로 길이 1로 맞춘다
            f_anchor = F.normalize(model(anchor), dim=1)
            f_positive = F.normalize(model(positive), dim=1)
            f_negative = F.normalize(model(negative), dim=1)

            # loss 계산
            loss = loss_fn(f_anchor, f_positive, f_negative)

            # optimizer.zero_grad() → loss.backward() → optimizer.step()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # loss 값을 누적
            total_loss += loss.item()

        # 평균 loss와 걸린 시간 출력
        print(f"epoch {epoch + 1}/{EPOCHS}, loss = {total_loss / len(loader):.4f}")
        print(f"epoch 학습 시간: {time.time() - start_time_epoch:.1f}")

    # checkpoints/ 폴더를 만들고 model.state_dict()를 torch.save로 저장
    CKPT_DIR.mkdir(exist_ok=True)
    torch.save(model.state_dict(), CKPT_DIR / "triplet.pt")
    print(f"총 학습 시간: {time.time() - start_time:.1f}")


# [5] main
if __name__ == "__main__":
    train()