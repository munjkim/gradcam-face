"""
data.py는 LFW 데이터를 준비하는 코드임.
1. 평가용 쌍
2. 학습용 사진 목록 확인
3. 이미지 전처리
"""
import pandas as pd
from pathlib import Path
from torchvision import transforms

DATA_DIR = Path(__file__).parent / "data"
IMG_DIR = DATA_DIR / "lfw-deepfunneled/lfw-deepfunneled"

def img_path(name, num):
    """이름과 사진 번호로 사진 경로를 만든다.

    예: ("Abdullah_Gul", 13) → .../Abdullah_Gul/Abdullah_Gul_0013.jpg
    """
    return IMG_DIR / name / f"{name}_{int(num):04d}.jpg"

def get_rows(fname):
    """csv 안의 row들을 리스트로 반환
    """
    df = pd.read_csv(DATA_DIR / fname)
    return df.values.tolist()

def eval_pairs():
    """평가용 쌍 목록 1000개를 반환한다. Baseline EER 계산용이다.

    Returns:
        (경로1, 경로2, label) 리스트. label은 같은 사람일때 1, 다른사람이면 0 이다.
    """
    # 같은 사람 쌍 500개: 한 줄 = 이름, 사진번호1, 사진번호2
    match = [(img_path(n, a), img_path(n, b), 1) for n, a, b in get_rows("matchpairsDevTest.csv")]
    # 다른 사람 쌍 500개: 한 줄 = 이름1, 사진번호1, 이름2, 사진번호2
    mismatch = [(img_path(n1, a), img_path(n2, b), 0) for n1, a, n2, b in get_rows("mismatchpairsDevTest.csv")]
    return match + mismatch

def train_images():
    """학습용 사진 목록 반환하는 함수. Triplet 학습에 필요하다.
    positive 가 필요하므로, 사진이 2장 이상인 사람을 골라야 한다.

    Returns:
        (경로, 사람번호) 리스트. 사람번호는 사진이 2상 이상인 사람 고른 후, 0,1,2,,,로 지정한다.
    """
    people_over2 = [(name, int(k)) for name, k in get_rows("peopleDevTrain.csv") if int(k) >= 2]

    return [(img_path(name, i), pid)
            for pid, (name, k) in enumerate(people_over2)
            for i in range(1, k+1)]

# 사진을 모델에 넣기 전에 전처리
transform = transforms.Compose([
    transforms.CenterCrop(150), # 250×250 원본에서 얼굴이 있는 가운데 150×150만 잘라낸다
    transforms.Resize(224), # 모델이 기대하는 입력 크기 224×224로 키운다
    transforms.ToTensor(), # 사진 → 숫자 텐서 (3, 224, 224), 값 범위 0~1
    transforms.Normalize([0.485, 0.456, 0.406], # ImageNet 평균 (R, G, B)
                         [0.229, 0.224, 0.225]), # ImageNet 표준편차 (R, G, B)
])

if __name__ == "__main__":
    from PIL import Image

    pairs = eval_pairs()
    train = train_images()

    print("eval pairs:", len(pairs), ", same:", sum(label for p1, p2, label in pairs))
    print("train images:", len(train), ", people:", len({pid for p, pid in train}))
    assert all(p.exists() for p1, p2, _ in pairs for p in (p1, p2)) # 평가 사진 모든 경로에 사진이 있는가
    assert all(p.exists() for p, _ in train) # 학습 사진 모든 경로에 사진이 있는가

    x = transform(Image.open(pairs[0][0]).convert("RGB")) # 사진 한 장을 변환
    print("tensor:", tuple(x.shape))
    assert x.shape == (3, 224, 224)

    # 사진 출력 확인.
    # transforms.CenterCrop(150)(Image.open(pairs[0][0])).show()

    # 평가와 학습 사람이 겹치지 않으면 통과
    assert {p.parent.name for p1, p2, label in pairs for p in (p1, p2)}.isdisjoint({p.parent.name for p, pid in train})
