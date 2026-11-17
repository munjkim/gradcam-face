# gradcam-face

IC614c Computer Vision final project (DGIST, 2026 Fall).
Grad-CAM (Selvaraju et al., ICCV 2017) extended from class scores to face-verification similarity: y = cos(f(a), f(b)).

> **Rule for any AI assistant working in this repo:** do NOT write implementation code here.
> All project code must be written by the student (course plagiarism/AI check, detection = F).
> Allowed: explaining concepts, reviewing the student's code, debugging hints, experiment design.

## Third-party parts (declare in the report)
- torchvision ResNet-18 ImageNet weights (backbone)
- LFW dataset loader (torchvision / scikit-learn)

## Apparatus
| Item | Value |
|---|---|
| Machine | Lab workstation (Windows) |
| GPU | NVIDIA GeForce RTX 5090, 32 GB VRAM, compute capability 12.0 |
| PyTorch | 2.10.0+cu130 |
| torchvision | 0.25.0+cu130 |
| CUDA (wheel) | 13.0 |
| Environment | conda env `cuda130` |
| Python | 3.11.15 |
| CPU / RAM | _TODO_ |

Check command:
```
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
```
Windows note: DataLoader with `num_workers > 0` needs the `if __name__ == "__main__":` guard.
