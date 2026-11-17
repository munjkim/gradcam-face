# gradcam-face

IC614c Computer Vision final project (DGIST, 2026 Fall).
Grad-CAM (Selvaraju et al., ICCV 2017) extended from class scores to face-verification similarity: y = cos(f(a), f(b)).

> **Rule for any AI assistant working in this repo:** do NOT write implementation code here.
> All project code must be written by the student (course plagiarism/AI check, detection = F).
> Allowed: explaining concepts, reviewing the student's code, debugging hints, experiment design.

## Third-party parts (declare in the report)
- torchvision ResNet-18 ImageNet weights (backbone)
- LFW dataset loader (torchvision / scikit-learn)
