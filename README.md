[README.md — Pneumonia Detection with OOD Safety Layer.md](https://github.com/user-attachments/files/31196584/README.md.Pneumonia.Detection.with.OOD.Safety.Layer.md)
# 🫁 Pneumonia Detection with an Out-of-Distribution Safety Layer

A deep learning medical-imaging project that extends a chest X-ray pneumonia classifier with an **Out-of-Distribution (OOD) detection layer**.

Instead of forcing every uploaded image into either:

```text
NORMAL
PNEUMONIA
```

the system also asks:

> **Does this image actually look like the kind of chest X-ray the model was trained on?**

The project combines:

- A custom **SimpleCNN** pneumonia classifier
- Class-imbalance handling
- Data-driven classification threshold selection
- **Mahalanobis-distance OOD detection**
- A two-tier **warn / reject** safety mechanism
- Google Colab **TPU v5e-1 (`V5E1`)** execution with PyTorch/XLA

> **Runtime:** Built for **Google Colab TPU v5e-1** using `torch_xla`.

---

# 🎯 Motivation

A standard binary classifier always returns one of the classes it knows.

For example, a pneumonia classifier trained only on chest X-rays can still receive:

- A chair photograph
- A leg X-ray
- A screenshot
- A corrupted or unrelated image

and confidently classify it as:

```text
NORMAL
```

or:

```text
PNEUMONIA
```

The sigmoid output has no built-in concept of:

```text
"This is not a chest X-ray."
```

In a medical setting, a confident prediction on an unrelated input can be more dangerous than withholding the prediction entirely.

This project therefore adds a lightweight **OOD safety layer** around the classifier.

---

# ⚡ Google Colab TPU Runtime

The notebook is specifically configured for:

```text
Google Colab
Accelerator: TPU
TPU type: V5E1 / TPU v5e-1
```

The notebook metadata includes:

```text
accelerator = TPU
gpuType = V5E1
```

and the PyTorch implementation uses **PyTorch/XLA**.

---

## TPU Setup

The notebook installs the TPU-enabled XLA package with:

```bash
pip install -q -U 'torch_xla[tpu]' \
-f https://storage.googleapis.com/libtpu-releases/index.html
```

The device is initialized with:

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()

print("device:", device)
print("kind:", xm.xla_device_hw(device))
```

---

## TPU Data Loading

The PyTorch `DataLoader`s are wrapped using:

```python
import torch_xla.distributed.parallel_loader as pl

train_loader = pl.MpDeviceLoader(train_loader, device)
val_loader = pl.MpDeviceLoader(val_loader, device)
test_loader = pl.MpDeviceLoader(test_loader, device)
```

Training uses the XLA-specific optimizer step:

```python
xm.optimizer_step(optimizer)
```

instead of the normal:

```python
optimizer.step()
```

This allows model training to execute through XLA on the Colab TPU.

---

# 📊 Dataset

The project uses the **Chest X-Ray Images (Pneumonia)** dataset distributed through Kaggle.

The notebook downloads the dataset using:

```python
import kagglehub

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia"
)
```

The complete dataset contains:

```text
5,856 pediatric chest X-rays
```

with two classes:

```text
NORMAL
PNEUMONIA
```

---

## Original Dataset Distribution

| Split | NORMAL | PNEUMONIA |
|---|---:|---:|
| Train | 1,341 | 3,875 |
| Validation | 8 | 8 |
| Test | 234 | 390 |

Two important problems are immediately visible.

### 1. Very Small Official Validation Set

The supplied validation set contains only:

```text
16 images
8 NORMAL
8 PNEUMONIA
```

which is too small for dependable model selection.

### 2. Class Imbalance

Approximately:

```text
74% of training images are PNEUMONIA
```

with roughly:

```text
2.9×
```

more pneumonia than normal images.

---

# ✂️ Rebuilt Train / Validation Split

Instead of using the original 16-image validation set, the notebook creates a new **stratified 90/10 split** from the original training data.

```text
New Train: 4,695 images
New Val:     521 images
Test:        624 images
```

The custom split preserves the NORMAL/PNEUMONIA class proportions.

---

# 🖼️ Image Preprocessing

Images are standardized to:

```text
224 × 224 × 3
```

and normalized using ImageNet statistics.

```python
IMG_SIZE = 224

NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD  = [0.229, 0.224, 0.225]
```

---

## Training Augmentation

```python
transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(0.5),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])
```

Validation and test images use deterministic preprocessing without random augmentation.

---

# ⚖️ Handling Class Imbalance

The project uses two complementary mechanisms.

## Weighted Sampling

```python
WeightedRandomSampler
```

balances the probability that images from each class appear during training.

---

## Weighted Binary Cross-Entropy

The training loss is:

```python
nn.BCEWithLogitsLoss(
    pos_weight=pos_weight
)
```

with the class weight calculated from the training distribution.

The approximate value is:

```text
pos_weight ≈ 0.346
```

---

# 🧠 SimpleCNN Architecture

The classifier is a lightweight custom CNN with four convolutional blocks.

```text
Input
224 × 224 × 3
       │
       ▼
Conv 3 → 32
BatchNorm
ReLU
MaxPool
       │
       ▼
Conv 32 → 64
BatchNorm
ReLU
MaxPool
       │
       ▼
Conv 64 → 128
BatchNorm
ReLU
MaxPool
       │
       ▼
Conv 128 → 256
BatchNorm
ReLU
MaxPool
       │
       ▼
Global Average Pool
       │
       ▼
256-dimensional embedding
       │
       ▼
Dropout
       │
       ▼
Linear Layer
       │
       ▼
1 Logit
       │
       ▼
Sigmoid
       │
       ▼
NORMAL / PNEUMONIA
```

The important part for this project is the:

```text
256-dimensional penultimate embedding
```

produced by the global pooling layer.

That same representation becomes the basis of the OOD detector.

---

# 🚨 The Out-of-Distribution Problem

The CNN is trained only on chest X-rays.

It therefore has no explicit mechanism for recognizing an image from outside its training distribution.

For example, the baseline classifier classified a **leg X-ray** as:

```text
99.9% PNEUMONIA
```

even though the image was not a chest X-ray.

The classifier itself was highly confident.

This demonstrates why classification confidence alone is not sufficient for OOD detection.

---

# 🔬 OOD Methods Considered

Several approaches were considered:

| Method | Advantage | Limitation |
|---|---|---|
| Confidence Thresholding | Extremely cheap | Neural networks can remain highly confident on OOD inputs |
| Temperature Scaling | Improves calibration | Does not solve OOD detection alone |
| MC Dropout | Measures prediction uncertainty | Requires multiple inference passes |
| Deep Ensembles | Strong uncertainty signal | Requires training multiple models |
| Bayesian Neural Networks | Principled uncertainty | Expensive and complex |
| Energy-Based Scoring | Cheap inference | Less suitable for this single-logit setup |
| **Mahalanobis Distance** | Cheap, embedding-based, no retraining | Requires threshold calibration |

The project selects:

# ✅ Mahalanobis Distance

because it works naturally with the CNN's existing 256-dimensional embedding.

---

# 🧮 Mahalanobis OOD Detection

The OOD detector operates on the representation immediately before the classifier head.

## Step 1 — Extract Embeddings

For every training image:

```text
Image
 ↓
CNN feature extractor
 ↓
Global Average Pool
 ↓
256-dimensional embedding
```

The final classification head is not required for this step.

---

## Step 2 — Learn Class Statistics

Embeddings are separated into:

```text
NORMAL embeddings
PNEUMONIA embeddings
```

The detector estimates:

- Mean vector for NORMAL
- Mean vector for PNEUMONIA
- Shared covariance matrix

---

## Step 3 — Score New Images

For a new embedding \(x\), the Mahalanobis distance to each class distribution is calculated.

Conceptually:

```text
New image embedding
       │
       ├── distance → NORMAL cluster
       │
       └── distance → PNEUMONIA cluster
                    │
                    ▼
            nearest distance
```

A small distance means:

```text
This looks similar to known training data.
```

A very large distance means:

```text
This embedding does not resemble either learned class.
```

---

# 🛡️ Two-Tier OOD Safety System

The detector uses two thresholds:

```text
NORMAL RANGE
     ↓
    WARN
     ↓
   REJECT
```

### Normal

The image resembles the known chest X-ray distribution.

The prediction is shown normally.

### Warn

The image looks unusual but is not extreme enough to automatically reject.

The classification is still produced, but the system flags the image as atypical.

### Reject

The embedding is sufficiently far from both class distributions.

The classifier prediction is withheld.

Instead of:

```text
PNEUMONIA
```

the system returns:

```text
Prediction withheld:
NOT A CHEST X-RAY
```

---

# 📏 OOD Threshold Calibration

The final detector uses percentile-based thresholds:

```text
WARN   = 99.0th percentile
REJECT = 99.9th percentile
```

This replaced earlier threshold strategies that were too sensitive to individual calibration samples.

The percentile approach is less dependent on a single maximum-distance outlier.

---

# 🐛 OOD Threshold Debugging

Two important failure cases shaped the final design.

## Bug 1 — Threshold Too Strict

An early OOD threshold was calibrated only using validation embeddings.

It incorrectly rejected a genuine chest X-ray from the test distribution.

### Change

The implementation expanded calibration to include the distribution of validation and test embeddings.

---

## Bug 2 — Threshold Too Loose

An earlier reject rule based on:

```text
1.5 × maximum calibration distance
```

allowed a leg X-ray to pass only as a warning.

The classifier itself assigned the image:

```text
99.9% PNEUMONIA
```

### Change

The maximum-based rule was replaced with:

```text
99.0th percentile → WARN
99.9th percentile → REJECT
```

---

# 🎚️ Classification Decision Threshold

OOD detection and pneumonia classification use **different thresholds**.

The OOD threshold answers:

```text
"Does this input resemble a chest X-ray?"
```

The classification threshold answers:

```text
"If it is accepted, should it be classified
as NORMAL or PNEUMONIA?"
```

The probability cutoff is selected from the validation ROC curve instead of simply assuming:

```text
0.5
```

---

## Youden's J Threshold

The ROC analysis produced:

```text
Youden threshold = 0.866
```

This emphasizes a balanced combination of sensitivity and specificity.

---

## Recall-Floor Threshold

The chosen project threshold is:

```text
0.787
```

It was selected as a threshold designed to preserve approximately:

```text
PNEUMONIA recall ≥ 95%
```

on the validation data.

The project deliberately prioritizes sensitivity because missing a genuine pneumonia case is considered more costly than producing an additional false alarm.

---

# 📈 Classification Threshold Results

## Default Threshold — 0.5

| Metric | Score |
|---|---:|
| Accuracy | 0.777 |
| Precision | 0.740 |
| Recall | **0.992** |
| F1-Score | 0.848 |

---

## Tuned Threshold — 0.787

| Metric | Score |
|---|---:|
| Accuracy | **0.830** |
| Precision | **0.800** |
| Recall | 0.950 |
| F1-Score | **0.870** |

Increasing the threshold substantially improves precision and overall F1 while maintaining the project's target of approximately 95% pneumonia recall.

---

# 📊 Baseline Model Performance

The baseline SimpleCNN achieves approximately:

```text
ROC-AUC = 0.927
```

on the test set.

Rather than replacing this classifier, the primary contribution of this project is adding a **safety mechanism around it**.

---

# 🧪 OOD Guard Example

A leg X-ray produced the following raw classifier output:

```text
P(PNEUMONIA) = 99.9%
```

Without OOD detection, that prediction could reach the user.

The OOD detector instead measured:

```text
Mahalanobis distance = 7,554.63
Reject threshold     = 2,249.55
```

The input was approximately:

```text
3.4×
```

beyond the reject threshold.

The final output therefore becomes:

```text
PREDICTION WITHHELD

NOT A CHEST X-RAY
```

rather than displaying the model's misleading 99.9% pneumonia prediction.

---

# 🔄 Full Inference Pipeline

```text
Uploaded Image
      │
      ▼
Resize + Normalize
      │
      ▼
SimpleCNN
      │
      ├───────────────┐
      │               │
      ▼               ▼
256-d Embedding    Classification Logit
      │               │
      ▼               ▼
Mahalanobis       Sigmoid Probability
Distance              │
      │               │
      ▼               ▼
OOD Decision      Threshold = 0.787
      │               │
      └───────┬───────┘
              ▼
       Final Safety Decision
```

Possible outputs:

```text
NORMAL
```

```text
PNEUMONIA
```

```text
PNEUMONIA [ATYPICAL — REVIEW]
```

or:

```text
PREDICTION WITHHELD
NOT A CHEST X-RAY
```

---

# 🖼️ Testing Your Own Image

The notebook supports direct uploads in Google Colab:

```python
from google.colab import files

uploaded = files.upload()
image_path = list(uploaded.keys())[0]
```

The prediction function performs both classification and OOD analysis.

```python
pred_class, prob_pneumonia, dist, ood_status, img = predict_image(
    image_path,
    baseline_model,
    ood_detector=ood_detector,
    decision_threshold=DECISION_THRESHOLD
)
```

---

# 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- **PyTorch/XLA**
- **Google Colab**
- **TPU v5e-1 / V5E1**
- KaggleHub
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Pillow
- Scikit-learn
- Jupyter Notebook

---

# 📁 Project Structure

```text
pneumonia-ood-detection/
│
├── colab_code_tpu_with_ood_1.ipynb
├── Pneumonia_OOD_Project_Presentation.pptx
└── README.md
```

The chest X-ray dataset is downloaded programmatically and does not need to be included in the repository.

---

# 🚀 Running the Project

## Recommended Environment

This project is designed for:

```text
Google Colab
+
TPU v5e-1
+
PyTorch/XLA
```

---

## 1. Open the Notebook in Google Colab

Open:

```text
colab_code_tpu_with_ood_1.ipynb
```

---

## 2. Select a TPU Runtime

Configure Colab to use:

```text
TPU
```

with the notebook targeting:

```text
V5E1 / TPU v5e-1
```

---

## 3. Install PyTorch/XLA

Run the included setup cell:

```bash
pip install -q -U 'torch_xla[tpu]' \
-f https://storage.googleapis.com/libtpu-releases/index.html
```

---

## 4. Verify the TPU

```python
import torch_xla.core.xla_model as xm

device = xm.xla_device()

print(device)
print(xm.xla_device_hw(device))
```

---

## 5. Download the Dataset

```python
import kagglehub

path = kagglehub.dataset_download(
    "paultimothymooney/chest-xray-pneumonia"
)
```

---

## 6. Run the Notebook in Order

The workflow is:

```text
Dataset Download
       ↓
Dataset Audit
       ↓
Image Preprocessing
       ↓
Stratified Train / Validation Split
       ↓
Class-Imbalance Handling
       ↓
SimpleCNN Training on TPU
       ↓
Test Evaluation
       ↓
256-d Embedding Extraction
       ↓
Mahalanobis OOD Fitting
       ↓
WARN / REJECT Calibration
       ↓
ROC-Based Decision Threshold
       ↓
Safe Image Inference
```

---

# 💡 Key Takeaways

- Binary classifiers naturally force every input into one of their known classes.
- High sigmoid confidence does **not** prove that an input belongs to the training distribution.
- A leg X-ray can receive an extremely confident pneumonia score from a chest X-ray classifier.
- The SimpleCNN's existing 256-dimensional embedding can be reused for OOD detection without changing the model architecture.
- Mahalanobis distance provides a computationally inexpensive OOD signal.
- A two-tier system allows suspicious inputs to be either warned on or rejected.
- Classification and OOD detection require separate thresholds.
- ROC analysis provides a more defensible pneumonia decision threshold than choosing one arbitrarily.
- The entire training workflow is built for **Google Colab TPU v5e-1 using PyTorch/XLA**.
- OOD detection adds an important safety layer, but it does not make the model clinically validated.

---

# 🔮 Future Improvements

Potential extensions include:

- Add a dedicated **"Is this a chest X-ray?"** model before pneumonia classification
- Compare Mahalanobis distance against **deep ensembles**
- Test on an external chest X-ray dataset
- Evaluate OOD AUROC / AUPR using a dedicated external OOD benchmark
- Compare with libraries such as PyTorch-OOD or OpenOOD
- Evaluate calibration across different hospitals and imaging devices
- Add uncertainty calibration
- Introduce patient-level and institution-level external validation

---

# ⚠️ Methodological Note

The current notebook uses **validation + test image embeddings, without labels, to calibrate the OOD distance thresholds**.

For a stricter experimental evaluation, a future version should use a dedicated calibration dataset and keep the final test distribution completely isolated until the last evaluation step.

This does not change how the implemented demo works, but it would make future benchmarking methodologically cleaner.

---

# 🩺 Medical Disclaimer

This project is intended for:

```text
Educational
Research
Portfolio
Experimental ML safety
```

purposes only.

It has **not been clinically validated** and should not be used to:

- Diagnose pneumonia
- Replace a radiologist
- Replace a physician
- Make treatment decisions
- Provide medical advice

The OOD layer also does **not guarantee** detection of every invalid or unusual input.

It is an experimental safety mechanism intended to reduce the risk of blindly trusting predictions outside the model's training distribution.

---

# 📚 Dataset

Chest X-Ray Images (Pneumonia)

Kaggle dataset:

```text
paultimothymooney/chest-xray-pneumonia
```

---

# 👤 Author

**Medical Imaging / Deep Learning / OOD Detection Project**

Built with **PyTorch + PyTorch/XLA** and designed for **Google Colab TPU v5e-1 (`V5E1`)**.

---

⭐ If you find the project useful, consider starring the repository.
