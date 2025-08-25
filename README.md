This repository is based on the **HAM10000 dataset**, a large collection of dermatoscopic images of skin lesions.  

- **Aim of the project**  
  - To classify 7 types of skin lesions from the HAM10000 dataset using deep convolutional neural networks (DCNNs), by fine-tuning pretrained models (VGG16, InceptionV3, Inception-ResNetV2, DenseNet201) and comparing their performance.
  - The project specifically explores whether transfer learning (using ImageNet-trained models) can adapt well to medical images (dermatoscopic lesions).

- ## Progress so far
### 1. Exploratory Data Analysis (EDA)
- Linked 10,015 image files with metadata  
- Added columns:  
  - `cell_type` (human-readable class name)  
  - `cell_type_idx` (numeric code 0–6)  
  - `Malignant` (binary: benign=0, malignant=1)  
- Explored dataset imbalance (many benign nevi vs fewer malignant cases)  
- Visualized distributions:
  - Lesion type counts  
  - Benign vs malignant cases  
  - Age, sex, body localization, and diagnosis method  
- Color analysis:
  - Mean RGB + grayscale values per image  
  - Sampled pairplots of lesion separability  
- Stratified **train/validation/test** split (72/18/10%)  
- Saved metadata as `data/processed/meta_ready.parquet` for reproducibility

### 2. Baseline CNN (Scratch)
- Small CNN trained on **64×64 resized images**  
- Architecture: 3 conv + pooling blocks + dense head  
- Data augmentation (random flips, shuffling)  
- Used class weights to handle imbalance  
- **Result:** ~54% test accuracy (mostly predicting “nevi”)  
- Served as a benchmark for transfer learning

### 3. VGG16 (Transfer Learning)
- Input size **224×224**  
- Stage A: freeze base, train new head (3 epochs)  
- Stage B: unfreeze `block5_*` and fine-tune (20 epochs, lr=1e-4)  
- **Result:** ~62% test accuracy  
  - Stronger recognition of minority classes (Actinic keratoses recall ~0.79)  
  - Weighted F1 ~0.65 (up from 0.53 baseline)  
- Confirmed benefit of transfer learning

---

## Dataset

**HAM10000 ("Human Against Machine with 10000 training images")**

- **10,015 dermatoscopic images**
- **7 classes of skin lesions**:
  - `nv` → Melanocytic nevi  
  - `mel` → Melanoma  
  - `bkl` → Benign keratosis-like lesions  
  - `bcc` → Basal cell carcinoma  
  - `akiec` → Actinic keratoses  
  - `vasc` → Vascular lesions  
  - `df` → Dermatofibroma  

- Metadata (`HAM10000_metadata.tab`) includes:
  - `lesion_id`, `image_id`, `dx` (diagnosis), `age`, `sex`, `localization`, `dx_type`

---

## EDA Highlights

- **Class imbalance**: ~7000 samples are benign nevi (`nv`), while malignant classes like melanoma are underrepresented.
- **Binary grouping**: Labels also mapped to **Benign (0)** vs **Malignant (1)**.
- **Explored metadata**:
  - Distribution of benign vs malignant cases
  - Counts for each lesion type
  - Distribution of lesion localization (body sites)
  - Diagnosis methods (`dx_type`)
  - Age distribution (overall + malignant only)
  - Sex distribution (overall + malignant only)

- **Visualizations**:
  - Sample grids of lesion types
  - Color channel statistics (mean Red, Green, Blue, Gray values per image)
  - Pairplots to see separability by color features
  - Variations of lesion appearance as RGB values change

---

## Preprocessing Steps

- **Image loading**: linked each `image_id` to its `.jpg` path.
- **Added columns**:
  - `cell_type` → full human-readable class name
  - `binary_label` → 0 = benign, 1 = malignant
  - `cell_type_idx` → numeric code for each lesion type

- **Resizing**:
  - Images resized to **64×64** for quick baselines
  - Images resized to **192×256** for fine-tuning CNN models
  - Images resized to **299×299** when preparing directories for Inception-style models

- **Standardization**:
  - Converted to RGB
  - Normalized pixel values to `[0,1]`

- **Data splits**:
  - Train / Validation / Test created using `train_test_split`
  - Saved preprocessed arrays to `.npy` for reuse:
    - `256_192_train.npy`, `256_192_val.npy`, `256_192_test.npy`
    - Corresponding label arrays

- **Alternative format**:
  - Flattened images into vectors (like MNIST) and saved as `hmnist_64_64_RGB.csv`

---

## Next Steps

Next Steps
- Implement **InceptionV3** fine-tuning (299×299 inputs)  
- Add **DenseNet201** (full model fine-tuning)  
- Combine models into an **ensemble**   
- Document final results and comparisons


---

## Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
```
