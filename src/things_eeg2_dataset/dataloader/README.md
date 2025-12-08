# THINGS-EEG2 DataLoader

PyTorch Dataset and DataModule for loading preprocessed THINGS-EEG2 data with image embeddings for neural decoding experiments.

## Features

- **Lazy Loading**: Efficient memory usage by loading only required subjects and time windows
- **Index-Based Lookup**: Fast O(1) access to EEG epochs, images, and embeddings via CSV indices
- **Flexible Embedding Support**: Works with multiple vision models (CLIP, DINOv2, IP-Adapter)
- **Multi-Subject Support**: Load and combine data from multiple subjects
- **Lightning Integration**: Built-in PyTorch Lightning DataModule for easy training
- **Validation Splits**: Automatic train/val splitting with configurable strategies
- **Time Window Selection**: Extract specific time ranges from EEG epochs

## Installation

The dataloader is part of the `things_eeg2_dataset` workspace. Install with:

```bash
cd things_eeg2_dataloader
uv sync
```

## Quick Start

### Basic Usage

```python
from things_eeg2_dataloader import ThingsEEGDataset

# Training dataset
train_dataset = ThingsEEGDataset(
    image_model="ViT-H-14",
    data_path="/path/to/processed_data",
    img_directory_training="/path/to/images/train",
    img_directory_test="/path/to/images/test",
    embeddings_dir="/path/to/embeddings",
    train=True,
    subjects=["sub-01", "sub-02", "sub-03"],
    time_window=(0.0, 1.0),  # seconds
    load_images=False,
)

# Access single item
item = train_dataset[0]
print(item.brain_signal.shape)    # (channels, timepoints)
print(item.embedding.shape)        # (embed_dim,) or (n_tokens, embed_dim)
print(item.subject)                # subject index
print(item.image_id)               # image index
print(item.text)                   # image caption
```

### Using PyTorch DataLoader

```python
from torch.utils.data import DataLoader

dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4,
)

for batch in dataloader:
    eeg = batch.brain_signal        # (batch, channels, time)
    embeddings = batch.embedding    # (batch, embed_dim)
    # ... training code
```

### Using Lightning DataModule

```python
from things_eeg2_dataloader import ThingsEEGDataModule

datamodule = ThingsEEGDataModule(
    image_model="ViT-H-14",
    data_path="/path/to/processed_data",
    img_directory_training="/path/to/images/train",
    img_directory_test="/path/to/images/test",
    embeddings_dir="/path/to/embeddings",
    subjects=["sub-01", "sub-02"],
    time_window=(0.0, 1.0),
    batch_size=32,
    num_workers=4,
    val_split_classes=200,  # Hold out 200 classes for validation
)

# Use with Lightning Trainer
import lightning as L

trainer = L.Trainer(max_epochs=100)
trainer.fit(model, datamodule=datamodule)
```

## Dataset Output

The `ThingsEEGDataset` returns `ThingsEEGItem` dataclass instances with the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `brain_signal` | `torch.Tensor` | EEG data, shape `(channels, timepoints)` |
| `embedding` | `torch.Tensor` | Image embedding, shape varies by model |
| `subject` | `int` | Subject index (0-indexed) |
| `image_id` | `int` | Unique image identifier |
| `image_class` | `int` | Image class/category ID |
| `sample_id` | `int` | Sample ID within class |
| `repetition` | `int` | EEG repetition number (or -1 for averaged test data) |
| `channel_positions` | `torch.Tensor` | 2D channel positions, shape `(channels, 2)` |
| `text` | `str` | Image caption |
| `image` | `Path` or `Tensor` | Image path or loaded tensor (if `load_images=True`) |

## Configuration Options

### Dataset Parameters

```python
ThingsEEGDataset(
    image_model: str,                    # Embedding model name (e.g., "ViT-H-14")
    data_path: str,                      # Path to processed EEG data
    img_directory_training: str,         # Path to training images
    img_directory_test: str,             # Path to test images
    embeddings_dir: str,                 # Path to embedding files
    embed_stats_dir: str | None = None,  # Optional normalization statistics
    normalize_embed: bool = True,        # Normalize embeddings
    flat_embed: bool = False,            # Flatten multi-token embeddings
    subjects: list[str] | None = None,   # Subject IDs (default: all 10)
    exclude_subs: list[str] | None = None,  # Subjects to exclude
    train: bool = True,                  # Training (True) or test (False) partition
    time_window: tuple = (0, 1.0),       # Time window in seconds
    load_images: bool = False,           # Load actual images (vs. paths)
)
```

### DataModule Parameters

The `ThingsEEGDataModule` accepts all dataset parameters plus:

```python
ThingsEEGDataModule(
    # ... all ThingsEEGDataset params ...
    batch_size: int = 32,
    num_workers: int = 4,
    val_split_classes: int = 200,        # Classes for validation split
    retrieval_set_size: int = 16540,     # Size of train-eval retrieval set
    seed: int = 42,
)
```

## Index Structure

The dataloader relies on index CSV files created by the `things_eeg2_raw_processing` package:

### EEG Index Files
- `training_eeg_index.csv`: Maps global dataset indices to subjects, images, and metadata
- `test_eeg_index.csv`: Same for test partition

Required columns:
- `global_index`: Unique row ID
- `subject`: Subject identifier (e.g., "sub-01")
- `class_id`, `sample_id`, `repetition`: Stimulus identifiers
- `image_index`: Links to image embeddings

### Image Index Files
- `training_image_index.csv`: Image metadata for training set
- `test_image_index.csv`: Image metadata for test set

Required columns:
- `image_index`: Unique image ID
- `image_path`: Relative path to image
- `caption`: Text description
- `class_id`, `sample_id`: Image identifiers

## Supported Embedding Models

| Model String | Embedding Shape (pooled) | Embedding Shape (full) |
|--------------|--------------------------|------------------------|
| `ViT-H-14` (OpenCLIP) | `(1024,)` | `(257, 1280)` |
| `openai_ViT-L-14` | `(768,)` | `(257, 768)` |
| `dinov2-reg` | `(768,)` | `(261, 768)` |
| `ip-adapter-plus-vit-h-14` | `(16, 1280)` | N/A |

## Data Shapes

**Training Data (per subject):**
- Dataset length: `1654 × 10 × 4 = 66,160` items (or `16,540` if repetitions averaged)
- EEG shape: `(63 channels, time_points_in_window)`

**Test Data (per subject):**
- Dataset length: `200` items (repetitions averaged)
- EEG shape: `(63 channels, time_points_in_window)`

## Validation Splitting

The DataModule supports flexible validation splitting:

```python
# Hold out 200 classes for validation (one sample per class)
datamodule = ThingsEEGDataModule(
    val_split_classes=200,
    seed=42,
    # ... other params
)

# Access splits
train_loader = datamodule.train_dataloader()
val_loader = datamodule.val_dataloader()
train_eval_loader = datamodule.train_eval_dataloader()  # For retrieval metrics
```

## Advanced Features

### Custom Time Windows

```python
# Extract 200ms-800ms post-stimulus
dataset = ThingsEEGDataset(
    time_window=(0.2, 0.8),
    # ... other params
)
```

### Multi-Token Embeddings

```python
# Use full token sequences (e.g., for attention models)
dataset = ThingsEEGDataset(
    image_model="ViT-H-14_full",  # Loads full sequence
    flat_embed=False,              # Keep (n_tokens, dim) shape
    # ... other params
)
```

### Loading Actual Images

```python
# Load PIL images (slower, higher memory)
dataset = ThingsEEGDataset(
    load_images=True,
    # ... other params
)

item = dataset[0]
print(type(item.image))  # PIL.Image.Image or torch.Tensor
```

## Example: Training Loop

```python
import torch
from things_eeg2_dataloader import ThingsEEGDataModule

# Setup data
datamodule = ThingsEEGDataModule(
    image_model="ViT-H-14",
    data_path="./processed",
    img_directory_training="./images/train",
    img_directory_test="./images/test",
    embeddings_dir="./embeddings",
    subjects=["sub-01"],
    time_window=(0.0, 1.0),
    batch_size=64,
)

# Training
for epoch in range(100):
    for batch in datamodule.train_dataloader():
        eeg = batch.brain_signal        # (batch, 63, time)
        target_embed = batch.embedding  # (batch, 1024)

        # Forward pass
        pred_embed = model(eeg)
        loss = criterion(pred_embed, target_embed)

        # Backward pass
        loss.backward()
        optimizer.step()
```

## Requirements

- Python 3.10+
- PyTorch
- PyTorch Lightning
- pandas, numpy
- PIL (if loading images)

## See Also

- **Raw processing**: `../things_eeg2_raw_processing/README.md`
- **Missing features**: `../docs/MISSING_FEATURES.md`
- **Dataset paper**: [THINGS-EEG2 (Gifford et al., 2022)](https://www.sciencedirect.com/science/article/pii/S1053811922008758)
