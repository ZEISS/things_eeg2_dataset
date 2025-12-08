<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/assets/things_eeg2_dataset-banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/assets/things_eeg2_dataset-banner-light.png">
  <img alt="things_eeg2_dataset" src="./.github/assets/things_eeg2_dataset-banner-light.png">
</picture>

<div align="center">

[![License][license-badge]](LICENSE)
[![CI Status][ci-badge]][ci]
<!-- [![Binary Build][binary-build-badge]][binary-build]
[![Conda Platform][conda-badge]][conda-url]
[![Conda Downloads][conda-downloads-badge]][conda-url]
[![Codecov][codecov]][codecov-url] -->

[license-badge]: https://img.shields.io/github/license/zeiss/things_eeg2_dataset?style=flat-square

[ci-badge]: https://img.shields.io/github/actions/workflow/status/zeiss/things_eeg2_dataset/ci.yml?branch=main&style=flat-square&label=CI
[ci]: https://github.com/zeiss/things_eeg2_dataset/actions/workflows/ci.yml

<!-- [binary-build-badge]: https://img.shields.io/github/actions/workflow/status/zeiss/things_eeg2_dataset/build.yml?branch=main&style=flat-square&label=Binary%20Build
[binary-build]: https://github.com/zeiss/things_eeg2_dataset/actions/workflows/build.yml -->

<!-- [conda-badge]: https://img.shields.io/conda/vn/conda-forge/things_eeg2_dataset?style=flat-square
[conda-downloads-badge]: https://img.shields.io/conda/dn/conda-forge/things_eeg2_dataset?style=flat-square
[conda-url]: https://prefix.dev/channels/conda-forge/packages/things_eeg2_dataset -->

<!-- [codecov]: https://img.shields.io/codecov/c/github/zeiss/things_eeg2_dataset/main?style=flat-square
[codecov-url]: https://codecov.io/gh/zeiss/things_eeg2_dataset -->

</div>

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
      <ul>
        <li><a href="#built-with">Built With</a></li>
      </ul>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#usage">Usage</a></li>
    <li><a href="#roadmap">Roadmap</a></li>
    <li><a href="#contributing">Contributing</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#acknowledgments">Acknowledgments</a></li>
  </ol>
</details>

# THINGS-EEG2 Raw Data Processing

This package provides tools for downloading, preprocessing raw THINGS-EEG2 EEG data, and generating image embeddings from various vision models.

> [!WARNING]
> This repository builds upon the original data processing by [Gifford et al (2022)](https://github.com/gifale95/eeg_encoding).
> Please check out their original code and the [corresponding paper](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub).
> We are in no way associated with the authors, but have conducted research using the data and code they provide.
> To make working with the dataset even easier, we wrote these utilities and hope that they will make things easier (pun intended) for other people as well.

## Data Structure

The pipeline expects and creates the following directory structure:

## Output structure

After running the downloading (which creates `raw_data/` and `source_data/`), preprocessing (which creates `processed/`, `DATA_VERSION.txt`) and embedding generation, you will get the following directory structure.

```bash
project_dir
└── DATA_VERSION.txt                              # Contains the commit hash of the package version
└── Image_set                                     # Raw data downloaded by the downloader
    └── test_images
        └── {XXXXX}_{image_condition}
            └── {image_condition}_{XX}.jpg
            └── ...
        └── ...
    └── training_images
        └── {XXXXX}_{image_condition}
            └── {image_condition}_{XX}.jpg
            └── ...
        └── ...
    └── image_metadata.npy
    └── LICENSE.txt
        └── ses-{XX}
            └── raw_eeg_test.npy
            └── raw_eeg_training.npy
        └── ...
└── raw_data                                      # Raw data downloaded by the downloader
    └── sub-{XX}
        └── ses-{XX}
            └── raw_eeg_test.npy
            └── raw_eeg_training.npy
        └── ...
└── source_data/                                  # Default directory (configurable)
    └── sub-{XX}/
        ├── preprocessed_eeg_training.npy         # Shape: (sessions, images_conditions, repetitions, channels, timepoints)
        ├── preprocessed_eeg_test.npy             # Shape: (sessions, images_conditions, repetitions, channels, timepoints)
        └── img_conditions_test.npy               # Shape: (sessions, images_conditions)
        └── img_conditions_training.npy           # Shape: (sessions, images_conditions)
        └── meta.json                             # Keys: ch_names & times
    └── ...
└── processed/                                    # Default directory (configurable)
    └── sub-{XX}/
        ├── preprocessed_eeg_training.npy         # Shape: (sessions, images_conditions, repetitions, channels, timepoints)
        ├── preprocessed_eeg_test.npy             # Shape: (sessions, images_conditions, repetitions, channels, timepoints)
        └── img_conditions_test.npy               # Shape: (sessions, images_conditions)
        └── img_conditions_training.npy           # Shape: (sessions, images_conditions)
        └── meta.json                             # Keys: ch_names & times
    └── ...
└── embeddings/
    ├── ViT-H-14_features_train.pt                # Pooled embeddings
    ├── ViT-H-14_features_train_full.pt           # Full token sequences
    ├── ViT-H-14_features_test.pt
    └── ViT-H-14_features_test_full.pt
```

### Embedding Generation (`embedding_processing/`)

The package supports multiple state-of-the-art vision models for generating image embeddings:

| Model | Embedder Class | Description |
|-------|----------------|-------------|
| `open-clip-vit-h-14` | `OpenClipViTH14Embedder` | OpenCLIP ViT-H/14 (SDXL image encoder) |
| `openai-clip-vit-l-14` | `OpenAIClipVitL14Embedder` | OpenAI CLIP ViT-L/14 |
| `dinov2` | `DinoV2Embedder` | DINOv2 with registers (self-supervised) |
| `ip-adapter` | `IPAdapterEmbedder` | IP-Adapter Plus projections |

Each embedder generates:
- **Pooled embeddings**: Single vector per image (e.g., `(1024,)` for ViT-H-14)
- **Full sequence embeddings**: All tokens (e.g., `(257, 1280)` for ViT-H-14)
- **Text embeddings**: Corresponding text features from image captions

**Output Files:**
```
embeddings/
├── ViT-H-14_features_train.pt           # Pooled embeddings
├── ViT-H-14_features_train_full.pt      # Full token sequences
├── ViT-H-14_features_test.pt
└── ViT-H-14_features_test_full.pt
```

## Installation

```bash
# Clone this repository, cd into it and run
uv sync
uv pip install --editable .
```

## Usage

```bash
python things-eeg2 process \
    --project_dir /path/to/things_eeg2 \
    --subjects 1 2 3 4 5 6 7 8 9 10 \
```

## References

- Original THINGS-EEG2 paper and code
- Implementation based on: https://www.sciencedirect.com/science/article/pii/S1053811922008758


### Using the dataloader

```python
from things_eeg2_dataset.dataloader import ThingsEEGDataset

dataset = ThingsEEGDataset(
    image_model="ViT-H-14",
    data_path="/path/to/processed_data",
    img_directory_training="/path/to/images/train",
    img_directory_test="/path/to/images/test",
    embeddings_dir="/path/to/embeddings",
    train=True,
    time_window=(0.0, 1.0),
)
```

See `things_eeg2_dataloader/README.md` for detailed usage.

## Citation

We are happy users of the THINGS-EEG2 dataset, but not associated with the original authors.
If you use this code, please cite the THINGS-EEG2 paper:
> Gifford, A. T., Lahner, B., Saba-Sadiya, S., Vilas, M. G., Lascelles, A., Oliva, A., ... & Cichy, R. M. (2022). The THINGS-EEG2 dataset. Scientific Data.

## License

This project follows the original THINGS-EEG2 license terms.
