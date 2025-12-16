<picture>
  <source media="(prefers-color-scheme: dark)" srcset="./.github/assets/things_eeg2_dataset-banner-dark.png">
  <source media="(prefers-color-scheme: light)" srcset="./.github/assets/things_eeg2_dataset-banner-light.png">
  <img alt="things_eeg2_dataset" src="./.github/assets/things_eeg2_dataset-banner-light.png">
</picture>

<div align="center">

[![PyPI][pypi-badge]][pypi]
[![License][license-badge]][license-url]
[![CI Status][ci-badge]][ci-url]
<!-- [![Conda Platform][conda-badge]][conda-url] -->

[pypi-badge]: https://img.shields.io/pypi/v/things_eeg2_dataset?style=flat-square&label=PyPI
[pypi]: https://pypi.org/project/things-eeg2-dataset/

[license-badge]: https://img.shields.io/badge/License-CC%20BY--NC%204.0-yellow.svg?style=flat-square
[license-url]: LICENSE

[ci-badge]: https://img.shields.io/github/actions/workflow/status/zeiss/things_eeg2_dataset/ci.yml?branch=main&style=flat-square&label=CI
[ci-url]: https://github.com/zeiss/things_eeg2_dataset/actions/workflows/ci.yml


<!-- [conda-badge]: https://img.shields.io/conda/vn/conda-forge/things_eeg2_dataset?style=flat-square -->

</div>

# THINGS-EEG2 Raw Data Processing

This package provides tools for downloading, preprocessing raw THINGS-EEG2 EEG data, and generating image embeddings from various vision models.

> [!WARNING]
> This repository builds upon the original data processing by [Gifford et al (2022)](https://github.com/gifale95/eeg_encoding).
> Please check out their original code and the [corresponding paper](https://www.sciencedirect.com/science/article/pii/S1053811922008758?via%3Dihub).
>
> We are in no way associated with the authors.
> Nonetheless we hope, that this makes things easier (pun intended) to use.

## Installation

```bash
git clone git@github.com:ZEISS/things_eeg2_dataset.git
cd things_eeg2_dataset

uv sync
uv pip install --editable .
source .venv/bin/activate

things-eeg2 --help
things-eeg2 --install-completion

# Then restart your terminal.
# Example for zsh:
source ~/.zshrc
```

## Usage

```bash
uv run things-eeg2 process \
    --project_dir /path/to/project_dir \
    --subjects 1 2 3 4 5 6 7 8 9 10 \


uv run things-eeg2 info \
    --project-dir /path/to/project_dir \
    --subject <EXAMPLE_SUBJECT \
    --session <EXAMPLE_SESSION> \
    --data-index <EXAMPLE_INDEX>
```

## Data Structure

For understanding the data structure that is created by the CLI (and then needed to perform proper preprocessing and loading), please see [paths.py](src/things_eeg2_dataset//paths.py). All code is configured to use the structure defined there as its ground truth.

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
├── ViT-H-14_features_training.pt           # Pooled embeddings
├── ViT-H-14_features_training_full.pt      # Full token sequences
├── ViT-H-14_features_test.pt
└── ViT-H-14_features_test_full.pt
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
