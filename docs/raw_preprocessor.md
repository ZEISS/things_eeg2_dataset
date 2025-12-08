# RawProcessor - Modular EEG Preprocessing

This package provides a modular, object-oriented approach to preprocessing raw EEG data for the Things-EEG2 dataset.

## Overview

The `RawProcessor` class encapsulates the complete preprocessing pipeline for raw EEG data, including:

- **Channel selection**: Selecting specific EEG channels for analysis
- **Epoching**: Extracting time-locked segments around stimuli
- **Baseline correction**: Removing pre-stimulus baseline activity
- **Frequency downsampling**: Reducing sampling rate for computational efficiency
- **Multivariate Noise Normalization (MVNN)**: Whitening the data to remove noise correlations
- **Data organization**: Sorting and reshaping data by image conditions

## Installation

Make sure you have the required dependencies installed:

```bash
pip install numpy scipy scikit-learn mne tqdm
```

## Quick Start

### Command-Line Interface

The simplest way to use the preprocessor is through the command-line script:

```bash
python -m things_eeg2_raw_processing.preprocessing \
    --sub 1 \
    --n_ses 4 \
    --sfreq 250 \
    --mvnn_dim epochs \
    --project_dir /path/to/your/project
```

### Python API - Complete Pipeline

```python
from things_eeg2_raw_processing.preprocessor import RawProcessor

# Create processor instance
processor = RawProcessor(
    sub=1,                          # Subject number
    n_ses=4,                        # Number of sessions
    sfreq=250,                      # Downsampling frequency (Hz)
    mvnn_dim="epochs",              # MVNN dimension: 'time' or 'epochs'
    project_dir="/path/to/project", # Project directory
    seed=20200220                   # Random seed for reproducibility
)

# Run complete pipeline
processor.run()
```

### Python API - Step-by-Step Processing

For more control over the preprocessing pipeline:

```python
from things_eeg2_raw_processing.preprocessor import RawProcessor

# Initialize processor
processor = RawProcessor(
    sub=1,
    n_ses=4,
    sfreq=250,
    mvnn_dim="epochs",
    project_dir="/path/to/project"
)

# Step 1: Epoch and sort the data
processor.epoch_and_sort()
print(f"Channels: {len(processor.ch_names)}")
print(f"Time points: {len(processor.times)}")

# Step 2: Apply multivariate noise normalization
processor.apply_mvnn()

# Step 3: Save preprocessed data
processor.save_preprocessed_data()
```

## Class Reference

### RawProcessor

#### Parameters

- **sub** (int): Subject number to process
- **n_ses** (int, optional): Number of EEG sessions (default: 4)
- **sfreq** (int, optional): Downsampling frequency in Hz (default: 250)
- **mvnn_dim** (str, optional): MVNN dimension mode - 'time' or 'epochs' (default: 'epochs')
- **project_dir** (str): Directory of the project folder containing raw data
- **seed** (int, optional): Random seed for reproducibility (default: 20200220)

#### Methods

##### `run()`
Executes the complete preprocessing pipeline in one step.

##### `epoch_and_sort()`
Performs channel selection, epoching, baseline correction, and frequency downsampling. Reshapes data to: Image conditions × EEG repetitions × EEG channels × EEG time points

##### `apply_mvnn()`
Applies Multivariate Noise Normalization to whiten the EEG data.

##### `save_preprocessed_data()`
Merges data from all sessions and saves the preprocessed test and training partitions.

##### `get_preprocessing_info()`
Returns a dictionary containing preprocessing parameters and data information.

#### Attributes

- **epoched_test**: Epoched test data (list of numpy arrays)
- **epoched_train**: Epoched training data (list of numpy arrays)
- **whitened_test**: Whitened test data after MVNN (list of numpy arrays)
- **whitened_train**: Whitened training data after MVNN (list of numpy arrays)
- **ch_names**: EEG channel names (list of strings)
- **times**: EEG time points (numpy array)
- **img_conditions_train**: Image conditions for training data (list of numpy arrays)

## Advanced Usage

### Custom Processing with Data Access

```python
from things_eeg2_raw_processing.preprocessor import RawProcessor
import numpy as np

processor = RawProcessor(
    sub=1,
    n_ses=4,
    sfreq=250,
    mvnn_dim="epochs",
    project_dir="/path/to/project"
)

# Run epoching
processor.epoch_and_sort()

# Access and inspect epoched data before MVNN
if processor.epoched_test is not None:
    # Get shape information
    n_sessions = len(processor.epoched_test)
    img_conds, reps, channels, timepoints = processor.epoched_test[0].shape

    print(f"Sessions: {n_sessions}")
    print(f"Image conditions: {img_conds}")
    print(f"Repetitions: {reps}")
    print(f"Channels: {channels}")
    print(f"Time points: {timepoints}")

    # Perform custom analysis on epoched data
    # ... your custom code here ...

# Continue with MVNN
processor.apply_mvnn()

# Access whitened data
if processor.whitened_test is not None:
    # Perform custom analysis on whitened data
    # ... your custom code here ...
    pass

# Save results
processor.save_preprocessed_data()
```

### Getting Preprocessing Information

```python
from things_eeg2_raw_processing.preprocessor import RawProcessor

processor = RawProcessor(
    sub=1,
    n_ses=4,
    sfreq=250,
    mvnn_dim="epochs",
    project_dir="/path/to/project"
)

processor.epoch_and_sort()
processor.apply_mvnn()

# Get comprehensive info
info = processor.get_preprocessing_info()

print("Parameters:")
for key, value in info["parameters"].items():
    print(f"  {key}: {value}")

print("\nData Info:")
for key, value in info["data_info"].items():
    print(f"  {key}: {value}")
```

## Data Output

Preprocessed data is saved in the following structure:

```
<project_dir>/
└── processed/
    └── sub-<XX>/
        ├── preprocessed_eeg_test.npy
        └── preprocessed_eeg_training.npy
```

Each `.npy` file contains a pickled dictionary with:
- `preprocessed_eeg_data`: numpy array of shape (image_conditions, repetitions, channels, timepoints)
- `ch_names`: list of channel names
- `times`: array of time points

## Migration from Old Script

If you were using the old `preprocessing.py` script, the command-line interface remains compatible:

**Old usage:**
```bash
python preprocessing.py --sub 1 --n_ses 4 --sfreq 250 --mvnn_dim epochs --project_dir /path
```

**New usage (same command works):**
```bash
python -m things_eeg2_raw_processing.preprocessing --sub 1 --n_ses 4 --sfreq 250 --mvnn_dim epochs --project_dir /path
```

## Benefits of the Modular Approach

1. **Reusability**: Use `RawProcessor` in your own scripts and notebooks
2. **Flexibility**: Run individual preprocessing steps or the complete pipeline
3. **Testability**: Easier to test individual components
4. **Maintainability**: Clear separation of concerns and better code organization
5. **Extensibility**: Easy to subclass and customize for specific needs
6. **Data Access**: Access intermediate results for custom analysis

## References

This preprocessing pipeline is based on:

> Gifford, A. T., Dwivedi, K., Roig, G., & Cichy, R. M. (2022). A large and rich EEG dataset for modeling human visual object recognition. *NeuroImage*, 264, 119754.
> https://www.sciencedirect.com/science/article/pii/S1053811922008758

## License

Please refer to the main project license.
