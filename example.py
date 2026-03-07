from pathlib import Path

from things_eeg2_dataset.dataloader import (
    DataModuleConfig,
    DatasetConfig,
    ThingsEEGDataModule,
    ThingsEEGDataset,
)

if __name__ == "__main__":
    # ---------- How to use the Dataset ----------
    train_ds = ThingsEEGDataset(
        DatasetConfig(
            partition="training",
            subjects=list(range(1, 11)),
            project_dir=Path.home() / "things_eeg2",
            image_model="siglip2-base-patch16-224",
            embed_variant="pooled",
            time_window=(0.0, 1.0),
        )
    )

    test_ds = ThingsEEGDataset(
        DatasetConfig(
            partition="test",
            subjects=list(range(1, 11)),
            project_dir=Path.home() / "things_eeg2",
            image_model="siglip2-base-patch16-224",
            embed_variant="pooled",
            time_window=(0.0, 1.0),
        )
    )

    print("train dataset length:", len(train_ds))
    print("test dataset length:", len(test_ds))
    print("eeg shape:", train_ds[0].brain_signal.shape)

    # ---------- How to use the DataModule ----------
    datamodule = ThingsEEGDataModule(
        DataModuleConfig(
            subjects=list(range(1, 11)),
            project_dir=Path.home() / "things_eeg2",
            image_model="siglip2-base-patch16-224",
            embed_variant="pooled",
            time_window=(0.0, 1.0),
            batch_size=4,
        )
    )
    datamodule.setup()

    train_loader = datamodule.train_dataloader()
    batch = next(iter(train_loader))

    print("batch size:", batch.brain_signal.shape[0])
    print("eeg shape:", batch.brain_signal.shape)
