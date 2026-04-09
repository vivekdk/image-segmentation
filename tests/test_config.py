from segmentation.config import load_config


def test_load_config_reads_expected_defaults() -> None:
    config = load_config("configs/oxford_pet_unet.yaml")
    assert config.data.dataset_name == "oxford_pet"
    assert config.data.image_size == 256
    assert config.model.base_channels == 32
    assert config.optimizer.epochs == 25
