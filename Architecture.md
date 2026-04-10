# Architecture

## Overview

This project is a image segmentation pipeline built in PyTorch. Its purpose is to learn, from labeled examples, which pixels belong to the pet and which belong to the background.

At a high level, the system does three things:

1. Prepares images and masks into a format the model can learn from.
2. Trains a U-Net model to predict a mask for each image.
3. Uses the trained model to evaluate results or generate predictions on new images.

## Core Concepts

### What is segmentation?

Segmentation is different from image classification.

- Classification answers: "What is in this image?"
- Segmentation answers: "Which pixels belong to the object?"

So instead of predicting one label for the whole image, the model predicts a mask with one value per pixel.

### What is a mask?

A mask is a black-and-white image aligned with the input image:

- `1` means the pixel belongs to the object
- `0` means the pixel belongs to the background

In this project, the original Oxford-IIIT Pet annotations are simplified into a binary mask so the task stays focused and easy to train.

### What is U-Net?

U-Net is a model architecture commonly used for segmentation.

It works in two stages:

- The encoder compresses the image into deeper feature representations.
- The decoder expands those features back into a full-resolution mask.

The important idea is that U-Net keeps skip connections between the encoder and decoder, so the model can preserve fine spatial details while still learning higher-level features.

## System Building Blocks

The codebase is organized into a few clear parts:

- `config.py`: loads experiment settings such as image size, batch size, learning rate, and output paths.
- `dataset.py`: loads Oxford Pet data, converts labels into binary masks, and applies preprocessing.
- `model.py`: defines the U-Net.
- `engine.py`: contains the training and evaluation loops.
- `losses.py` and `metrics.py`: define how model quality is measured during training and evaluation.
- `visualization.py`: saves preview images, predicted masks, and overlays.
- `train.py`, `evaluate.py`, and `predict.py`: command-line entry points for the main workflows.

## High-Level Flow

```text
Config
  -> Dataset and preprocessing
  -> Model
  -> Training / Evaluation engine
  -> Saved outputs
```

## Training Flow

Training is the process of teaching the model from labeled examples.

High-level flow:

```text
YAML config
  -> load dataset
  -> preprocess images and masks
  -> build U-Net
  -> train over multiple epochs
  -> evaluate on validation data after each epoch
  -> save checkpoints and metrics
```

What happens during training:

- Images are resized and normalized.
- Masks are converted into binary targets.
- The model predicts a mask.
- A loss function compares the prediction with the true mask.
- The optimizer updates model weights to reduce that loss.

The training loop also tracks validation performance, so the project can keep the best checkpoint instead of only the latest one.

## Evaluation Flow

Evaluation answers: "How well does the trained model perform on unseen test data?"

High-level flow:

```text
Load config
  -> rebuild model
  -> load saved checkpoint
  -> run on test dataset
  -> report metrics
```

This uses the same preprocessing and model shape as training, which keeps results consistent.

## Prediction Flow

Prediction is the inference path for new images outside the dataset split workflow.

High-level flow:

```text
Load config and checkpoint
  -> preprocess input image
  -> run model
  -> convert output into binary mask
  -> save mask and overlay
```

The overlay output is mainly for human inspection, so it is easy to see where the model thinks the pet is in the image.

## How Quality is Measured

This project uses both a loss function and segmentation metrics.

### Loss

During training, loss tells the optimizer how wrong the prediction is.

This project combines:

- Binary cross-entropy: checks pixel-by-pixel correctness
- Dice loss: checks how well the predicted region overlaps the true region

This combination is useful because segmentation needs both local pixel accuracy and good object-level shape overlap.

### Metrics

The main reported metrics are:

- Dice: how much the predicted mask overlaps the true mask
- IoU: another overlap-based segmentation metric
- Pixel accuracy: how many pixels were predicted correctly overall

These metrics make it easier to judge model quality than loss alone.

## Outputs and Artifacts

Every training run creates an output folder containing the main artifacts:

- `config.yaml`: the resolved experiment settings
- `metrics.csv`: epoch-by-epoch training and validation results
- `last.pt`: the most recent checkpoint
- `best.pt`: the checkpoint with the best validation Dice
- visualization images: quick visual previews of predictions

This keeps each run reproducible and easy to inspect later.

## Design Intent

The project is designed as a clean learning-oriented baseline rather than a large production framework.

That means:

- the architecture is small and easy to follow
- the data pipeline is explicit
- training, evaluation, and prediction are separate but share the same core components
- the outputs are saved in a way that makes experiments easy to compare

## Summary

The system can be thought of as a straightforward segmentation pipeline:

```text
Images + masks
  -> preprocessing
  -> U-Net
  -> predicted mask
  -> metrics, checkpoints, and visual outputs
```

Its main strength is clarity: each part of the project maps cleanly to one stage of the segmentation workflow.
