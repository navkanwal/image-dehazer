# Single Image Dehazing with a Convolutional Autoencoder

A PyTorch notebook for reconstructing clear images from hazy inputs using supervised learning on paired Dense-Haze images. It includes preprocessing, model training, reconstruction plots, PSNR evaluation, and custom-image inference.

## Dataset and preprocessing

The original experiment loaded **55 hazy/clear image pairs** from [Dense-Haze, NTIRE 2019](https://data.vision.ee.ethz.ch/cvl/ntire19/dense-haze/). Images are resized to **256 × 256** and scaled to **[0, 1]**. The cleaned notebook matches pairs by filename and uses RGB consistently.

Each color channel is processed separately by the same single-channel network. Thus, 55 image pairs yield 165 channel pairs; these are not 165 independent images.

Download the dataset separately and follow its terms of use. Dataset images and trained weights are not included.

## Model

| Component | Structure |
| --- | --- |
| Input | One color channel, 1 × 256 × 256 |
| Encoder | Seven 3 × 3 convolutions; channels 1 → 32 → 32 → 64 → 64 → 128 → 128 → 256 |
| Downsampling | Two 2 × 2 max-pooling layers |
| Encoded representation | 256 × 64 × 64 |
| Decoder | Seven 3 × 3 transposed convolutions; channels 256 → 128 → 128 → 64 → 64 → 32 → 32 → 1 |
| Upsampling | Two stride-2 transposed convolutions |
| Activations | ReLU; batch normalization after intermediate layers |
| Output | One reconstructed channel, recombined into an RGB image |

There are no skip connections or attention layers. Training uses **MSE loss**, **Adam**, a learning rate of **0.0005**, a channel batch size of **1**, and **1,000 configured epochs**.

## Recorded result

| Metric | Original notebook output | Evaluation data |
| --- | ---: | --- |
| Mean PSNR | **25.4685 dB** | The same 55 image pairs used for training |

This is a training-set reconstruction measurement, not held-out test performance. The original notebook does not record a computed SSIM score or an inference-time measurement. Its saved output reports PSNR, but the training progress output does not establish completion of all 1,000 configured epochs.

The cleaned notebook preserves the convolutional architecture and configured hyperparameters while correcting inference mode, image preprocessing, and loss logging. It has not been retrained with the dataset, so the historical PSNR is not claimed as a result of the cleaned version.

By default, `TRAIN_FRACTION = 1.0` preserves the original all-data training protocol. Set it to `0.8` before training to reserve complete image pairs for evaluation. This optional random split is not the official NTIRE benchmark split.

## Run locally

Use Python 3.10 or newer. Install PyTorch with the appropriate CPU or CUDA build for your machine, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
python -m jupyter notebook image_dehazing_autoencoder.ipynb
```

Arrange the dataset as follows, or change `DATA_DIR`, `GT_DIR`, and `HAZY_DIR` in the configuration cell:

```text
data/Dense_Haze_NTIRE19/GT/01_GT.png
data/Dense_Haze_NTIRE19/hazy/01_hazy.png
```

Run the cells in order. Training uses the GPU when available and otherwise runs on CPU. Full training is computationally expensive; reduce `EPOCHS` for an initial check.

Generated files go to `outputs/`:

- `dehazing_autoencoder.pt`: model weights, configuration, loss history, and split IDs.
- `psnr_by_image.csv`: input and reconstruction PSNR for each evaluated image.
- `dehazed_custom.png`: output for an optional custom image.

To reuse a checkpoint created by this notebook, set `RUN_TRAINING = False` and keep the saved dataset/split configuration. To process another image, set `CUSTOM_IMAGE_PATH = Path("data/my_hazy_image.jpg")`.

PSNR uses a peak intensity of 1 and is averaged over complete RGB images. Raw model outputs are used for PSNR; clipping to [0, 1] is applied only when displaying or saving images.

## Tools

Python, PyTorch, NumPy, Pillow, Matplotlib, tqdm, and Jupyter Notebook.
