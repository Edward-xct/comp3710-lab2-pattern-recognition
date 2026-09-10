# COMP3710 Lab Demonstration 2 - Pattern Recognition

This repository contains runnable PyTorch code for COMP3710 Lab Demonstration 2.

The lab is implemented as scripts rather than a single notebook so that the same code can be run locally for checks and on the Rangpur GPU cluster for full training.

## Implemented Parts

Part 1 - Discrete Fourier Transform:

- Square wave construction with Fourier harmonics.
- Naive NumPy DFT.
- NumPy FFT baseline.
- PyTorch tensor DFT by explicit matrix multiplication on CPU or CUDA GPU.
- Timing table and plots.

Part 2 - Eigenfaces:

- Loads the LFW people dataset.
- Computes PCA/eigenfaces using SVD on centered training data.
- Projects train/test data into face space.
- Trains a Random Forest classifier.
- Saves eigenface gallery, compactness plot, classification report, and metrics.

Part 3 - CNNs:

- LFW CNN classifier with two 3x3 convolution layers with 32 filters each.
- CIFAR10 ResNet-18 implemented from scratch for the DAWNBench-style task.
- CUDA AMP and channels-last options for fast cluster training.

Part 4 - Recognition:

- VAE for OASIS brain MRI slices, including reconstruction and 2D manifold visualisation.
- UNet for OASIS segmentation with 4-channel categorical output and DSC metrics for all labels.
- DCGAN for OASIS brain MRI generation with generated samples and loss curves.

## Environment

Create a Python environment, then install:

```bash
pip install -r requirements.txt
```

On this Mac, the Lab 1 environment can also be reused for local checks:

```bash
cd "/Users/xct/Documents/New project/comp3710_lab2_demo"
source "../comp3710_lab1_demo/.venv/bin/activate"
```

On Rangpur, use the Python/CUDA module recommended by the course, create a venv if needed, and install the same requirements.

## OASIS Data

The PDF link may be stale. The course staff posted an updated AARNet FileSender link in the Blackboard Demo 2 folder, and noted that the data is already available on Rangpur.

Do not commit tokenized FileSender links to a public GitHub repository.

The downloaded archive is:

```text
/Users/xct/Downloads/keras_png_slices_data.zip
```

It contains MRI slices and segmentation masks:

- `keras_png_slices_train`
- `keras_png_slices_validate`
- `keras_png_slices_test`
- `keras_png_slices_seg_train`
- `keras_png_slices_seg_validate`
- `keras_png_slices_seg_test`

Inspect it:

```bash
python scripts/prepare_oasis_data.py
```

Extract it into this project:

```bash
python scripts/prepare_oasis_data.py --extract
```

The scripts can also read directly from the zip for quick local checks:

```bash
python scripts/visualize_oasis_sample.py --data /Users/xct/Downloads/keras_png_slices_data.zip
```

Do not commit the dataset or trained checkpoints to GitHub.

## Validation and saved evidence

This check confirms that the code imports, the OASIS zip can be read, and all model forward passes work:

```bash
python scripts/smoke_tests.py
```

Every Part 1-4 entry script writes a timestamped console log under its output directory. Training tasks also save per-epoch CSV history and final JSON metrics containing loss, accuracy or Dice where applicable, device information, and elapsed time.

## Full Runs

Part 1:

```bash
python scripts/part1_dft.py
```

Part 2:

```bash
python scripts/part2_eigenfaces.py
```

Part 3.1:

```bash
python scripts/part3_lfw_cnn.py --epochs 20
```

Part 3.2 on a GPU:

```bash
python scripts/part3_cifar_resnet18.py --download --experiment-name final_amp_channels_last --epochs 80 --batch-size 512 --lr 0.4 --amp --channels-last
```

Part 4 on a GPU:

```bash
python scripts/part4_vae.py --data /home/groups/comp3710/OASIS
python scripts/part4_unet.py --data /home/groups/comp3710/OASIS
python scripts/part4_gan.py --data /home/groups/comp3710/OASIS
```

## Rangpur SLURM

Submit jobs from the project root:

```bash
sbatch slurm/part3_cifar_resnet18.sbatch
sbatch slurm/part3_cifar_ablation.sbatch
sbatch slurm/part3_cifar_ablation_detailed.sbatch
sbatch slurm/part4_vae.sbatch
sbatch slurm/part4_unet.sbatch
sbatch slurm/part4_gan.sbatch
```

Check the queue:

```bash
squeue -u "$USER"
```

Watch a log:

```bash
tail -f slurm-cifar-<jobid>.out
```

On Rangpur for COMP3710, these scripts use the `comp3710` account/partition, request one A100 GPU with `--gres=gpu:a100:1`, and request a 12-hour wall time. The scripts do not set `--mem` because this partition rejected explicit memory requests during testing. The venv supplies Python/PyTorch, so the scripts avoid loading a separate CUDA module.

For Part 3.2 demo evidence, prefer `slurm/part3_cifar_ablation_detailed.sbatch`. It runs seven real CIFAR10 experiments and preserves each run separately under `outputs/part3_cifar_resnet18/<experiment>`. It then writes `outputs/part3_cifar_resnet18/part3_cifar_experiment_summary.md` and `outputs/part3_cifar_resnet18/part3_cifar_experiment_summary.png`, which can be used to explain the improvement from baseline to the final AMP run. The shorter `slurm/part3_cifar_ablation.sbatch` is kept as a compact version.

## Outputs

All generated evidence is written under:

```text
outputs/
checkpoints/
```

For demo, show:

- Code for each part.
- Output plots and metrics.
- SLURM logs for GPU runs.
- Saved checkpoints for trained models.
- This README and `DEMO_STEPS_CN.md`.
- GitHub commit history.

See `docs/GITHUB_PUSH_CN.md` for the exact commands to create and push the GitHub repository.
