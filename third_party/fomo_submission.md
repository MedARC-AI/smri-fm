## Table of Contents
${toc}

## TL;DR

Please note that

* All submissions must be Apptainer (Singularity) containers -- i.e., we only accept `.sif` files.
* For each task, the input will be given as command-line arguments to your script (e.g., for task 3: `python predict.py --t1 path/to/t1.nii.gz --output path/to/write/output.txt`)
* We have made example templates and a local validator here [container-validator](https://github.com/fomo26/container-validator).
Details are available below.

## Overview

Participants are required to submit the models as containerized solutions. This containerization approach ensures that your model can run in the evaluation environment exactly as it does on your own system, with all dependencies properly packaged. The container creates a standardized, isolated environment where your model can operate regardless of the host system configuration.

## Prerequisites

Before beginning the container validation process, ensure you have installed all necessary tools and dependencies.

### Install Apptainer

You need to install Apptainer (formerly Singularity) to build and run your container. Apptainer primarily supports Linux environments (Ubuntu, Debian, etc). If using MacOS or Windows, you'll need to use virtualization tools (Docker, Virtual Machines, or WSL2).

Installation instructions by platform:
- [Install in Linux (Ubuntu, Debian, Fedora, ...)](https://apptainer.org/docs/admin/main/installation.html#installation-on-linux)
    - For most use cases: [installing unprivileged from pre-built binaries](https://apptainer.org/docs/admin/main/installation.html#install-unprivileged-from-pre-built-binaries) should suffice
- [Install in MacOS](https://apptainer.org/docs/admin/main/installation.html#mac)
- [Install in Windows](https://apptainer.org/docs/admin/main/installation.html#windows)

Once you have installed it, verify your Apptainer installation with:

```bash
apptainer --version
```

## 1. Task Specific Requirements

To participate in the FOMO26 Challenge, you must prepare a container that meets specific requirements for each downstream task. Each task has its own input and output specifications. In order to ensure that your evaluation is successful, you must follow the guidelines below.

You must prepare the following files for your submission (all these files are **mandatory**). Your container **must** have the following internal structure:

```
/
├── app/              # Your application code
│   └── predict.py    # Main inference script (REQUIRED)
├── input/            # Mounted input directory (DO NOT include in container)
├── output/           # Mounted output directory (DO NOT include in container)
└── ...               # Other system files
```

Important notes:
- Your predict.py file must be located at `/app/predict.py`
- The input and output directories are mounted at runtime and should not be included in your container

### Task 1: Infarct Detection

This task requires you to classify the presence of an infarct(s) in brain MRI images (binary classification).

**predict.py**
This script will be executed inside your container to perform the classification. It should take the required input images, process them, and output a single probability value indicating the presence of an infarct.

_Input_: T2 FLAIR, DWI (b-value 1000), ADC, and either T2* or SWI images. You will always receive all four images, but you can use only the ones you need for your model.
_Output_: A text file (.txt) with the probability that an infarct is present. Single probability value (e.g, 0.750).

Your `predict.py` script should handle the following command-line arguments:
- `--flair`: Path to T2 FLAIR image
- `--adc`: Path to ADC image
- `--dwi`: Path to DWI b1000 image
- `--t2s`: Path to T2* image (optional, can be replaced with SWI)
- `--swi`: Path to SWI image (optional, can be replaced with T2*)
- `--output`: Path to save the output .txt file with probability

**Example usage**:
```bash
python predict.py \
  --flair /path/to/flair.nii.gz \
  --adc /path/to/adc.nii.gz \
  --dwi /path/to/dwi.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --swi /path/to/swi.nii.gz \
  --output /path/to/output.txt
```
**Apptainer.def**
This file defines how your container is built and what dependencies it includes. It specifies the base image, environment variables, files to include, and the command to run when the container starts. Remember to include the necessary dependencies for your model, such as PyTorch, NumPy, and any other libraries you use in `predict.py`. Ensure that the script is executable and that it correctly handles the input and output paths specified in the command-line arguments.

[Generic example Apptainer.def for Task 1](https://github.com/fomo26/container-validator/tree/main/templates/task1/Apptainer.def)
[Generic example predict.py for Task 1](https://github.com/fomo26/container-validator/tree/main/templates/task1/predict.py)

### Task 2: Meningioma Segmentation

This task requires you to segment meningiomas in brain MRI images. The output should be a binary mask indicating the presence of meningioma in the images.

**predict.py**
This script will be executed inside your container to perform the segmentation. It should take the required input images, process them, and output a binary segmentation mask.

_Input_: T2 FLAIR, DWI (b-value 1000), and either T2* or SWI images.
_Output_: A NIfTI file (.nii.gz) containing the binary segmentation mask of the meningioma (label 0=background, label 1=meningioma). It should have the same dimensions and affine as the input images.

Your `predict.py` script should handle the following command-line arguments:
- `--flair`: Path to T2 FLAIR image
- `--dwi`: Path to DWI b1000 image
- `--t2s`: Path to T2* image (optional, can be replaced with SWI)
- `--swi`: Path to SWI image (optional, can be replaced with T2*)
- `--output`: Path to save segmentation NIfTI file

**Example usage**:
```bash
python predict.py \
  --flair /path/to/flair.nii.gz \
  --dwi /path/to/dwi.nii.gz \
  --t2s /path/to/t2s.nii.gz \
  --swi /path/to/swi.nii.gz \
  --output /path/to/output.nii.gz
```

**Apptainer.def**

[Generic example Apptainer.def for Task 2](https://github.com/fomo26/container-validator/tree/main/templates/task2/Apptainer.def)
[Generic example predict.py for Task 2](https://github.com/fomo26/container-validator/tree/main/templates/task2/predict.py)

### Task 3: Brain Age Estimation

**predict.py**

_Input_: T1-weighted images.
_Output_: A text file (.txt) containing the predicted brain age in years.

Required arguments for `predict.py`:
- `--t1`: Path to T1-weighted image
- `--output`: Path to save output .txt file with predicted brain age. Single value (e.g., 35)

**Example usage**:
```bash
python predict.py \
  --t1 /path/to/t1.nii.gz \
  --output /path/to/output.txt
```

**Apptainer.def**

[Generic example Apptainer.def for Task 3](https://github.com/fomo26/container-validator/tree/main/templates/task3/Apptainer.def)
[Generic example predict.py for Task 3](https://github.com/fomo26/container-validator/tree/main/templates/task3/predict.py)

### Task 4: Trigeminal Neuralgia Segmentation

**predict.py**
This script will be executed inside your container to perform the segmentation. It should take the required input image, process it, and output a segmentation.

_Input_: T2-weighted images.
_Output_: A NIfTI file (.nii.gz) containing the segmentation mask with the following labels: 0=background, 1=trigeminal neuralgia nerves, 2=vessels. It should have the same dimensions and affine as the input images.

Required arguments for `predict.py`:
- `--t2`: Path to T2-weighted image
- `--output`: Path to save the segmentation NIfTI file

**Example usage**:
```bash
python predict.py \
  --t2 /path/to/t2.nii.gz \
  --output /path/to/output.nii.gz
```

**Apptainer.def**

[Generic example Apptainer.def for Task 4](https://github.com/fomo26/container-validator/tree/main/templates/task4/Apptainer.def)
[Generic example predict.py for Task 4](https://github.com/fomo26/container-validator/tree/main/templates/task4/predict.py)

### Task 5: Polymicrogyria Classification

This task requires you to classify the presence of polymicrogyria in brain MRI images (binary classification).

**predict.py**
This script will be executed inside your container to perform the classification. It should take the required input image, process it, and output a single probability value indicating the presence of polymicrogyria.

_Input_: T1-weighted images.
_Output_: A text file (.txt) with the probability that polymicrogyria is present. Single probability value (e.g, 0.750).

Your `predict.py` script should handle the following command-line arguments:
- `--t1`: Path to T1-weighted image
- `--output`: Path to save output .txt file with probability

**Example usage**:
```bash
python predict.py \
  --t1 /path/to/t1.nii.gz \
  --output /path/to/output.txt
```
**Apptainer.def**

[Generic example Apptainer.def for Task 5](https://github.com/fomo26/container-validator/tree/main/templates/task5/Apptainer.def)
[Generic example predict.py for Task 5](https://github.com/fomo26/container-validator/tree/main/templates/task5/predict.py)

### Tasks 6 and 7: Linear Probing and Bias & Fairness

This task requires you to provide an embedding for an arbitrary MR image.

**predict.py**
This script will be executed inside your container to perform the linear probing and bias & fairness evaluation. It should take the required input image, process it, and output a 1D embedding.

_Input_: MR images.
_Output_: A NumPy file (.npy) with a fixed-length 1D embedding (same dimensionality for every input image).

Your `predict.py` script should handle the following command-line arguments:
- `--input`: Path to an MR image
- `--output`: Path to save the output .npy file with embedding

**Example usage**:
```bash
python predict.py \
  --input /path/to/scan.nii.gz \
  --output /path/to/output.npy
```
**Apptainer.def**

[Generic example Apptainer.def for Tasks 6 and 7](https://github.com/fomo26/container-validator/tree/main/templates/task6/Apptainer.def)
[Generic example predict.py for Tasks 6 and 7](https://github.com/fomo26/container-validator/tree/main/templates/task6/predict.py)

## 2. Build your container

Build your container using the Apptainer.def file you prepared in step 2:

```bash
apptainer build --fakeroot /path/to/save/your/container.sif path/to/Apptainer.def --arch amd64
```
This command creates a `.sif` container file that encapsulates your model and all its dependencies. The `--arch amd64` flag makes sure your container can run on x86 architectures.

## 3. Validate your container

To ensure your container satisfies the requirements of the evaluation pipeline, we have created a simple script which will evaluate the container.

The container validator tool is available here:

[Container Validator](https://github.com/fomo26/container-validator)

## 4. Submit your container

To submit your containers, please follow the [Submission Instructions](https://www.synapse.org/Synapse:syn72120565/wiki/640841).

## FAQ

**Q: Do I need to include training code in my submission?**
A: No, only the inference code is required. The evaluation will only run your `predict.py` script.

**Q: Can I use frameworks other than PyTorch?**
A: Yes, you can use any framework as long as it's included in your container. Make sure to specify all dependencies in your `Apptainer.def` file.

**Q: How do I handle GPU support?**
A: The validation script will test GPU support if available. Include GPU-compatible versions of your libraries if your model uses GPU acceleration.
1. Preparing your models for submission
Table of Contents
Table of Contents
TL;DR
Overview
Prerequisites
Install Apptainer
1. Task Specific Requirements
Task 1: Infarct Detection
Task 2: Meningioma Segmentation
Task 3: Brain Age Estimation
Task 4: Trigeminal Neuralgia Segmentation
Task 5: Polymicrogyria Classification
Tasks 6 and 7: Linear Probing and Bias & Fairness
2. Build your container
3. Validate your container
4. Submit your container
FAQ

TL;D
