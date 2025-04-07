# Import necessary libraries
import os
import zipfile
import shutil
import subprocess
from pathlib import Path

# Define paths for dataset and output directories
drive_dataset_zip = "path/to/your/finalDataset.zip"  # Update with the local path to your dataset zip file
dataset_path = Path("finalDataset")  # Directory for the extracted dataset
umx_repo_path = Path("open-unmix")  # Directory for the OpenUnmix repository
output_path = Path("output")  # Directory for model output
umx_train_script = umx_repo_path / "scripts" / "train.py"  # Path to the training script

# Step 1: Remove existing dataset directory if it exists
if dataset_path.exists():
    print("Removing existing dataset directory...")
    shutil.rmtree(dataset_path)

# Step 2: Extract the dataset if it doesn't already exist
if not dataset_path.exists():
    print("Extracting dataset...")
    with zipfile.ZipFile(drive_dataset_zip, 'r') as zip_ref:
        zip_ref.extractall(".")

# Step 3: Clone the OpenUnmix repository if it doesn't already exist
if not umx_repo_path.exists():
    print("Cloning OpenUnmix repository...")
    subprocess.run(["git", "clone", "https://github.com/sigsep/open-unmix-pytorch.git", str(umx_repo_path)])

# Step 4: Install the repository dependencies
print("Installing dependencies...")
os.chdir(umx_repo_path)
subprocess.run(["pip", "install", "-r", "requirements.txt"])

# Step 5: Fine-tune the model using the training script
print("Starting training...")
subprocess.run([
    "python", str(umx_train_script),
    "--target", "acoustic_guitar",
    "--dataset", "aligned",
    "--root", str(dataset_path),
    "--output", str(output_path),
    "--epochs", "60",
    "--batch-size", "32",
    "--seq-dur", "6",
    "--nb-workers", "4",
    "--nb-channels", "1",
    "--seed", "42",
    "--input-file", "mix.wav",  # Make sure this file exists in the dataset
    "--output-file", "acoustic_guitar.wav"
])

# Check installed version of OpenUnmix
subprocess.run(["pip", "show", "openunmix"])
