import os
import subprocess
import soundfile as sf
import pandas as pd
import museval
import numpy as np

# Define directories
test_dir = "test"
model_path = "open_unmix_guitar_final"
output_dir = f'output_{model_path}'
target_value = 'acoustic_guitar'  # or bass

# Ensure output directory exists
os.makedirs(output_dir, exist_ok=True)

# CSV file to save results
csv_file = f'metrics_results_for_{model_path}.csv'

# List to store metrics results
results = []

# Process each track
for track_id in sorted(os.listdir(test_dir), key=int):  # Sort numerically
    track_dir = os.path.join(test_dir, track_id)
    mix_path = os.path.join(track_dir, "mix.wav")
    target_path = os.path.join(track_dir, "acoustic_guitar.wav")

    # Skip if mix or target files are missing
    if not os.path.exists(mix_path) or not os.path.exists(target_path):
        print(f"Missing files in {track_dir}, skipping...")
        continue

    # Run Open-Unmix
    print(f"Processing: {mix_path}")
    track_output_dir = os.path.join(output_dir, track_id)
    os.makedirs(track_output_dir, exist_ok=True)
    guitar_path = os.path.join(track_output_dir, 'mix',f'{target_value}.wav')

    command = [
        "umx",
        "--model", model_path,
        "--targets", target_value,
        "--residual", "true",
        "--outdir", track_output_dir,
        mix_path
    ]
    subprocess.run(command, check=True)#, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # Check if the separated file exists
    if not os.path.exists(guitar_path):
        print(f"Separated file for {track_id} not found, skipping metrics calculation.")
        continue

    try:
        # Load audio files
        estimated_audio, _ = sf.read(guitar_path)
        reference_audio, _ = sf.read(target_path)
        estimated_audio = estimated_audio[:, 1:]
        estimated_audio = np.squeeze(estimated_audio)

        if np.all(estimated_audio[0] == 0):
            print(f"Estimated audio for track {track_id} is silent, skipping...")
            continue
        if np.all(reference_audio[0] == 0):
            print(f"Reference audio for track {track_id} is silent, skipping...")
            continue

        # Ensure shapes match
        min_len = min(estimated_audio.shape[0], reference_audio.shape[0])
        estimated_audio = estimated_audio[:min_len]
        reference_audio = reference_audio[:min_len]

        # Reshape to match museval input format (tracks should be [samples, channels])
        if len(estimated_audio.shape) == 1:  # Mono audio
            estimated_audio = estimated_audio[:, None]
        if len(reference_audio.shape) == 1:  # Mono audio
            reference_audio = reference_audio[:, None]

        # Calculate SDR, SAR using museval
        metrics = museval.evaluate(reference_audio.T, estimated_audio.T)  # Transpose for museval compatibility
        sdr = metrics[0][0][0].mean()  # First target, first channel
        sar = metrics[3][0][0].mean()  # First target, first channel
        # Round metrics
        sdr, sar = round(sdr, 5), round(sar, 5)
        guitar_path_csv = os.path.join(track_id, f'{target_value}.wav')
        results.append([track_id, guitar_path_csv, sdr, sar])
        print(f"Metrics for track {track_id}: SDR={sdr}, SAR={sar}")
    except Exception as e:
        print(f"Error calculating metrics for track {track_id}: {e}")

# Save results to a CSV file
df = pd.DataFrame(results, columns=["Track ID", "Output File", "SDR", "SAR"])
df.to_csv(csv_file, index=False)
print(f"Metrics saved to {csv_file}")