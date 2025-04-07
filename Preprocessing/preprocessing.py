import os
import shutil
import yaml
import soundfile as sf
import numpy as np
import librosa
import random
import time 

start_time = time.time()

np.random.seed(42)


def filter_acoustic_guitar_and_bass_tracks(dataset_path, output_path):
    """
    Filter the tracks containing both acoustic guitar and bass from the dataset
    and save them to a new directory.

    Args:
        dataset_path (str): Path to the main dataset directory.
        output_path (str): Path to the directory where the filtered tracks will be saved.

    Returns:
        int: Number of tracks containing both acoustic guitar and bass.
    """
    # Counter for tracks meeting the criteria
    valid_tracks_count = 0

    # Ensure the output directory exists
    os.makedirs(output_path, exist_ok=True)

    # Iterate over each folder in the dataset in a deterministic order
    for track_folder in sorted(os.listdir(dataset_path)):  # Sorted for determinism
        track_path = os.path.join(dataset_path, track_folder)
        
        # Check if it's a directory and contains a metadata.yaml file
        if os.path.isdir(track_path):
            metadata_file = os.path.join(track_path, "metadata.yaml")
            
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    try:
                        # Load the data from metadata.yaml
                        metadata = yaml.safe_load(f)
                        stems = metadata.get('stems', {})
                        
                        # Check for acoustic guitar and bass presence
                        contains_acoustic_guitar = any(
                            stem_data.get('inst_class', "") == "Guitar" and "Acoustic" in stem_data.get('midi_program_name', "")
                            for stem_data in stems.values()
                        )
                        contains_bass = any(
                            stem_data.get('inst_class', "") == "Bass"
                            for stem_data in stems.values()
                        )
                        
                        # If both conditions are met, copy the track
                        if contains_acoustic_guitar and contains_bass:
                            valid_tracks_count += 1
                            
                            # Copy the entire track folder to the output directory
                            output_track_path = os.path.join(output_path, track_folder)
                            shutil.copytree(track_path, output_track_path)
                            print(f"Track copied: {track_folder}")

                    except Exception as e:
                        print(f"Error loading the file {metadata_file}: {e}")
    
    print(f"Total number of tracks containing both acoustic guitar and bass: {valid_tracks_count}")
    return valid_tracks_count


def clean_acoustic_guitar_dataset(dataset_path):
    """
    Cleans the dataset by removing stems that are not related to acoustic guitars and combines the acoustic guitar stems into a single audio file.

    Args:
        dataset_path (str): Path to the main directory of the dataset.
    """
    for track_folder in sorted(os.listdir(dataset_path)):  # Sort for deterministic order
        track_path = os.path.join(dataset_path, track_folder)
        
        metadata_file = os.path.join(track_path, "metadata.yaml")
        stems_dir = os.path.join(track_path, "stems")
        
        if os.path.exists(metadata_file) and os.path.isdir(stems_dir):
            with open(metadata_file, 'r') as f:
                try:
                    metadata = yaml.safe_load(f)
                    stems = metadata.get('stems', {})
                    
                    acoustic_stems = []
                    sample_rate = None
                    
                    # Iterate over the sorted stems to ensure deterministic order
                    for stem_key in sorted(stems.keys()):  
                        stem_data = stems[stem_key]
                        inst_class = stem_data.get('inst_class', "")
                        midi_program_name = stem_data.get('midi_program_name', "")
                        
                        if inst_class == "Guitar" and "Acoustic" in midi_program_name:
                            # Add the path of the stem
                            stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                            if os.path.exists(stem_audio_file):
                                audio, sr = sf.read(stem_audio_file)
                                if sample_rate is None:
                                    sample_rate = sr
                                elif sample_rate != sr:
                                    raise ValueError(f"Mismatch in sample rate: {sample_rate} vs {sr}")
                                acoustic_stems.append(audio)
                        else:
                            # Remove the non-acoustic guitar stem
                            stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                            if os.path.exists(stem_audio_file):
                                os.remove(stem_audio_file)
                                print(f"Deleted: {stem_audio_file}")
                    
                    # Combine the acoustic guitar stems
                    if acoustic_stems:
                        combined_signal = np.sum(np.stack(acoustic_stems), axis=0)  # Stack ensures consistent summation
                        combined_audio_file = os.path.join(stems_dir, "acoustic_guitar.wav")
                        sf.write(combined_audio_file, combined_signal, sample_rate)
                        print(f"Created combined file: {combined_audio_file}")
                    
                        # Delete the original individual acoustic guitar files
                        for stem_key in sorted(stems.keys()):  # Sorted ensures deterministic deletion
                            stem_data = stems[stem_key]
                            inst_class = stem_data.get('inst_class', "")
                            midi_program_name = stem_data.get('midi_program_name', "")
                            if inst_class == "Guitar" and "Acoustic" in midi_program_name:
                                stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                                if os.path.exists(stem_audio_file):
                                    os.remove(stem_audio_file)
                                    print(f"Deleted original stem: {stem_audio_file}")

                except Exception as e:
                    print(f"Error loading the file {metadata_file}: {e}")


def clean_acoustic_guitar_and_bass_dataset(dataset_path):
    """
    Cleans the dataset by removing stems not related to acoustic guitars or bass,
    and combines the acoustic guitar stems into a single audio file.

    Args:
        dataset_path (str): Path to the main directory of the dataset.
    """
    for track_folder in sorted(os.listdir(dataset_path)):  # Sort for deterministic order
        track_path = os.path.join(dataset_path, track_folder)
        
        metadata_file = os.path.join(track_path, "metadata.yaml")
        stems_dir = os.path.join(track_path, "stems")
        
        if os.path.exists(metadata_file) and os.path.isdir(stems_dir):
            with open(metadata_file, 'r') as f:
                try:
                    metadata = yaml.safe_load(f)
                    stems = metadata.get('stems', {})
                    
                    acoustic_stems = []
                    sample_rate = None
                    
                    # Iterate over the sorted stems to ensure deterministic order
                    for stem_key in sorted(stems.keys()):
                        stem_data = stems[stem_key]
                        inst_class = stem_data.get('inst_class', "")
                        midi_program_name = stem_data.get('midi_program_name', "")
                        
                        if inst_class == "Guitar" and "Acoustic" in midi_program_name:
                            # Add the path of the stem
                            stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                            if os.path.exists(stem_audio_file):
                                audio, sr = sf.read(stem_audio_file)
                                if sample_rate is None:
                                    sample_rate = sr
                                elif sample_rate != sr:
                                    raise ValueError(f"Mismatch in sample rate: {sample_rate} vs {sr}")
                                acoustic_stems.append(audio)
                        elif inst_class == "Bass":
                            # Rename bass stem to bass.wav and keep it
                            bass_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                            if os.path.exists(bass_audio_file):
                                new_bass_audio_file = os.path.join(stems_dir, "bass.wav")
                                os.rename(bass_audio_file, new_bass_audio_file)
                                print(f"Renamed bass stem to: {new_bass_audio_file}")
                        else:
                            # Remove non-acoustic guitar and non-bass stems
                            stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                            if os.path.exists(stem_audio_file):
                                os.remove(stem_audio_file)
                                print(f"Deleted: {stem_audio_file}")
                    
                    # Combine the acoustic guitar stems
                    if acoustic_stems:
                        combined_signal = np.sum(np.stack(acoustic_stems), axis=0)  # Stack ensures consistent summation
                        combined_audio_file = os.path.join(stems_dir, "acoustic_guitar.wav")
                        sf.write(combined_audio_file, combined_signal, sample_rate)
                        print(f"Created combined file: {combined_audio_file}")
                    
                        # Delete the original individual acoustic guitar files
                        for stem_key in sorted(stems.keys()):  # Sorted ensures deterministic deletion
                            stem_data = stems[stem_key]
                            inst_class = stem_data.get('inst_class', "")
                            midi_program_name = stem_data.get('midi_program_name', "")
                            if inst_class == "Guitar" and "Acoustic" in midi_program_name:
                                stem_audio_file = os.path.join(stems_dir, f"{stem_key}.wav")
                                if os.path.exists(stem_audio_file):
                                    os.remove(stem_audio_file)
                                    print(f"Deleted original stem: {stem_audio_file}")

                except Exception as e:
                    print(f"Error loading the file {metadata_file}: {e}")


def clean_dataset(dataset_path):
    """
    Removes all unnecessary files and folders, keeping only mix.wav and the acoustic guitar stem
    in the main track folder.
    
    Args:
        dataset_path (str): Path to the directory containing the tracks.
    """
    # Iterate over each track in the dataset
    tracks = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    for track in tracks:
        track_path = os.path.join(dataset_path, track)
        mix_file = os.path.join(track_path, "mix.wav")
        stems_dir = os.path.join(track_path, "stems")
        
        # Check if mix.wav exists
        if not os.path.exists(mix_file):
            print(f"WARNING: mix.wav not found for {track}. Skipping...")
            continue
        
        # Check if there is a stem in the stems folder
        if not os.path.exists(stems_dir):
            print(f"WARNING: Stems folder not found for {track}. Skipping...")
            continue
        
        stem_files = [
            os.path.join(stems_dir, f) for f in os.listdir(stems_dir) if f.endswith(".wav")
        ]
        
        if len(stem_files) != 2:
            print(f"WARNING: Not exactly two stem found for {track}. Skipping...")
            continue
        
        acoustic_stem = stem_files[0]
        bass_stem = stem_files[1]
        acoustic_filename = os.path.basename(acoustic_stem)
        bass_filename = os.path.basename(bass_stem)
        new_acoustic_path = os.path.join(track_path, acoustic_filename)
        new_bass_path = os.path.join(track_path, bass_filename)
        
        # Move the stem to the main folder
        shutil.move(acoustic_stem, new_acoustic_path)
        shutil.move(bass_stem, new_bass_path)
        
        # Delete the stems folder
        shutil.rmtree(stems_dir)
        
        # Delete other unnecessary files and folders
        for root, dirs, files in os.walk(track_path, topdown=False):
            # Delete unnecessary files
            for file in files:
                file_path = os.path.join(root, file)
                if file_path not in {mix_file, new_acoustic_path, new_bass_path}:
                    os.remove(file_path)
            
            # Delete unnecessary folders
            for dir_ in dirs:
                dir_path = os.path.join(root, dir_)
                shutil.rmtree(dir_path)

        print(f"Track {track} cleaned. Only mix.wav {acoustic_filename} and {bass_filename} retained.")



# Function to add Gaussian noise
def add_gaussian_noise(audio, noise_level=0.005):
    noise = np.random.normal(0, noise_level, audio.shape)
    return audio + noise

# Function to apply pitch shift
def apply_pitch_shift(audio, sample_rate, pitch_factor=2):
    return librosa.effects.pitch_shift(audio, sr=sample_rate, n_steps=pitch_factor)

# Function to apply time stretch
def apply_time_stretch(audio, rate=1.2):
    return librosa.effects.time_stretch(audio, rate=rate)

# Data augmentation function
def augment_audio_variants(audio, sample_rate):
    augmented = {}

    # Normal
    augmented["normal"] = audio

    # Gaussian noise
    augmented["gaussian_noise"] = add_gaussian_noise(audio)

    # Pitch shift
    augmented["pitch_shift"] = apply_pitch_shift(audio, sample_rate)

    # Time stretch
    augmented["time_stretch"] = apply_time_stretch(audio)

    return augmented


def split_and_augment_tracks(dataset_path, output_path, segment_duration=20, overlap=10, sample_rate=44100, silence_threshold=0.15):
    """
    Splits mix and stem into 20-second segments with a 10-second overlap, saves the segments into separate folders,
    applies data augmentation (multiple variants), and removes segments where the stem contains less than 15% non-zero values.
    
    Args:
        dataset_path (str): Path to the dataset directory.
        output_path (str): Path to the output directory for the segments.
        segment_duration (int): Duration of each segment in seconds (default: 20).
        overlap (int): Overlap between segments in seconds (default: 10).
        sample_rate (int): Sampling rate in Hz (default: 44100).
        silence_threshold (float): Minimum percentage of non-zero values required to keep the segment (default: 0.15).
    """
    segment_samples = segment_duration * sample_rate
    overlap_samples = overlap * sample_rate
    step_size = segment_samples - overlap_samples

    # Iterate over each track in the dataset folder
    tracks = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    
    for track in tracks:
        track_path = os.path.join(dataset_path, track)
        mix_path = os.path.join(track_path, "mix.wav")
        acoustic_path = os.path.join(track_path, "acoustic_guitar.wav")
        bass_path = os.path.join(track_path, "bass.wav")

        if not os.path.exists(mix_path) or not os.path.exists(acoustic_path) or not os.path.exists(bass_path):
            print(f"WARNING: Missing files for track {track}. Skipping...")
            continue

        # Load the audio files (mono=True for mono audio)
        mix, _ = librosa.load(mix_path, sr=sample_rate, mono=True)
        acoustic, _ = librosa.load(acoustic_path, sr=sample_rate, mono=True)
        bass, _ = librosa.load(bass_path, sr=sample_rate, mono=True)

        # Split the audio files into segments
        num_segments = (len(mix) - overlap_samples) // step_size
        for i in range(num_segments):
            start = i * step_size
            end = start + segment_samples

            # If the segment exceeds the length, truncate
            if end > len(mix):
                break

            mix_segment = mix[start:end]  # Extract the mono channel
            acoustic_segment = acoustic[start:end]  # Extract the mono channel
            bass_segment = bass[start:end]

            # Calculate the percentage of non-zero values in the stem
            non_zero_percentage = np.count_nonzero(acoustic_segment) / len(acoustic_segment)
            print(f"Segment {i} of track {track}: Non-zero value percentage = {non_zero_percentage:.2%}")

            # If the stem is too quiet, do not save the segment
            if non_zero_percentage < silence_threshold:
                print(f"Segment {i} of track {track} discarded due to silence in the stem.")
                continue

            # Create a new folder for each variant
            mix_augmented_variants = augment_audio_variants(mix_segment, sample_rate)
            acoustic_augmented_variants = augment_audio_variants(acoustic_segment, sample_rate)
            bass_augmented_variants = augment_audio_variants(bass_segment, sample_rate)

            for variant_name, mix_variant in mix_augmented_variants.items():
                # Create folders for each variant
                segment_folder = os.path.join(output_path, f"{track}-{i}-{variant_name}")
                os.makedirs(segment_folder, exist_ok=True)

                # Save the mix variant
                mix_variant_path = os.path.join(segment_folder, f"mix.wav")
                sf.write(mix_variant_path, mix_variant, sample_rate)

                # Logic for the stem:
                # - If the variant is 'gaussian_noise', keep the original stem
                # - For other variants, modify the stem
                if variant_name == "gaussian_noise":
                    acoustic_variant = acoustic_segment 
                    bass_variant = bass_segment # The stem remains unaltered
                else:
                    acoustic_variant = acoustic_augmented_variants[variant_name]
                    bass_variant = bass_augmented_variants[variant_name] # The stem is modified

                # Save the stem variant
                acoustic_variant_path = os.path.join(segment_folder, f"acoustic_guitar.wav")
                bass_variant_path = os.path.join(segment_folder, f"bass.wav")
                sf.write(acoustic_variant_path, acoustic_variant, sample_rate)
                sf.write(bass_variant_path, bass_variant, sample_rate)

                print(f"Segment {i} of track {track}, variant {variant_name} saved in {segment_folder}.")



def reorganize_for_openunmix(output_path, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Reorganizes audio files into train, validation, and test folders for input to OpenUnmix.

    Args:
        output_path (str): Path to the directory containing tracks with mix.wav and target files.
        train_ratio (float): Percentage of files to allocate to the 'train' folder (default: 0.8).
        val_ratio (float): Percentage of files to allocate to the 'validation' folder (default: 0.1).
        test_ratio (float): Percentage of files to allocate to the 'test' folder (default: 0.1).
        seed (int): Seed for random number generation to ensure reproducibility (default: 42).
    """

    # Validate ratios
    if not (0 <= train_ratio <= 1 and 0 <= val_ratio <= 1 and 0 <= test_ratio <= 1):
        raise ValueError("Ratios must be between 0 and 1")
    if not round(train_ratio + val_ratio + test_ratio, 5) == 1.0:
        raise ValueError("Ratios do not sum to 1")

    # Set seed for reproducibility
    random.seed(seed)

    # Create folders
    train_dir = os.path.join(output_path, "train")
    val_dir = os.path.join(output_path, "valid")
    test_dir = os.path.join(output_path, "test")

    os.makedirs(os.path.join(train_dir, "mix"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "acoustic"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "bass"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "mix"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "acoustic"), exist_ok=True)
    os.makedirs(os.path.join(val_dir, "bass"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "mix"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "acoustic"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "bass"), exist_ok=True)

    # Get sorted list of tracks
    tracks = sorted([d for d in os.listdir(output_path) if os.path.isdir(os.path.join(output_path, d))])

    for track in tracks:
        track_path = os.path.join(output_path, track)
        mix_path = os.path.join(track_path, "mix.wav")
        acoustic_path = os.path.join(track_path, "acoustic_guitar.wav")
        bass_path = os.path.join(track_path, "bass.wav")

        # Check file existence
        if os.path.exists(mix_path) and os.path.exists(acoustic_path) and os.path.exists(bass_path):
            mix_file = f"{track}_mix.wav"
            acoustic_file = f"{track}_acoustic_guitar.wav"
            bass_file = f"{track}_bass.wav"

            # Determine destination folder
            rand_value = random.random()
            if rand_value < train_ratio:
                dest_mix = os.path.join(train_dir, "mix", mix_file)
                dest_acoustic = os.path.join(train_dir, "acoustic", acoustic_file)
                dest_bass = os.path.join(train_dir, "bass", bass_file)
            elif rand_value < train_ratio + val_ratio:
                dest_mix = os.path.join(val_dir, "mix", mix_file)
                dest_acoustic = os.path.join(val_dir, "acoustic", acoustic_file)
                dest_bass = os.path.join(val_dir, "bass", bass_file)
            else:
                dest_mix = os.path.join(test_dir, "mix", mix_file)
                dest_acoustic = os.path.join(test_dir, "acoustic", acoustic_file)
                dest_bass = os.path.join(test_dir, "bass", bass_file)

            # Move files
            shutil.move(mix_path, dest_mix)
            shutil.move(acoustic_path, dest_acoustic)
            shutil.move(bass_path, dest_bass)
            
            print(f"Moved {mix_file}, {acoustic_file}, {bass_file} to {os.path.dirname(dest_mix)}")
        else:
            print(f"WARNING: Missing files in track {track}!")
    
    os.makedirs("finalDataset", exist_ok=True)
    print("Folder finalDataset created successfully!")
    shutil.move(train_dir, "finalDataset")
    shutil.move(val_dir, "finalDataset")
    shutil.move(test_dir, "finalDataset")
    
    print(f"Reorganization complete. Folders created:\n - Train: {train_dir}\n - Validation: {val_dir}\n - Test: {test_dir}")


def reorganize_dataset(root_dir):
    """
    Reorganizes the dataset into a structure compatible with OpenUnmix.

    Args:
        root_dir (str): Path to the directory containing train, validation, and test splits.
    """
    for split in ["train", "valid", "test"]:
        split_path = os.path.join(root_dir, split)
        mix_path = os.path.join(split_path, "mix")
        acoustic_path = os.path.join(split_path, "acoustic")
        bass_path = os.path.join(split_path, "bass")

        if not os.path.exists(mix_path) or not os.path.exists(acoustic_path):
            print(f"Path not found for {split}. Skipping...")
            continue

        # Ensure files are processed in a deterministic order
        mix_files = sorted([f for f in os.listdir(mix_path) if f.endswith(".wav")])
        acoustic_files = sorted([f for f in os.listdir(acoustic_path) if f.endswith(".wav")])
        bass_files = sorted([f for f in os.listdir(bass_path) if f.endswith(".wav")])

        for mix_file, acoustic_file, bass_file in zip(mix_files, acoustic_files, bass_files):
            track_name = os.path.splitext(mix_file)[0].replace("_mix", "")
            track_folder = os.path.join(split_path, track_name)
            os.makedirs(track_folder, exist_ok=True)

            # Move the files into the track directory
            shutil.move(os.path.join(mix_path, mix_file), os.path.join(track_folder, "mix.wav"))
            shutil.move(os.path.join(acoustic_path, acoustic_file), os.path.join(track_folder, "acoustic_guitar.wav"))
            shutil.move(os.path.join(bass_path, bass_file), os.path.join(track_folder, "bass.wav"))

        # Remove the old directories for mix and targets
        shutil.rmtree(mix_path)
        shutil.rmtree(acoustic_path)
        shutil.rmtree(bass_path)
        

def rename_folders(main_directory):
    # Get the list of folders in the main directory
    folders = [d for d in os.listdir(main_directory) if os.path.isdir(os.path.join(main_directory, d))]
    
    # Optionally sort the folders if you want a specific order
    folders.sort()

    # Rename the folders with sequential numbers
    for i, folder in enumerate(folders, start=1):
        old_name = os.path.join(main_directory, folder)
        new_name = os.path.join(main_directory, str(i))
        os.rename(old_name, new_name)
        print(f"Renamed: {folder} -> {i}")


        

# Path to the original dataset
dataset_path = "babyslakh_16k"

# Path to the intermediate output directory
intermediate_path = "intermediateDataset"
os.makedirs(intermediate_path, exist_ok=True)
print(f"Folder '{intermediate_path}' created successfully!")

# Path to the final preprocessed dataset directory
output_path = "preprocessedDataset"
os.makedirs(output_path, exist_ok=True)
print(f"Folder '{output_path}' created successfully!")

# Execute the function to filter acoustic guitar tracks
filter_acoustic_guitar_and_bass_tracks(dataset_path, intermediate_path)

# Execute the cleaning function
#clean_acoustic_guitar_dataset(intermediate_path)
clean_acoustic_guitar_and_bass_dataset(intermediate_path)

folder_to_remove = os.path.join(intermediate_path, "17") # They forgot to add the bass stem
if os.path.exists(folder_to_remove):
    shutil.rmtree(folder_to_remove)
    print(f"Folder '{folder_to_remove}' deleted successfully!")
else:
    print(f"Folder '{folder_to_remove}' not found.")

clean_dataset(intermediate_path)

# Split, augment tracks, and save the output
split_and_augment_tracks(intermediate_path, output_path)

# Reorganize files for OpenUnmix
reorganize_for_openunmix(output_path)

if os.path.exists(output_path):
    shutil.rmtree(output_path)
    print(f"Folder '{output_path}' deleted successfully!")
else:
    print(f"Folder '{output_path}' not found.")



reorganize_dataset("finalDataset")

# Example usage
directory1 = 'finalDataset/train'
directory2 = 'finalDataset/valid'
directory3 = 'finalDataset/test'


rename_folders(directory1)
rename_folders(directory2)
rename_folders(directory3)

if os.path.exists('intermediateDataset'):
    shutil.rmtree('intermediateDataset')
    print("Folder intermediateDataset deleted successfully!")
else:
    print("Folder intermediateDataset not found.")
    
if os.path.exists('preprocessedDataset'):
    shutil.rmtree('preprocessedDataset')
    print("Folder preprocessedDataset deleted successfully!")
else:
    print("Folder preprocessedDataset not found.")
    
end_time = time.time()
execution_time = end_time - start_time

print(f"Time of execution: {execution_time} seconds")
