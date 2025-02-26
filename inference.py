import os
import sys
import time
import torchaudio
import numpy as np

from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from faster_whisper import WhisperModel, BatchedInferencePipeline
from faster_whisper.audio import decode_audio, pad_or_trim


class AudioDataset(Dataset):
    """
    A dataset that, given a list of audio files, decodes them and returns
    pre-processed features ready for the model.

    This moves the CPU-bound operations (decoding, feature extraction, padding)
    to the DataLoader workers, allowing better GPU utilization.
    """

    def __init__(self, audio_paths, model, max_len):
        self.audio_paths = audio_paths
        self.model = model
        self.sampling_rate = model.feature_extractor.sampling_rate
        self.max_len = max_len

    def __len__(self):
        return len(self.audio_paths)

    def __getitem__(self, idx):
        audio_path = self.audio_paths[idx]

        # Decode audio
        audio = decode_audio(input_file=audio_path, sampling_rate=self.sampling_rate)

        # Measure duration in seconds
        duration_sec = len(audio) / self.sampling_rate

        # Compute mel spectrogram
        features = self.model.feature_extractor(audio)[
            ..., :-1
        ]  # shape (n_mels, n_frames)

        # Pad or trim to max_len milliseconds
        features = pad_or_trim(features, self.max_len * 100)

        # Create metadata
        meta = {
            "start_time": 0.0,
            "end_time": duration_sec,
        }

        return audio_path, features, meta


def audio_collate_fn(batch):
    """
    Custom collate function for audio batches.
    Returns paths, stacked features, and metadata.
    """
    # Unzip the batch into paths, features, and metadata
    paths, features, metas = zip(*batch)

    # Stack features along batch dimension
    features_stacked = np.stack(features, axis=0)

    return list(paths), features_stacked, list(metas)


def create_dataloader(
    audio_paths, model, batch_size, max_len, num_workers, prefetch_factor
):
    dataset = AudioDataset(audio_paths=audio_paths, model=model, max_len=max_len)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
        collate_fn=audio_collate_fn,
    )
    return dataloader


def transcribe_with_dataloader(
    audio_dir,
    output_dir,
    model,
    batch_size,
    beam_size,
    confidence_threshold,
    max_len,
    num_workers,
    prefetch_factor,
):
    os.makedirs(output_dir, exist_ok=True)

    # Gather all supported audio files
    print(f"Scanning directory: {audio_dir}")
    audio_files = []
    for file in os.listdir(audio_dir):
        if file.endswith((".mp3", ".wav", ".m4a", ".flac")):
            audio_path = os.path.join(audio_dir, file)
            audio_files.append(audio_path)

    total_files = len(audio_files)
    if total_files == 0:
        print("No audio files found in the input directory.")
        return 0, 0.0

    print(f"Found {total_files} audio files to process")

    # Initialize your ctranslate2 pipeline
    batched_pipeline = BatchedInferencePipeline(model)

    # Create dataloader with the updated implementation
    dataloader = create_dataloader(
        audio_files,
        model,
        batch_size=batch_size,
        max_len=max_len,
        num_workers=num_workers,
        prefetch_factor=prefetch_factor,
    )

    total_files_processed = 0
    total_transcription_time = 0.0

    # Create a progress bar for the dataloader
    pbar = tqdm(dataloader, desc="Transcribing batches", unit="batch")

    for paths, features, metas in pbar:
        start_time = time.time()

        # Now we directly pass the preprocessed features to the model
        # instead of raw audio files
        results_list = batched_pipeline.forward_preprocessed(
            features=features,
            metas=metas,
            beam_size=beam_size,
            without_timestamps=True,
        )

        batch_time = time.time() - start_time
        total_transcription_time += batch_time

        # Now handle post-processing and saving
        batch_processed = 0
        for path, result in zip(paths, results_list):
            if result["confidence"] < confidence_threshold:
                continue

            base_filename = os.path.basename(path)
            base = os.path.splitext(base_filename)[0]
            out_path = os.path.join(output_dir, f"{base}.txt")
            with open(out_path, "w", encoding="utf-8") as f:
                f.write(result["text"])

            batch_processed += 1
            total_files_processed += 1

        # Update progress bar with additional information
        pbar.set_postfix(
            {
                "processed": total_files_processed,
                "batch_time": f"{batch_time:.2f}s",
                "batch_processed": batch_processed,
            }
        )

    print(
        f"Finished all. Processed {total_files_processed} files. "
        f"Total transcription time {total_transcription_time:.2f} seconds."
    )
    return total_files_processed, total_transcription_time


if __name__ == "__main__":
    """
    Main entry point.
    Usage:
        python batched_inference.py <start> <end>

    This script:
    1. Uses <start> and <end> to create paths for audio segments and their
       corresponding output directories.
    2. Loads a Whisper model (here, "whisper-d-ct2") from disk.
    3. Runs the `process_directory` function on the audio segment directory.
    4. Prints a summary of total time and average time per file.
    """
    if len(sys.argv) < 3:
        print("Usage: python batched_inference.py <start> <end>")
        sys.exit(1)

    start = int(sys.argv[1])
    end = int(sys.argv[2])

    # Build folder/file paths
    base = f"{start}to{end}"
    audio_dir = f"./{base}/{start}to{end}_segments"
    transcription_dir = f"./{base}/{start}to{end}_transcriptions"
    segment_info_path = f"./{base}/{start}to{end}_segments/segment_info.csv"

    # Adjust batch size as desired
    BATCH_SIZE = 40
    BEAM_SIZE = 3
    CONFIDENCE_THRESHOLD = 0.8
    MAX_LEN = 30
    NUM_WORKERS = 4
    PREFETCH_FACTOR = 2

    # Start timing the entire process
    start_time = time.time()

    # Load model with progress indication
    print("Loading model...")
    model_load_start = time.time()
    model = WhisperModel("whisper-d-ct2", device="cuda", compute_type="float16")
    model_load_end = time.time()
    print(f"Model loaded in {model_load_end - model_load_start:.2f} seconds")

    # Process the directory of audio segments
    total_files, total_transcription_time = transcribe_with_dataloader(
        audio_dir=audio_dir,
        output_dir=transcription_dir,
        model=model,
        batch_size=BATCH_SIZE,
        beam_size=BEAM_SIZE,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        max_len=MAX_LEN,
        num_workers=NUM_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
    )

    # End timing
    end_time = time.time()

    # Print a concise summary
    print("\nTranscription Summary:")
    print(f"Total files processed: {total_files}")
    print(f"Total transcription time: {total_transcription_time:.2f} seconds")
    print(f"Total execution time: {end_time - start_time:.2f} seconds")
    if total_files > 0:
        print(
            f"Average time per file: {total_transcription_time / total_files:.2f} seconds"
        )
