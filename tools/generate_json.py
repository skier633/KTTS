import os
import glob
import json

# Define your paths
wav_dir = 'sample_dataset/output'  # The directory containing A_B_C.wav files
audio_ref_dir = 'sample_dataset/audio'   # The directory containing A.wav
text_ref_dir = 'sample_dataset/text'    # The directory containing *_B.txt
output_file = 'sample_dataset/train_step1.json'

results = []

# 1. Get all wav files in the format A_B_C.wav
for wav_path in glob.glob(os.path.join(wav_dir, "*_*_*.wav")):
    wav_filename = os.path.basename(wav_path)
    # Remove .wav suffix for the ID
    wav_id = os.path.splitext(wav_filename)[0]
    
    # Split the filename to get parts A, B, and C
    parts = wav_id.split('_')
    if len(parts) < 3:
        continue
        
    speaker_a = parts[0]
    identifier_b = parts[1]

    # 2. Construct the audio path (../audio/A.wav)
    audio_file_path = os.path.join(audio_ref_dir, f"{speaker_a}.wav")

    # 3. Find the unique text file (../text/*_B.txt)
    text_pattern = os.path.join(text_ref_dir, f"*_{identifier_b}.txt")
    text_files = glob.glob(text_pattern)
    
    text_content = ""
    if text_files:
        # Assuming the first match is the unique one
        with open(text_files[0], 'r', encoding='utf-8') as f:
            text_content = f.read().strip()

    # 4. Create the record
    record = {
        "id": wav_id,
        "audio": wav_path,
        "text": text_content,
        "speaker": speaker_a
    }
    results.append(record)

# 5. Output to JSON
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=4, ensure_ascii=False)

print(f"Successfully processed {len(results)} files into {output_file}")

