import torch
import librosa
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
import time

# Memuat model dan tokenizer yang telah dilatih sebelumnya
model = Wav2Vec2ForCTC.from_pretrained("C:\\Users\\USER\\.cache\\huggingface\\hub\\models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese\\snapshots\\e5e699fa5aa5bdce999276a90a29589587b58ac9")
tokenizer = Wav2Vec2Processor.from_pretrained("C:\\Users\\USER\\.cache\\huggingface\\hub\\models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese\\snapshots\\e5e699fa5aa5bdce999276a90a29589587b58ac9")

# Menetapkan parameter untuk transkripsi audio
audio_path = "halo.mp3"
sampling_rate = 16000
duration = 10

# Memuat file audio dan mengekstrak fiturnya
start_time = time.time()
audio, rate = librosa.load(audio_path, sr=sampling_rate, duration=duration, mono=True, res_type='kaiser_fast')
input_values = tokenizer(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values

# Membuat transkripsi
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.decode(predicted_ids[0])
end_time = time.time()

# Menghitung durasi proses transkripsi
transcription_time = end_time - start_time

# Mendapatkan durasi audio dalam menit dan detik
duration_min = int(duration // 60)
duration_sec = int(duration % 60)

# Mencetak transkripsi bersama dengan durasi audio dan durasi proses transkripsi
print()
print(f"durasi audio: {duration_min}:{duration_sec}")
print(f"durasi transkripsi: {transcription_time:.2f} detik")
print(f"transkripsi mp3: {transcription}")
print()
