import librosa
import torch
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor

# Memuat model dan tokenizer yang telah dilatih sebelumnya
model = Wav2Vec2ForCTC.from_pretrained("C:\\Users\\USER\\.cache\\huggingface\\hub\\models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese\\snapshots\\e5e699fa5aa5bdce999276a90a29589587b58ac9")
tokenizer = Wav2Vec2Processor.from_pretrained("C:\\Users\\USER\\.cache\\huggingface\\hub\\models--indonesian-nlp--wav2vec2-indonesian-javanese-sundanese\\snapshots\\e5e699fa5aa5bdce999276a90a29589587b58ac9")

# Menetapkan parameter untuk transkripsi audio
audio_path = "akudion.wav"
duration = 5
sampling_rate = 16000

# Memuat file audio
audio, sr = librosa.load(audio_path, sr=sampling_rate, duration=duration, mono=True)

input_values = tokenizer(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values

# Membuat transkripsi
with torch.no_grad():
    logits = model(input_values).logits

predicted_ids = torch.argmax(logits, dim=-1)
transcription = tokenizer.batch_decode(predicted_ids)[0]

# Mencetak transkripsi dan durasi audio
print("Transkripsi audio:", transcription)
print("Durasi audio:", duration, "detik")
