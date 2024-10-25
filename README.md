# TTS_IITR
Fine Tuned TTS model on different Languages
# Importing necessary libraries
import os
import torch
from IPython.display import Audio
import soundfile as sf
from speechbrain.pretrained import EncoderClassifier
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

spk_model_name = "speechbrain/spkrec-xvect-voxceleb"

device = "cuda" if torch.cuda.is_available() else "cpu"
speaker_model = EncoderClassifier.from_hparams(
    source=spk_model_name,
    run_opts={"device": device},
    savedir=os.path.join("/tmp", spk_model_name),
)
# Load a sample from the dataset for speaker embedding
try:
    dataset = load_dataset("mozilla-foundation/common_voice_17_0", "hi", split="validated", trust_remote_code=True)
    dataset = dataset.cast_column("audio", Audio(sampling_rate=16000))
    sample = dataset[0]
    speaker_embedding = create_speaker_embedding(sample['audio']['array'])
except Exception as e:
    print(f"Error loading dataset: {e}")
    # Use a random speaker embedding as fallback
    speaker_embedding = torch.randn(1, 512)

def create_speaker_embedding(waveform):
    with torch.no_grad():
        speaker_embeddings = speaker_model.encode_batch(torch.tensor(waveform))
        speaker_embeddings = torch.nn.functional.normalize(speaker_embeddings, dim=2)
        speaker_embeddings = speaker_embeddings.squeeze().cpu().numpy()
    return speaker_embeddings

# You can access the fine-tuned model using the command mentioned below
model = SpeechT5ForTextToSpeech.from_pretrained("Sana1207/Hindi_SpeechT5_finetuned")

# Enter the text you want to convert in Audio
text = "API stands for Application programming interface"
inputs = processor(text, return_tensors="pt")
speech = model.generate_speech(inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

Audio(speech.numpy(), rate=16000)
# Save the audio to a file (e.g., 'output.wav')
sf.write('output.wav', speech.numpy(), 16000)
  
