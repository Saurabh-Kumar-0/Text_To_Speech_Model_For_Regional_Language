# TTS_IITR
Fine Tuned TTS model on different Languages
# importing necessary libraries
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan

processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
# You can access the fine-tuned model using the command mentioned below
model = SpeechT5ForTextToSpeech.from_pretrained("Sana1207/Hindi_SpeechT5_finetuned")

vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")
