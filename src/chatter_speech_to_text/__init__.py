# # El nombre es text_to_speech_chatterbox pero lo escribí al revez :(

import torchaudio as ta
from chatterbox.mtl_tts import ChatterboxMultilingualTTS
print("Cargando el modelo...")

modelo = ChatterboxMultilingualTTS.from_pretrained(
    device="cpu"
)

texto = "Hola, bienvenido a IA generativa, vamos a hacer texto, imagenes y si la pc aguanta (que adelanto que no lo va a hacer) haremos videos tambien"
print("Generando el audio...")

audio = modelo.generate(
    texto,
    language_id="es",
    exaggeration=0.7,
    cfg_weight=0.4,
    temperature=0.8,
    repetition_penalty=1.2,
    min_p=0.05,
    top_p=1.0
)

ta.save(
    "audio.wav",
    audio,
    modelo.sr
)

print("Audio generado.")
