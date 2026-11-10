# Hinweise zu diesem Fork

Dieser Fork von [SecCodeSmith/text-to-audiobook](https://github.com/SecCodeSmith/text-to-audiobook)
ist auf deutsche Texte und Grafikkarten mit wenig Speicher (getestet: NVIDIA RTX A500, 4 GB)
angepasst.

## Was anders ist

- **Absatzpausen:** Abschnitte enden jetzt immer an einem Absatzende. Nach jedem Absatz
  fügt das Programm die eingestellte Absatzpause ein (`SILENCE_AFTER_PARAGRAPH_MS`,
  Standard 1200 ms). Pausen landen auch dann an der richtigen Stelle, wenn ein Abschnitt
  intern in mehrere Audioteile zerlegt wird (`chunk_NNN_0.wav`, `_1.wav`, …).
- **Deutsche Satzgrenzen:** Sätze, die mit Umlaut (Ä, Ö, Ü) oder deutschem
  Anführungszeichen („) beginnen, werden als Satzanfang erkannt.
- **Feste Sprache:** `TTS_LANGUAGE = "german"`. Qwen3-TTS spricht jeden Abschnitt deutsch,
  unabhängig davon, was das kleine Llama-Modell erkennt. Die zehn Spracherkennungs-Abfragen
  am Anfang entfallen. `"auto"` stellt das ursprüngliche Verhalten wieder her.
- **Kleines Modell:** Standard ist `Qwen/Qwen3-TTS-12Hz-0.6B-Base` (passt in 4 GB).
- **bfloat16 statt float16:** behebt den Absturz „CUDA error: device-side assert triggered“.
  Grafikkarten ohne bfloat16 fallen automatisch auf float16 zurück.
- **Llama auf der CPU:** `LLM_GPU_LAYERS = 0` lässt den Grafikspeicher ganz für die
  Sprachausgabe frei.
- **Neutrale Erzählstimme** im `NARRATOR_PROMPT` und deutscher Text für die automatisch
  erzeugte Referenzstimme.

## Installation unter Linux mit NVIDIA-Treiber 550 (CUDA 12.4)

Aktuelle torch-Versionen von PyPI brauchen einen Treiber mit CUDA 13. Für ältere Treiber
zuerst torch 2.14 mit CUDA 12.6 installieren, danach den Rest:

```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip uv
uv pip install torch==2.14.0 torchvision==0.29.0 torchaudio \
  --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Die letzte Zeile muss `2.14.0+cu126 True` ausgeben.

## Referenzstimme

Ohne eigene Referenz erzeugt das Programm eine zufällige Stimme. Besser ist eine eigene
Aufnahme von 10 bis 15 Sekunden (eine Stimme, kein Hall, keine Musik):

```bash
ffmpeg -i aufnahme.mp3 -t 15 -ac 1 -ar 24000 cache/voice_reference.wav
```

Eine frei verwendbare deutsche Stimme (CC0) gibt es im
[Thorsten-Voice-Datensatz](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full).

## Einstellungen

Alle neuen Werte lassen sich in `settings.json` im Projektordner überschreiben, zum Beispiel:

```json
{
  "TTS_LANGUAGE": "german",
  "TTS_MODEL_NAME": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "TTS_DTYPE": "bfloat16",
  "LLM_GPU_LAYERS": 0,
  "SILENCE_AFTER_PARAGRAPH_MS": 1200,
  "SILENCE_AFTER_PERIOD_MS": 500
}
```

## Wichtig bei schon verarbeiteten Texten

Die neue Aufteilung passt nicht zu alten Zwischenständen. Vor dem ersten Lauf mit diesem
Fork für jeden schon verarbeiteten Text seinen Zwischenstand im `cache`-Ordner und seinen
Ordner im gewählten Ausgabeordner löschen (Beispiel für einen Text namens `Engels` und den
Standard-Ausgabeordner `output`):

```bash
rm -rf cache/Engels output/Engels
```

Die Referenzstimme `cache/voice_reference.wav` bleibt dabei erhalten.

## Fehlerbehebung

- **„flash-attn is not installed“:** nur eine Warnung, kann ignoriert werden.
- **„CUDA unknown error“ nach Standby:** das Kernelmodul neu laden oder neu starten:
  `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`
- **„CUDA out of memory“:** beim 0.6B-Modell bleiben und andere Programme schließen, die
  die Grafikkarte nutzen.
