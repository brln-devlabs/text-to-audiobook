# Hinweise zu diesem Fork

Dieser Fork von [SecCodeSmith/text-to-audiobook](https://github.com/SecCodeSmith/text-to-audiobook)
ist auf deutsche Texte und Grafikkarten mit wenig Speicher angepasst (die Python-Version
wurde mit einer NVIDIA RTX A500 mit 4 GB getestet). Er ist inzwischen in Rust neu umgesetzt:
ein einzelnes Programm ohne Python, PyTorch oder ffmpeg. Einstellungen, Zwischenstände und
Ausgabeordner der Python-Version werden weiterverwendet.

## Was anders ist

- **Absatzpausen:** Abschnitte enden immer an einem Absatzende. Nach jedem Absatz fügt das
  Programm die eingestellte Absatzpause ein (`SILENCE_AFTER_PARAGRAPH_MS`, Standard 1200 ms).
  Pausen landen auch dann an der richtigen Stelle, wenn ein Abschnitt intern in mehrere
  Audioteile zerlegt wird (`chunk_NNN_0.wav`, `_1.wav`, …).
- **Deutsche Satzgrenzen:** Sätze, die mit Umlaut (Ä, Ö, Ü) oder deutschem
  Anführungszeichen („) beginnen, werden als Satzanfang erkannt.
- **Feste Sprache:** `TTS_LANGUAGE = "german"`. Qwen3-TTS spricht jeden Abschnitt deutsch,
  unabhängig davon, was das kleine Llama-Modell erkennt. Die zehn Spracherkennungs-Abfragen
  am Anfang entfallen. `"auto"` stellt das ursprüngliche Verhalten wieder her.
- **Modellauswahl in der Oberfläche:** im Bereich „Quick Config“ und unter „Settings → TTS“
  zwischen 0.6B (ab 4 GB VRAM, Standard) und 1.7B (ab 6 GB VRAM) wählen. Unter „Settings“
  lassen sich dort auch Sprache und Rechengenauigkeit einstellen, unter „LLM“ die Zahl der
  Llama-Schichten auf der GPU.
- **bfloat16 statt float16:** Grafikkarten ohne bfloat16 fallen automatisch auf float16
  zurück. Auf der CPU rechnet das Programm immer mit float32.
- **Llama ist optional und standardmäßig aus** (`USE_LLM = false`): Kapiteltitel kommen aus
  der ersten Überschrift jeder Datei, der Buchtitel aus dem Projektnamen, die Reihenfolge
  aus den Dateinamen. Dabei werden Zahlwörter erkannt (`Kapitel_zwei` nach `Kapitel_eins`),
  Vorwort, Prolog und Einleitung kommen zuerst, Nachwort, Epilog und Anhang zuletzt.
  Die Base-Modelle von Qwen3-TTS können mit den Emotionen des LLM ohnehin nichts anfangen,
  und Stufe 1 ist ohne LLM in Sekunden fertig. Wer Llama trotzdem nutzen will, baut mit
  `--features llm` und setzt `USE_LLM = true`. `LLM_GPU_LAYERS = 0` lässt dann den
  Grafikspeicher ganz für die Sprachausgabe frei.
- **Zahlen ausschreiben** (`NORMALIZE_NUMBERS = true`, Settings → Text Processing): vor der
  Sprachausgabe werden Zahlen, Jahreszahlen, Datumsangaben, Ordnungszahlen, Uhrzeiten,
  Geldbeträge, Prozent, Maßeinheiten, Bereiche und gängige Abkürzungen ausgeschrieben, nur
  bei deutschen Abschnitten. Ausprobieren ohne Hörbuch:
  `text-to-audiobook normalize "Am 3. Mai 1995 um 14:30 Uhr kostete es 5,50 €."`
  Ordnungszahlen werden nur mit eindeutigem Wort davor umgewandelt („am 3.“, „der 3.“,
  „in der 3.“). Römische Zahlen nach Wörtern wie Kapitel, Teil, Band oder Akt werden zur
  Grundzahl („Kapitel IV“ → „Kapitel vier“), vor „Jahrhundert“ mit eindeutigem Wort davor
  zur Ordnungszahl („im XIX. Jahrhundert“ → „im neunzehnten Jahrhundert“).
  Für Fachtexte: Abschnittsnummern („Abschnitt 3.3“ → „drei Punkt drei“, „Anhang B.1“ →
  „B Punkt eins“; „Anhang C“ bleibt ein Buchstabe), Modellgrößen („1.3B“ → „eins Komma drei
  Milliarden“, „125M“), GB/MB/TB, Versionen („SQuAD2.0“ → „zwei Punkt null“), arXiv-Nummern
  Ziffer für Ziffer, Tausender mit Leerzeichen („600 000“ → „sechshunderttausend“), Literaturangaben („33:1877–1901“ → „Band dreiunddreißig, Seiten …“;
  Seitenzahlen nie als Jahreszahl), Hochzahlen („10²“ → „zehn hoch zwei“) und Rechenzeichen (=, <, >, ≤, ≥,
  ≈, ∈, ×, ^).
- **Textbereinigung** (immer): Markdown-Escapes („\*“, „\_“, „\[“) werden zum Zeichen,
  Fußnotenzeichen („Carlini¹“, „¹ Google Research“, „\*“) entfallen, Tabellen werden Zeile
  für Zeile als Aufzählung gelesen, jeder Listenpunkt (z. B. Literaturangaben) wird ein
  eigener Absatz, und bei entfernten Links fällt das „unter“ davor mit weg. Sätze werden
  nicht mehr bei Abkürzungen, Initialen oder Ordnungszahlen getrennt („z. B.“, „et al.“,
  „Christopher A. Choquette“, „am 15. September“). Gesprochene Kapitelnamen
  (`VOCALIZE_CHAPTER_NAMES`) nehmen die erste Überschrift der Datei, sonst den Dateinamen in
  seiner Schreibweise; derselbe Name steht dann auch als Kapiteltitel in `prepared.txt` und in den
  Titel-Tags (statt „Chapter 1“). Kapiteltitel, die das LLM erfindet und die nicht im Text stehen
  (etwa „The Beginning“), werden verworfen.
- **Geänderter Text wird erkannt:** Ändert sich der vorbereitete Text (neue Programmversion,
  bearbeitete Eingabe), verwirft Stufe 1 die betroffenen Einträge im Cache, statt sie nach
  ihrer Nummer weiterzuverwenden (ein Eintrag gilt nur bei genau gleichem Text); „Clear Cache“ ist dafür nicht mehr nötig. Neben jedem
  Audiostück liegt `chunk_NNNNNN.txt` mit dem gesprochenen Text. Passt er nicht mehr (anderer
  Text, andere Prüfentscheidung), erzeugt Stufe 2 das Stück neu.
- **Unsichere Zahlen prüfen** (`REVIEW_NUMBERS = true`, zusätzlich zu `NORMALIZE_NUMBERS`):
  Stellen, die die Regeln nicht sicher entscheiden können, also Ordnungszahlen ohne
  eindeutiges Wort davor („Sie kam als 3. ins Ziel“, „Er wurde 3.“) und römische Zahlen
  außerhalb der sicheren Muster („Heinrich IV.“, „Ludwig XIV“), werden gesammelt. Initialen in Namenslisten („Foster, C., Phang, J.“) gelten nicht als
  römische Zahl, die letzte Zahl einer Zahlenliste oder Tabellenzeile („25, 25, 13.“) nicht
  als Ordnungszahl. Nach
  „1. Prepare Segments“ öffnet sich das Fenster „Review Numbers“ und zeigt jede Stelle im
  Satz mit Vorschlägen („dritte“, „dritten“, „Dritter“, „drei.“, „der Vierte“, „des
  Vierten“ …). Ein Klick übernimmt den Vorschlag und springt zur nächsten Stelle; mit
  „Apply to all identical spots“ gilt die Wahl für alle gleichen Stellen. Die letzte Auswahl
  („keep as written“) lässt die Stelle genau so, wie sie im Text steht; auch die Regeln
  ändern sie dann nicht. „Program default“ überlässt die Stelle den Regeln, „Skip“ lässt sie
  offen. Bei Gliederungspunkten am
  Satzanfang („II. Kapitalistische Revolution“) steht „Zweitens:“ zuerst. Ist der Satz allein
  nicht eindeutig, klappt „▼ More context“ je drei Abschnitte davor und danach auf; bei der
  nächsten Stelle ist die Ansicht wieder zu (Kommandozeile: `m`). Das Fenster lässt sich jederzeit
  über „Review Numbers“ neben der Projektliste öffnen; vor „2. Generate Audio“ fragt das
  Programm nach, wenn noch Stellen offen sind. Ohne Oberfläche: `text-to-audiobook review`.
  Die Entscheidungen liegen in `cache/<Textname>/review.json`; „Clear Cache“ löscht sie mit.
  Lesbar stehen sie in `output/<Textname>/text/review.txt`: jede unsichere Stelle mit ihrem
  Satz, der Entscheidung („OFFEN“, „Program default“, „keep as written“ oder die Auswahl) und
  wie der Ausschnitt gesprochen wird. Die Datei wird beim Vorbereiten, beim Schließen des
  Prüffensters und nach `text-to-audiobook review` neu geschrieben.
  Sie gelten für neu erzeugte Audiostücke.
- **Zwei Knöpfe:** „1. Prepare Segments“ bereitet den Text vor und schreibt ihn nach
  `output/<Textname>/text/prepared.txt`, eine Zeile pro Audiostück mit dessen Nummer
  (`000012 | …` gehört zu `chunk_000012.wav`). Mit `NORMALIZE_NUMBERS` kommt `spoken.txt`
  dazu: der Text, wie er gesprochen wird. „2. Generate Audio“ erzeugt die Sprache und baut das
  Hörbuch zusammen.
- **Automatische Qualitätsprüfung** (`AUTO_CLEANUP = true`, Settings → Audio): nach der
  Sprachausgabe prüft das Programm jedes Audiostück (Sprachanteil, Lautstärke,
  Übersteuerung, Stimmähnlichkeit), löscht durchgefallene und erzeugt sie mit einem anderen
  Seed neu. Das geschieht bis zu zweimal; danach bleibt der letzte Versuch erhalten, damit
  kein Text fehlt. Ohne Oberfläche gibt es weiter `text-to-audiobook cleanup` und
  `text-to-audiobook assemble`.
- **Neutrale Erzählstimme** im `NARRATOR_PROMPT` und deutscher Text für die automatisch
  erzeugte Referenzstimme.

## Installation

Benötigt werden [Rust](https://rustup.rs/) ab Version 1.95, CMake und ein C/C++-Compiler
(llama.cpp, der MP3-Encoder LAME und SQLite werden mitgebaut).

Unter Ubuntu/Debian zum Beispiel:

```bash
sudo apt install build-essential cmake libxkbcommon-x11-0
```

Programm bauen, ohne Grafikkarte:

```bash
cargo build --release
```

Mit NVIDIA-Grafikkarte (CUDA-Toolkit mit `nvcc` nötig). Damit rechnet die Sprachausgabe
auf der GPU:

```bash
cargo build --release --features cuda
```

Mit dem optionalen Llama zusätzlich `llm` angeben (`--features cuda,llm`); Llama läuft dann
auf der CPU. Soll auch Llama auf die GPU, mit `--features llm-cuda` bauen. llama.cpp bindet CUDA dann
statisch ein und braucht `libcudart_static.a`, `libcublas_static.a`, `libcublasLt_static.a`
und `libculibos.a`. Gesucht wird in `/usr/local/cuda`, `/opt/cuda` oder im Ordner aus
`CUDA_LIBRARY_PATH`. Bei den Debian-Paketen liegen sie, falls installiert, meist unter
`/usr/lib/x86_64-linux-gnu`:

```bash
CUDA_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu cargo build --release --features llm-cuda
```

Das CUDA-Toolkit darf nicht neuer sein als die CUDA-Version des Treibers. Beim Treiber 550
(CUDA 12.4) also ein Toolkit bis Version 12.4 installieren. Prüfen:

```bash
nvcc --version
nvidia-smi
```

Danach liegt das Programm unter `target/release/text-to-audiobook`. Gestartet wird es im
Projektordner (dort liegen `settings.json`, `input/`, `output/`, `cache/`):

```bash
./target/release/text-to-audiobook
```

Ohne Befehl öffnet sich die Oberfläche. `text-to-audiobook --help` zeigt die Befehle für die
Kommandozeile, etwa `run`, `prepare` oder `generate`.

## Modelle

Beim ersten Start lädt das Programm die Modelle herunter:

- Llama 3.2 1B (GGUF, etwa 800 MB) nach `models/`, nur mit `USE_LLM = true`
- Qwen3-TTS (0.6B oder 1.7B) in den Hugging-Face-Zwischenspeicher
  (`~/.cache/huggingface`). Von der Python-Version schon geladene Modelle werden
  weiterverwendet.

Falls Hugging Face den Download ablehnt, ein Zugangstoken setzen:

```bash
export HF_TOKEN=hf_...
```

## Referenzstimme

Ohne eigene Referenz erzeugt das Programm eine zufällige Stimme. Besser ist eine eigene
Aufnahme von 10 bis 15 Sekunden (eine Stimme, kein Hall, keine Musik):

```bash
ffmpeg -i aufnahme.mp3 -t 15 -ac 1 -ar 24000 cache/voice_reference.wav
```

Andere Abtastraten rechnet das Programm selbst um, ffmpeg wird dafür nur zum Zuschneiden
und Umwandeln der Aufnahme gebraucht.

Eine frei verwendbare deutsche Stimme (CC0) gibt es im
[Thorsten-Voice-Datensatz](https://huggingface.co/datasets/Thorsten-Voice/TV-44kHz-Full).

## Einstellungen

Alle Werte lassen sich in der Oberfläche einstellen oder direkt in `settings.json` im
Projektordner, zum Beispiel:

```json
{
  "TTS_LANGUAGE": "german",
  "TTS_MODEL_NAME": "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
  "TTS_DTYPE": "bfloat16",
  "NORMALIZE_NUMBERS": true,
  "REVIEW_NUMBERS": true,
  "AUTO_CLEANUP": true,
  "USE_LLM": false,
  "SILENCE_AFTER_PARAGRAPH_MS": 1200,
  "SILENCE_AFTER_PERIOD_MS": 500
}
```

`TTS_MODEL_NAME` kann auch ein Ordner mit einem heruntergeladenen Modell sein.

## Unterschiede zur Python-Version

- Kapitel werden in Lesereihenfolge nummeriert und zusammengefügt. Die Python-Version
  sortierte sie alphabetisch, sodass `10_…` vor `2_…` kam.
- Der Stimmvergleich der Qualitätsprüfung ist aktiv. In der Python-Version fehlte
  dafür die Bibliothek librosa, der Vergleich entfiel dort stillschweigend.
- „Force Stop“ hält am nächsten möglichen Punkt an und löscht das angefangene Audiostück.
  Ein Modellaufruf, der schon läuft, wird noch zu Ende gerechnet.
- MP3-Dateien mit 24 kHz haben 160 kbit/s, das Maximum für diese Abtastrate.

## Wichtig bei schon verarbeiteten Texten

Nach einem Modellwechsel verwendet das Programm bereits erzeugte Audiostücke weiter. Damit ein
Text komplett mit dem neuen Modell gesprochen wird, seinen Ordner im Ausgabeordner löschen
(`output/<Textname>`). Die Referenzstimme muss nicht neu erzeugt werden.

Die Aufteilung dieses Forks passt nicht zu Zwischenständen des ursprünglichen Projekts. Wer
Texte noch mit dem ursprünglichen Projekt verarbeitet hat, löscht vor dem ersten Lauf für
jeden dieser Texte seinen Zwischenstand im `cache`-Ordner und seinen Ordner im gewählten
Ausgabeordner (Beispiel für einen Text namens `Engels` und den Standard-Ausgabeordner
`output`):

```bash
rm -rf cache/Engels output/Engels
```

Zwischenstände der Python-Version dieses Forks bleiben gültig. Die Referenzstimme
`cache/voice_reference.wav` bleibt in beiden Fällen erhalten.

## Fehlerbehebung

- **„could not find native static library `cudart_static`“ beim Bauen:** mit
  `--features cuda` statt `llm-cuda` bauen (siehe Installation).
- **„CUDA unknown error“ nach Standby:** das Kernelmodul neu laden oder neu starten:
  `sudo rmmod nvidia_uvm && sudo modprobe nvidia_uvm`
- **„CUDA out of memory“:** beim 0.6B-Modell bleiben und andere Programme schließen, die
  die Grafikkarte nutzen.
- **Oberfläche startet unter Linux nicht (Fehler mit `xkbcommon`):** das Paket
  `libxkbcommon-x11-0` installieren.
- **Ausprobieren ohne Modelle:** `text-to-audiobook --mock-models run` durchläuft alle
  Schritte und schreibt Stille statt Sprache.
