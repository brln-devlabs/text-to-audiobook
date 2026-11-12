# App-Icon

Ein aufgeschlagenes Buch: links Textzeilen, rechts eine Schallwelle – aus Text wird ein
Hörbuch. Farben passend zur Oberfläche (Blau/Violett, Orange als Akzent).

| Datei | Zweck |
|---|---|
| `icon.svg` | Vorlage für alle Größen ab 40 px |
| `icon-small.svg` | vereinfachte Fassung für 16–32 px (dickere Striche, weniger Details) |
| `icon-256.png` | Fenster-, Taskleisten- und Dock-Icon (`src/gui.rs`), unter Linux auch für den Startmenü-Eintrag (`src/desktop_entry.rs`) |
| `icon-64.png` | Logo neben dem Titel in der Oberfläche (`src/gui.rs`) |
| `icon.ico` | Windows: in die `.exe` eingebettet (`build.rs`); 16–32 px aus `icon-small.svg`, 40–256 px aus `icon.svg` |

Nach einer Änderung an den SVGs die PNGs und die ICO neu erzeugen, z. B. mit Inkscape und
ImageMagick:

```bash
inkscape icon.svg -w 256 -o icon-256.png
inkscape icon.svg -w 64 -o icon-64.png
for s in 256 128 64 48 40; do inkscape icon.svg -w $s -o ico-$s.png; done
for s in 32 24 20 16; do inkscape icon-small.svg -w $s -o ico-$s.png; done
magick ico-256.png ico-128.png ico-64.png ico-48.png ico-40.png ico-32.png ico-24.png ico-20.png ico-16.png icon.ico
rm ico-*.png
```
