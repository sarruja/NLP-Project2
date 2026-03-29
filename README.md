# NLP-Project2 – Swiss German Dialects  

Team: 2 Personen  
Deadline Poster: 14.04.2026  
Postersession: 17.04.2026 

Abgabe: Wissenschaftliches Poster (A0, PDF, 1 Seite)  

---

## Forschungsfrage  
> ***Wie verändert sich die ASR-Qualität von Whisper v1→v2→v3 auf Schweizerdeutsch, und welche Fehlertypen dominieren?***  
- Fehleranaylse mit verschiedene Whisper Modellen (v1, v2, v3 und WER vs CHR wenn zeit)

 
### Mögliche Fehlertypen (für Analyse)
 
- Substitutionen: falsches Wort (`"ihr"` → `"er"`)
- Deletions: Wort fehlt komplett
- Insertions: extra Wort eingefügt
- Zahlen: `"zwanzig"` → `"20"`
- Dialekt-spezifische Wörter die falsch transkribiert werden
- - Eigennamen / Ortsnamen (z.B. "Zürich" → "Zurich")
- Seltene Wörter zerstückelt: `"Postulat"` → `"Post und Art"`
- Phonetische Verwechslungen: `"Syrer"` → `"Säurer"`
- Code-switching: Dialekt → Hochdeutsch


### Mögliche andere Modelle zum Vergleich:
- Originale Whisper-Modelle (OpenAI)
  - openai/whisper-large-v1, v2, v3 — das verwenden wir sicher

- Whisper-Varianten von anderen Anbietern
  - Distil-Whisper (HuggingFace, 2023): komprimierte Version via Knowledge Distillation — 51% weniger Parameter, 5.8x schneller, aber nur ~1% schlechter WER Towards AI  
  - Faster-Whisper (SYSTRAN): technisch dasselbe Modell, aber optimierte Inferenz — kein anderes Training  
  - WhisperX (Uni Oxford): erweitert Whisper um word-level Timestamps und Speaker Diarization Towards AI

- Andere ASR-Modelle (nicht Whisper-basiert) zum Vergleich  
  - facebook/wav2vec2 (fine-tuned auf Deutsch)  
  - das von fhnw
  
Das STT4SG-eigene Baseline-Modell aus dem Paper — das wäre eigentlich ein guter Vergleichspunkt.

Daten: STT4SG-350 Korpus

_Wie funktioneirt Whisper (kurz und bündig)_
Whisper auf Audioclips laufen lassen und Output mit Ground Truth vergleichen:   
```
Audio: [Jemand sagt "Guete Morge, wie gaats?"]      
Ground Truth: "Guten Morgen, wie geht es?"    
Whisper Output: "Guten Morgen, wie geht's dir?"     
```
→ WER = Anteil falsch erkannter Wörter  

---

## Project Structure
 
```
Project_2/
│
├── data/                        ← alle Datenfiles (lokal / Google Drive, nicht in Git)
│   ├── clips__test/             ← Audiodaten Test   
│   ├── clips__train_valid/      ← Audiodaten Train (nicht verwendet)  
│   ├── test.tsv                 ← Ground Truth + Metadaten für Test-Set  
│   ├── train_all.tsv
│   ├── train_balanced.tsv
│   └── valid.tsv
│
├── results/                     ← Ergebnisse
│   ├── checkpoint_openai_whisper-large-v1.csv
│   ├── checkpoint_openai_whisper-large-v2.csv
│   ├── checkpoint_openai_whisper-large-v3.csv
│   ├── wer_overall.csv
│   └── wer_by_region.csv
│
├── figures/                     ← Figures für das Poster (wird automatisch generiert)
│   ├── fig_wer_overall.png
│   └── .... .png
│
├── whisper_pipeline.ipynb       ← Hauptnotebook: Test-Run (lokal) + Full-Run (Colab)
├── .gitignore
└── README.md
```
 
> ⚠️ `data/` ist in `.gitignore` — diese müssen lokal vorhanden sein oder auf Google Drive liegen (siehe unten).
 
---

## Setup 
  
### HuggingFace Account
 
Um die Whisper-Modelle von HuggingFace herunterzuladen, wird ein persönlicher Access Token benötigt.
 
#### Account erstellen
1. Geh auf [huggingface.co](https://huggingface.co) → Sign Up
2. Mit Email registrieren (ich habe die ZHAW-Mail verwenden), als Username habe ich das ZHAW-Kürzel verwendet (aber ist egal)
3. Email Adresse bestätigen (Mailbox überprüfen)
 
#### Token erstellen
1. [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) → `New token`
2. Token type: **Read**
3. Name z.B. `nlp_project2` → `Create token`
4. Token kopieren und sicher aufbewahren (wird nur einmal angezeigt!)

> ⚠️ Token nicht ins Git einchecken und nicht teilen — jede Person erstellt ihren eigenen!

---

### Lokal (für Test-Run)
 
#### 1. Repository klonen
```bash
git clone <repo-url>
cd Project_2
```
 
### 2. Daten beschaffen
In `data/` die nötigen Daten speichern (werden nicht ins Git eingecheckt):
- `test.tsv` ← Ground Truth für alle Experimente
- `clips__test/` ← alle MP3-Audioclips (~701 MB)
 
### 3. Python Packages installieren
Nötge Package isntallierne (im Jupyter Notbook hats bereits eine Zelle - gegebfalls noch weitere isntallieren wenn nötig)  
`conda install -c conda-forge ffmpeg -y `, muss im PowerShell oder Anaconda Prompt ausgeführt werden  
Ams Schluss noch Kernel neustarten

### 4. HuggingFace Token in Colab hinterlegen
1. Links in Colab auf das 🔑 **Secrets** Icon klicken
2. `Add new secret` → Name: `HF_TOKEN` → Value: Token einfügen
3. **Notebook-Zugriff Toggle: AN**
 
Die folgenden Zeilen sind bereits im Notebook Setup-Block vorhanden und lesen den Token automatisch:
```python
from google.colab import userdata
import os
os.environ['HF_TOKEN'] = userdata.get('HF_TOKEN')
```

---
 
### Google Colab (für Full-Run)
 
Der Full-Run (~24k Samples × 3 Modelle) braucht eine GPU und läuft auf Google Colab.
 
### 1. Google Drive vorbereiten
Folgende Ordnerstruktur auf Google Drive erstellen:
```
MyDrive/
└── NLP_Project2/
    ├── data/
    │   ├── test.tsv
    │   └── clips__test/       ← 701 MB hochladen (dauert ~10-20 Min)
    └── results/               ← leeren Ordner erstellen
```
 
### 2. Notebook auf Colab öffnen
- **Option A:** `File → Open notebook → GitHub → Repo URL` eingeben
  - Link zu Google Colab: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/sarruja/NLP-Project2/blob/main/whisper_pipeline.ipynb)  
- **Option B:** Notebook in `NLP_Project2/` auf Drive hochladen → Doppelklick → öffnet in Colab
 
### 3. GPU aktivieren
`Runtime → Change runtime type → T4 GPU → Save`
 
### 4. Im Notebook: Pfade auf Colab umstellen
Im Setup-Block (Zelle 0) die lokalen Zeilen auskommentieren und Colab-Zeilen einkommentieren:
```python
from google.colab import drive
drive.mount('/content/drive')
DATA_DIR    = '/content/drive/MyDrive/NLP_Project2/data/'
AUDIO_DIR   = '/content/drive/MyDrive/NLP_Project2/data/clips__test/'
RESULTS_DIR = '/content/drive/MyDrive/NLP_Project2/results/'
TEST_FILE   = '/content/drive/MyDrive/NLP_Project2/data/test.tsv'
```

### 6. Full-Run starten
Alle Zellen von oben nach unten ausführen. Checkpoints werden alle 200 Samples auf Google Drive gespeichert → bei Absturz einfach neu starten, er macht weiter wo er aufgehört hat.
 
> ⚠️ Colab trennt die Session nach ~90 Min Inaktivität — dank Checkpoints kein Datenverlust!
 

 
