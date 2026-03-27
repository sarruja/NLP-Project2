# NLP-Project2 – Swiss German Dialects  

Team: 2 Personen  
Deadline Poster: 14.04.2026  
Postersession: 17.04.2026 

Abgabe: Wissenschaftliches Poster (A0, PDF, 1 Seite)  

## Forschungsfrage  
Wie verändert sich die ASR-Qualität von Whisper v1→v2→v3 auf Schweizerdeutsch, und welche Fehlertypen dominieren?
- Fehleranaylse mit verschiedene Whisper Modellen (und WER vs CHR wenn zeit)

Fehlertypen die man kategorisieren könnte (nur eine Idee):  
- Zahlen: "zwanzig" vs. "20"
- Dialekt-spezifische Wörter die falsch transkribiert werden
- Eigennamen / Ortsnamen (z.B. "Zürich" → "Zurich")
- Auslassungen (Wort komplett fehlt)
- Substitutionen (falsches Wort)
- Code-switching Fehler (Dialekt → Hochdeutsch)  

Mögliche andere Modelle zum Vergleich:
- Originale Whisper-Modelle (OpenAI)
  - openai/whisper-large-v1, v2, v3 — das sind eure Hauptkandidaten
- Whisper-Varianten von anderen Anbietern
  - Distil-Whisper (HuggingFace, 2023): komprimierte Version via Knowledge Distillation — 51% weniger Parameter, 5.8x schneller, aber nur ~1% schlechter WER Towards AI  
  - Faster-Whisper (SYSTRAN): technisch dasselbe Modell, aber optimierte Inferenz — kein anderes Training  
  - WhisperX (Uni Oxford): erweitert Whisper um word-level Timestamps und Speaker Diarization Towards AI

- Andere ASR-Modelle (nicht Whisper-basiert) zum Vergleich  
  - facebook/wav2vec2 (fine-tuned auf Deutsch)  
  - das von fhnw
  
Das STT4SG-eigene Baseline-Modell aus dem Paper — das wäre eigentlich der natürlichste Vergleichspunkt!



Daten: STT4SG-350 Korpus

Whisper auf Audioclips laufen lassen und Output mit Ground Truth vergleichen:  

```
Audio: [Jemand sagt "Guete Morge, wie gaats?"]      
Ground Truth: "Guten Morgen, wie geht es?"    
Whisper Output: "Guten Morgen, wie geht's dir?"     
```
→ WER = Anteil falsch erkannter Wörter  


## Project Structure

```
Project_2/
│
├── data/                        ← alle Datenfiles (lokal, nicht in Git)
│   ├── clips__test/             ← Audio Daten test
│   ├── clips__train_valid/      ← Audio Daten train
│   ├── test.csv
│   ├── train_all.tsv
│   ├── train_balanced.tsv
│   └── valid.tsv
│ 
├── .gitignore
└── README.md
```

