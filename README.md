# NLP-Project2 – Swiss German Dialects  

Team: 2 Personen  
Deadline Poster: 14.04.2026  
Postersession: 17.04.2026 

Abgabe: Wissenschaftliches Poster (A0, PDF, 1 Seite)  

## Forschungsfrage  
- Fehleranaylse mit verschiedene Whisper Modellen (und WER vs CHR wenn zeit)

  
- Whisper WER (Word Error Rate) pro Dialektregion
- Vertiefung: Alter & Geschlecht als Einflussfaktoren

Wie gut erkennt Whisper gesprochenes Schweizerdeutsch in den sieben Dialektregionen des STT4SG-350-Korpus, gemessen anhand der Word Error Rate (WER)? Als Vertiefung untersuchen wir, ob Alter und Geschlecht der Sprechenden die WER systematisch beeinflussen.

Vorgehen (kurz):
Whisper auf dem **Testset** des STT4SG-350 laufen lassen   
→ 1. WER pro Dialektregion berechnen   
→ 2. WER nach Altersgruppe und Geschlecht aufschlüsseln   
→ 3. Ergebnisse visualisieren und im Poster präsentieren.


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

