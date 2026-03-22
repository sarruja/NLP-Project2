# NLP-Project2 – Swiss German Dialects  

Team: 2 Personen  
Deadline Poster: 14.04.2026  
Postersession: 17.04.2026 

Abgabe: Wissenschaftliches Poster (A0, PDF, 1 Seite)  

## Forschungsfrage  
- Whisper WER (Word Error Rate) pro Dialektregion
- VertiefungRQ1: Alter & Geschlecht als Einflussfaktoren

Daten: STT4SG-350 Korpus

Whisper auf Audioclips laufen lassen und Output mit Ground Truth vergleichen:  

```
Audio: [Jemand sagt "Guete Morge, wie gaats?"]      
Ground Truth: "Guten Morgen, wie geht es?"    
Whisper Output: "Guten Morgen, wie geht's dir?"     
```
→ WER = Anteil falsch erkannter Wörter  
