# HR Churn Prediction – Employee Attrition Analysis  
Datenanalyse von Mitarbeiterstammdaten, Performance, Abwesenheiten und HR-Events zur Vorhersage von Mitarbeiterabwanderung

## 🧩 Problemstellung

Ein mittelständisches Unternehmen hat in den letzten Jahren eine steigende Mitarbeiterfluktuation, insbesondere in bestimmten Abteilungen und Rollen. Kündigungen kommen oft überraschend, Wissen geht verloren und Nachbesetzungen dauern lange. HR und Fachbereiche haben das Gefühl, nur zu reagieren statt frühzeitig gegensteuern zu können.

Ziel der Analyse ist es:

- zentrale Einflussfaktoren für Mitarbeiterabwanderung sichtbar zu machen  
- ein Modell zu entwickeln, das die Churn-Wahrscheinlichkeit pro Mitarbeiter schätzt  
- Mitarbeitergruppen mit erhöhtem Risiko zu identifizieren  
- Muster in Performance, Abwesenheiten und HR-Events zu erkennen  
- eine datenbasierte Grundlage für gezielte Retention-Maßnahmen zu schaffen  

## 📦 Projektüberblick

Dieses Projekt bildet ein vollständiges, realitätsnahes HR‑Datenumfeld ab und zeigt, wie ein Data Analyst Mitarbeiterdaten strukturiert analysiert, Risikofaktoren identifiziert und ein Vorhersagemodell für Mitarbeiterabwanderung entwickelt.

Das Projekt umfasst:

- 500 Mitarbeiter  
- 2.000 Abwesenheiten  
- 1.500 Performance‑Reviews  
- 800 HR‑Events (z. B. Beförderungen, Abmahnungen, Rollenwechsel)  
- vollständige Feature‑Engineering‑Pipeline  
- automatisierte Modellbewertung  
- Churn‑Score zur Priorisierung gefährdeter Mitarbeiter  

Alle Daten sind synthetisch, aber realistisch modelliert.

## 📁 Projektstruktur

hr_churn_prediction/  
│  
├── main.py                           – Hauptpipeline  
│  
├── data/                             – CSV‑Daten  
│   └── raw/hr_data.csv               – Gesamter HR‑Datensatz (synthetisch)  
│  
├── plots/                            – Automatisch erzeugte Diagramme  
│   ├── eda/                          – EDA‑Visualisierungen  
│   ├── feature_engineering/          – Feature‑Engineering‑Reports  
│   └── model/                        – Modell‑Evaluationsplots  
│  
├── src/  
│   ├── __init__.py                   – Paketinitialisierung  
│   ├── data_loading.py               – Laden der Rohdaten  
│   ├── data_cleaning.py              – Bereinigung & Vorbereitung der Einzel‑CSV  
│   ├── data_preparation.py           – Zusammenführung & Aufbereitung des HR‑Gesamtdatensatzes  
│   ├── aggregation.py                – Datenaggregation auf Mitarbeiterebene  
│   ├── eda.py                        – Explorative Datenanalyse  
│   ├── feature_engineering.py        – Feature Engineering & Encoding  
│   ├── feature_engineering_visualization.py – Visualisierung der erzeugten Features  
│   ├── model_training.py             – Training des Balanced Random Forest  
│   └── churn_score.py                – Churn‑Score & Top‑Risikomitarbeiter  
│  
└── eda/  
    ├── data_understanding.py         – Struktur‑ & Plausibilitätschecks  
    └── explorative_analyse.py        – Vertiefende Analysen zu HR‑Mustern  

## 🗂️ Datensätze

Die HR-Daten liegen logisch getrennt nach Entitäten vor und werden in der Pipeline zu einem konsistenten Mitarbeiter-Datensatz zusammengeführt.

### employees.csv
- Mitarbeiter-ID  
- Name (synthetisch)  
- Abteilung  
- Rolle  
- Einstellungsdatum  
- Vertragsart (z. B. Vollzeit, Teilzeit, Befristet)  

### absences.csv
- Mitarbeiter-ID  
- Datum  
- Abwesenheitsart (z. B. Krankheit, Urlaub, Sonderurlaub)  
- Dauer in Tagen  

### performance_reviews.csv
- Mitarbeiter-ID  
- Bewertungsdatum  
- Performance-Rating (z. B. 1–5)  
- Kommentar (synthetisch)  

### hr_events.csv
- Mitarbeiter-ID  
- Event-Datum  
- Event-Typ (z. B. Beförderung, Abmahnung, Rollenwechsel, Gehaltserhöhung)  
- Beschreibung (synthetisch)  

## 📊 Analyse & Erkenntnisse

Bei der Analyse der HR‑Daten habe ich zunächst untersucht, wie sich Mitarbeiter über die letzten Jahre entwickelt haben. Dabei wurde schnell deutlich, dass bestimmte Muster stark mit erhöhter Fluktuation zusammenhängen.

Auffällig war insbesondere:

- Mitarbeiter mit häufigen oder langen Abwesenheiten zeigen ein erhöhtes Churn‑Risiko  
- Performance‑Einbrüche gehen oft mehreren HR‑Events voraus  
- bestimmte Abteilungen haben deutlich höhere Fluktuationsraten als andere  
- Rollen mit hoher Arbeitsbelastung weisen überdurchschnittlich viele Kündigungen auf  

Über das Feature Engineering und die Modellierung konnte ich diese Beobachtungen quantifizieren:

- Performance‑Rating ist einer der stärksten Prädiktoren  
- Abwesenheitstage korrelieren deutlich mit Churn‑Wahrscheinlichkeit  
- negative HR‑Events (z. B. Abmahnungen) erhöhen das Risiko signifikant  
- Beförderungen und Gehaltserhöhungen wirken stabilisierend  

Insgesamt zeigt die Analyse:

- Fluktuation ist kein Zufall, sondern folgt klaren Mustern  
- bestimmte Mitarbeitergruppen sind besonders gefährdet  
- HR‑Events und Performance‑Verläufe liefern wertvolle Frühindikatoren  
- datenbasierte Retention‑Maßnahmen könnten viele Kündigungen verhindern  

## 🔄 Analyse der HR‑Events

Die HR‑Events zeigen, welche wichtigen Personalereignisse Mitarbeiter im Laufe ihrer Unternehmenszugehörigkeit durchlaufen haben und wie diese Ereignisse mit der Fluktuation zusammenhängen.

Das Projekt berücksichtigt ein breites Spektrum an Events, darunter:

- Beförderungen  
- Rollenwechsel  
- Gehaltserhöhungen  
- Abmahnungen  
- Versetzungen  
- Onboarding‑/Offboarding‑Ereignisse  

Auffällig war:

- Mitarbeiter mit negativen Events (z. B. Abmahnungen) haben ein deutlich erhöhtes Churn‑Risiko  
- Beförderungen und Gehaltserhöhungen wirken stabilisierend und senken die Kündigungswahrscheinlichkeit  
- Rollenwechsel kurz vor einer Kündigung treten häufiger auf als erwartet  
- Einige Abteilungen zeigen besonders viele kritische HR‑Events innerhalb kurzer Zeiträume  

Daraus folgt:

- HR‑Events sind starke Frühindikatoren für Mitarbeiterabwanderung  
- Mitarbeiter mit mehreren kritischen Events sollten gezielt beobachtet und unterstützt werden  
- Positive Events (z. B. Beförderungen) können als Retention‑Instrument genutzt werden  
- Häufungen negativer Events in einzelnen Teams können auf Führungs‑ oder Prozessprobleme hinweisen  

## 📈 Berechnete KPIs

### Churn-Wahrscheinlichkeit  
Vom Modell berechnete Wahrscheinlichkeit, dass ein Mitarbeiter das Unternehmen innerhalb eines definierten Zeitraums verlässt.

### Risiko-Score  
Kombination aus Modellvorhersage und relevanten Einflussfaktoren (z. B. Performance, Abwesenheiten, HR‑Events).  
Dient zur Priorisierung gefährdeter Mitarbeiter.

### Performance-Trend  
Entwicklung der Mitarbeiterleistung über die Zeit.  
Negative Trends sind ein häufiger Frühindikator für steigende Fluktuation.

### Abwesenheitsindex  
Misst Häufigkeit und Dauer von Abwesenheiten.  
Ein hoher Wert korreliert oft mit erhöhtem Churn‑Risiko.

### Event-Intensität  
Anzahl und Art der HR‑Events pro Mitarbeiter.  
Negative Events erhöhen das Risiko, positive Events wirken stabilisierend.

## 🚨 Churn‑Score (USP des Projekts)

Der Churn‑Score kombiniert die Modellvorhersage mit zentralen Einflussfaktoren wie Performance‑Trend, Abwesenheitsindex und HR‑Events. Dadurch entsteht ein klarer, interpretierbarer Risikoindikator pro Mitarbeiter.

Churn‑Score = Modell‑Wahrscheinlichkeit × (1 + gewichtete Risikofaktoren)

Interpretation:

- hoher Score → kritisch  
- mittlerer Score → beobachten  
- niedriger Score → stabil  

Der Score ermöglicht es HR, gefährdete Mitarbeiter frühzeitig zu identifizieren und gezielt Maßnahmen einzuleiten.

## 📉 Visualisierungen

Automatisch erzeugte Diagramme:

- Top‑Mitarbeiter nach Churn‑Score  
- Mitarbeiter mit negativem Performance‑Trend  
- Abteilungen mit der höchsten Fluktuationsrate  
- Verteilung der Abwesenheitstage  
- Häufigste HR‑Events pro Mitarbeiter  
- Modell‑Evaluationsplots (ROC‑Curve, Confusion Matrix, Feature Importance)

Alle Diagramme werden im Ordner plots/ gespeichert.

## 🔧 Technische Umsetzung

### Hauptpipeline (main.py)
1. HR‑Daten laden und zusammenführen  
2. Struktur‑ und Plausibilitätschecks durchführen  
3. Feature Engineering & Encoding anwenden  
4. Balanced Random Forest trainieren  
5. Modell bewerten und Kennzahlen ausgeben  
6. Churn‑Score berechnen  
7. Top‑Risikomitarbeiter identifizieren  
8. Visualisierungen erzeugen und speichern  

### Technologien
- Python 3  
- pandas  
- seaborn  
- matplotlib  
- scikit-learn  
- imbalanced-learn  
- joblib  

## ▶️ Ausführung

Im Projektverzeichnis:

python main.py

## 🧠 Was dieses Projekt zeigt

- strukturierte Analyse von HR‑Daten  
- Verständnis zentraler Einflussfaktoren auf Mitarbeiterabwanderung  
- Entwicklung eines interpretierbaren Risiko‑Scores  
- Kombination aus EDA, Feature Engineering und Modellierung  
- saubere, modulare Code‑Struktur  
- datenbasiertes Storytelling für HR‑Entscheidungsträger  

## 👤 Autor

Jan‑Ivo Oelfke  
Fachkraft für Lagerlogistik auf dem Weg zum Data/Prozess Analyst