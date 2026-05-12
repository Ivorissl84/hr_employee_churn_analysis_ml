# HR Churn Prediction – Employee Attrition Analysis  
Technische Umsetzung einer vollständigen HR‑Datenpipeline zur quantitativen Analyse von Mitarbeiterfluktuation

Dieses Projekt bildet ein realitätsnahes HR‑Datenumfeld ab und zeigt, wie Mitarbeiterstammdaten, Abwesenheiten, Performance‑Reviews und HR‑Events zu einem konsistenten Datensatz zusammengeführt werden können, um **Fluktuation messbar zu machen**.  
Der Schwerpunkt liegt auf der **technischen Pipeline**, der Datenaufbereitung und der Berechnung einer **quantitativen Churn‑Rate**.

## 🧩 Problemstellung

Ein mittelständisches Unternehmen stellt fest, dass die Mitarbeiterfluktuation in den letzten Jahren gestiegen ist. Besonders betroffen sind einzelne Abteilungen und Rollen. Kündigungen kommen oft überraschend, was zu Wissensverlust, längeren Nachbesetzungszeiten und zusätzlicher Belastung für die Teams führt.

Dieses Projekt konzentriert sich auf die **quantitative Analyse der Fluktuation**.  
Ziel ist es, die HR‑Daten so aufzubereiten und zusammenzuführen, dass:

- die Anzahl der Abgänge pro Zeitraum messbar wird  
- Fluktuation strukturiert und reproduzierbar berechnet werden kann  
- ein konsistenter Datensatz entsteht, der zukünftige Analysen ermöglicht  
- ein technisches Fundament für spätere, tiefere Churn‑Analysen gelegt wird  

Der Fokus liegt **nicht** auf der Identifikation von Ursachen oder Einflussfaktoren, sondern auf der **technischen Pipeline**, die Fluktuation sichtbar und auswertbar macht.

## 📦 Projektüberblick

Dieses Projekt bildet ein vollständiges, realitätsnahes HR‑Datenumfeld ab und zeigt, wie Mitarbeiterstammdaten, Abwesenheiten, Performance‑Reviews und HR‑Events technisch zu einem konsistenten Datensatz zusammengeführt werden können.  
Der Fokus liegt auf der **Datenqualität**, der **Plausibilisierung** und der **quantitativen Berechnung der Fluktuation**.

Das Projekt umfasst:

- 500 Mitarbeiter  
- 2.000 Abwesenheiten  
- 1.500 Performance‑Reviews  
- 800 HR‑Events (z. B. Beförderungen, Abmahnungen, Rollenwechsel)  

Alle Daten sind synthetisch, aber realitätsnah modelliert und wurden hinsichtlich Struktur, Vollständigkeit und Logik verbessert.

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

## 🗂️ Datensätze

Die HR‑Daten liegen logisch getrennt nach Entitäten vor und werden in der Pipeline zu einem konsistenten Mitarbeiter‑Datensatz zusammengeführt.  
Jede Tabelle bildet einen klar abgegrenzten Teil des HR‑Prozesses ab:

- **employees.csv** – Stammdaten der Mitarbeiter  
- **absences.csv** – Krankmeldungen, Urlaub, sonstige Abwesenheiten  
- **performance_reviews.csv** – jährliche Leistungsbeurteilungen  
- **hr_events.csv** – HR‑Ereignisse wie Beförderungen, Rollenwechsel oder Abmahnungen  

Alle Datensätze sind synthetisch erzeugt, aber realitätsnah modelliert und wurden hinsichtlich Struktur, Vollständigkeit und Plausibilität überarbeitet.

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

Die Analyse konzentriert sich auf die quantitative Betrachtung der Mitarbeiterfluktuation.  
Ziel war es, die HR‑Daten so aufzubereiten, dass Abgänge strukturiert, reproduzierbar und über mehrere Jahre hinweg vergleichbar ausgewertet werden können.

Im Rahmen der Analyse wurden folgende Schritte durchgeführt:

- Zusammenführung der Stammdaten, Abwesenheiten, Performance‑Reviews und HR‑Events  
- Bereinigung und Plausibilisierung der Einzel‑Datensätze  
- Aggregation der Informationen auf Mitarbeiterebene  
- Berechnung der Fluktuationskennzahlen auf Basis des aggregierten Datensatzes

Die Ergebnisse zeigen:

- Die Fluktuation variiert deutlich zwischen Abteilungen  
- Bestimmte Rollen weisen überdurchschnittlich viele Abgänge auf  

Der Fokus liegt ausschließlich auf der **quantitativen Messung** der Fluktuation.  
Eine Analyse von Ursachen oder Risikofaktoren findet bewusst nicht statt.

## 🔄 HR‑Events im Datensatz

Der Datensatz enthält ein breites Spektrum an HR‑Events, die wichtige Stationen im Mitarbeiterlebenszyklus abbilden, darunter:

- Beförderungen  
- Rollenwechsel  
- Gehaltserhöhungen  
- Abmahnungen  
- Versetzungen  
- Onboarding‑/Offboarding‑Ereignisse  

Diese Informationen stehen als strukturierte Datenbasis zur Verfügung und ermöglichen grundsätzlich weiterführende Analysen, z. B. Zusammenhänge zwischen bestimmten Events und der Fluktuation.

Im Rahmen dieses Projekts werden die HR‑Events jedoch **nicht separat ausgewertet**. Sie dienen primär als Kontextinformationen innerhalb des Gesamtdatensatzes, ohne dass einzelne Eventtypen explizit visualisiert oder modelliert werden.

## 📈 Berechnete KPIs

Die Pipeline liefert drei Arten von Kennzahlen:  
(1) deskriptive HR‑KPIs,  
(2) Modellmetriken,  
(3) Risikoscores.

### 🧩 1. Deskriptive HR‑KPIs

Diese Kennzahlen stammen aus der aggregierten Mitarbeiterbasis:

- **Churn‑Verteilung** (Anteil der Mitarbeitenden, die das Unternehmen verlassen haben)  
- **Verteilung der Unternehmenszugehörigkeit**  
- **Churn nach Abteilung**  
- **Churn nach Job‑Level**  
- **Statistische Grundkennzahlen** (Alter, Gehalt, Performance, Abwesenheiten)

Diese KPIs bilden die Grundlage für die spätere Modellierung.

### 🤖 2. Modell‑KPIs

Für die Vorhersage der Kündigungswahrscheinlichkeit wird ein **Random Forest Classifier** eingesetzt.

Wesentliche Metriken:

- **Accuracy:** 0.73  
- **Precision:** 0.80  
- **Recall:** 0.36  
- **F1‑Score:** 0.50  

Die wichtigsten Einflussfaktoren laut Modell:

- years_at_company  
- performance_mean  
- job_level  
- warnings  
- sick_days  

### 🚨 3. Churn‑Scoring

Auf Basis der Modellwahrscheinlichkeit wird ein **Churn‑Score** berechnet, der Mitarbeitende in drei Risikogruppen einteilt:

- **high risk**  
- **medium risk**  
- **low risk**

Dieser Score ermöglicht eine priorisierte Betrachtung potenziell gefährdeter Mitarbeitender.

## 📉 Visualisierungen

Die Pipeline erzeugt automatisch mehrere Diagramme, die in drei Kategorien organisiert sind:

### 🧩 Explorative Datenanalyse (EDA)  
Gespeichert unter: `plots/eda/`

- **correlation_matrix.png**  
  Visualisiert Zusammenhänge zwischen numerischen Variablen.

- **distribution_years_at_company.png**  
  Histogramm der Unternehmenszugehörigkeit.

- **churn_by_department.png**  
  Vergleich der Kündigungsraten zwischen Abteilungen.

- **churn_by_job_level.png**  
  Analyse der Fluktuation über verschiedene Job-Level.

### 🛠 Feature Engineering  
Gespeichert unter: `plots/feature_engineering/`

- **feature_type_counts.png**  
  Übersicht über numerische, kategorische und One‑Hot‑Features.

### 🤖 Modellbewertung  
Gespeichert unter: `plots/model/`

- **confusion_matrix.png**  
  Zeigt True/False Positives und Negatives des Modells.

- **feature_stability.png**  
  Vergleich von Gini‑Importance und Permutation‑Importance zur Bewertung der Modellrobustheit.

Alle Diagramme werden automatisch im jeweiligen Unterordner des Verzeichnisses `plots/` gespeichert.

## 🔧 Technische Umsetzung

### Hauptpipeline (main.py)

Die Pipeline ist modular aufgebaut und führt die folgenden Schritte automatisiert aus:

1. **HR‑Eventdaten laden und auf Mitarbeiterebene aggregieren**  
   (inkl. Berechnung von Tenure, Performance‑Trend, Abwesenheiten, Warnungen usw.)

2. **Struktur‑ und Plausibilitätschecks**  
   Prüfung auf fehlende Werte, Datentypen und erwartete Spalten.

3. **Explorative Datenanalyse (EDA)**  
   Erzeugt mehrere Visualisierungen zu Churn‑Mustern und Datenverteilungen.

4. **Feature Engineering & One‑Hot‑Encoding**  
   Transformation numerischer und kategorischer Variablen, Oversampling der Minderheitsklasse.

5. **Training eines Random Forest Classifiers**  
   Modellierung der Kündigungswahrscheinlichkeit.

6. **Modellbewertung**  
   Ausgabe von Accuracy, Precision, Recall, F1‑Score sowie Confusion Matrix und Feature‑Stabilität.

7. **Churn‑Scoring**  
   Berechnung eines kombinierten Risiko‑Scores pro Mitarbeiter und Einteilung in Risiko‑Kategorien.

8. **Erzeugung und Speicherung aller Visualisierungen**  
   Ablage in den Ordnern `plots/eda/`, `plots/feature_engineering/` und `plots/model/`.

### Technologien

- **Python 3**  
- **pandas** – Datenverarbeitung  
- **numpy** – numerische Berechnungen  
- **matplotlib & seaborn** – Visualisierungen  
- **scikit‑learn** – Modelltraining & Evaluation  
- **imbalanced‑learn** – Oversampling (SMOTE/RandomOverSampler)  
- **joblib** – Speichern und Laden des Modells  

## ▶️ Ausführung

Im Projektverzeichnis:

python main.py

## 🧠 Was dieses Projekt zeigt

- strukturierte und reproduzierbare Analyse eines komplexen HR‑Datensatzes  
- Identifikation relevanter Einflussfaktoren auf Mitarbeiterabwanderung  
- Entwicklung eines interpretierbaren Churn‑Scores zur Risikobewertung  
- Kombination aus EDA, Feature Engineering und praxistauglicher Modellierung  
- klare, modulare und nachvollziehbare Code‑Struktur  
- datenbasiertes Storytelling für HR‑Entscheidungsträger  

## 👤 Autor

Jan‑Ivo Oelfke  
Fachkraft für Lagerlogistik auf dem Weg zum Data/Prozess Analyst