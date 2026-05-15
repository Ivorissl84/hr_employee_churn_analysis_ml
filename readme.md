# Mitarbeiter‑Churn Analyse  
Machine‑Learning Modell zur Vorhersage von Mitarbeiterabwanderung und Risikofaktoren

## 🧩 Problemstellung

Das Unternehmen hat Schwierigkeiten, Mitarbeiter langfristig zu binden. Die Fluktuation ist in den letzten Jahren deutlich gestiegen, was zu höheren Kosten, Wissensverlust und organisatorischen Engpässen führt. HR und Führungskräfte reagieren häufig erst, wenn Kündigungen bereits ausgesprochen sind, da ein systematisches Verständnis der Risikofaktoren fehlt.

Ziel der Analyse ist es:

- Transparenz über Muster und Einflussfaktoren der Mitarbeiterabwanderung zu schaffen  
- ein Machine‑Learning‑Modell zu entwickeln, das gefährdete Mitarbeiter frühzeitig erkennt  
- datenbasierte Entscheidungsgrundlagen für HR‑Maßnahmen zu liefern  
- die Grundlage für eine nachhaltige Mitarbeiterbindungsstrategie zu schaffen  

## 📦 Projektüberblick  

Dieses Projekt bildet ein vollständiges HR‑Analytics‑Szenario ab und zeigt, wie ein Data Analyst Mitarbeiterabwanderung untersucht, Risikofaktoren identifiziert und ein Machine‑Learning‑Modell zur Vorhersage von Churn entwickelt.

Der Datensatz umfasst zentrale HR‑Variablen wie:

- Mitarbeiterstammdaten  
- Abteilung & Job Level  
- Gehalt  
- Performance  
- Krankheitstage  
- Betriebszugehörigkeit (years_at_company)  
- Kündigungsstatus (left_company)  

Alle Daten sind synthetisch und wurden so realistisch wie möglich modelliert, ohne den Anspruch zu erheben, echte Unternehmensstrukturen oder reale Abwanderungsmuster exakt abzubilden.

Das Projekt zeigt:

- eine vollständige ML‑Pipeline zur Churn‑Vorhersage  
- datengetriebene Analyse von Risikofaktoren  
- Feature Engineering & Modellinterpretation  
- einen Churn‑Score zur Priorisierung gefährdeter Mitarbeiter  
- automatische Visualisierungen und Ergebnisaufbereitung  

Der Fokus liegt darauf, Muster der Mitarbeiterabwanderung sichtbar zu machen und einen praktischen Lösungsansatz für datenbasierte HR‑Entscheidungen zu entwickeln.

## 📁 Projektstruktur

hr_employee_churn_analysis_ml/  
│  
├── main.py                     – Hauptpipeline zur Ausführung der Analyse  
│  
├── data/                       – Daten und Modelle  
│   ├── raw/                    – Rohdaten (synthetisch modellierte HR‑Daten)  
│   │   ├── employees.csv  
│   │   ├── absences.csv  
│   │   ├── hr_events.csv  
│   │   └── performance_reviews.csv  
│   │  
│   ├── model/                  – gespeichertes Machine‑Learning‑Modell  
│   │   └── random_forest_model.pkl  
│   │  
│   └── plots/                  – automatisch erzeugte Diagramme  
│       ├── aggregation/        – Abteilungs‑ und Tenure‑Analysen  
│       ├── feature_engineering/– Feature‑Importances & Typen  
│       └── model/              – Modellbewertung (ROC, Precision‑Recall, Confusion Matrix)  
│  
├── src/                        – Modulcode für Datenverarbeitung & Analyse  
│   ├── data_loading.py  
│   ├── data_cleaning.py  
│   ├── data_preparation.py  
│   ├── feature_engineering.py  
│   ├── feature_engineering_visualization.py  
│   ├── model_training.py  
│   ├── churn_score.py  
│   ├── aggregation.py  
│   └── eda.py  
│  
├── tools/                      – Hilfsfunktionen & Zusatzskripte  
│  
└── readme.md                   – Projektdokumentation  

## 🗂️ Datensätze

Das Projekt verwendet vier Rohdatensätze, die zu einem konsolidierten HR‑Datensatz zusammengeführt werden und gemeinsam ein vollständiges HR‑Analytics‑Szenario abbilden.

### Rohdaten
- **employees.csv** – Stammdaten der Mitarbeiter mit Abteilung, Job Level, Gehalt, Betriebszugehörigkeit und Kündigungsstatus  
- **absences.csv** – Informationen zu Krankheitstagen und Abwesenheiten  
- **hr_events.csv** – HR‑Ereignisse wie Trainings, Verwarnungen und Beförderungen  
- **performance_reviews.csv** – Leistungsbewertungen und Performance‑Scores  

### Aggregierter Datensatz
- **hr_data.csv** – Zusammenführung und Bereinigung der vier Rohdatensätze; dient als Hauptquelle für Analyse und Modelltraining  

### Modell‑Datei
- **random_forest_model.pkl** – gespeichertes Machine‑Learning‑Modell zur Vorhersage der Mitarbeiterabwanderung  

Alle Datensätze sind synthetisch und wurden so realistisch wie möglich modelliert, ohne den Anspruch zu erheben, echte Unternehmensstrukturen oder reale HR‑Muster exakt abzubilden.

## 📊 Analyse & Erkenntnisse

### 🔹 Abteilungsanalyse
Die Auswertung zeigt deutliche Unterschiede zwischen den Unternehmensbereichen. Besonders betroffen ist die **IT‑Abteilung**, die über alle Joblevel hinweg die höchsten Abwanderungsraten aufweist. Auch **Sales** und **Logistics** verzeichnen überdurchschnittliche Werte, während **Production** einzelne Ausreißer in höheren Joblevels zeigt.  
Im Gegensatz dazu ist **Procurement** die stabilste Abteilung mit der geringsten Fluktuation. Eine Besonderheit stellt der **HR‑Bereich** dar, der im niedrigsten Joblevel eine auffällige Spitze zeigt.

Diese Muster deuten auf strukturelle Unterschiede in Arbeitsbelastung, Vergütung und Entwicklungsmöglichkeiten zwischen den Abteilungen hin.

![Churn Rate by Department × Job Level](data/plots/aggregation/churn_department_joblevel.png)

---

### 🔹 Feature‑Analyse
Das Machine‑Learning‑Modell identifiziert folgende Hauptfaktoren für Mitarbeiterabwanderung:

| Rang | Feature | Bedeutung (Gini Importance) | Interpretation |
|------|----------|-----------------------------|----------------|
| 1 | performance_mean | 0.15 | Durchschnittliche Performance korreliert stark mit Verbleib oder Abwanderung |
| 2 | years_at_company | 0.14 | Längere Betriebszugehörigkeit erhöht das Risiko, das Unternehmen zu verlassen |
| 3 | salary | 0.13 | Niedrigere Vergütung steht in Zusammenhang mit höherer Fluktuation |
| 4 | age | 0.11 | Jüngere Mitarbeiter wechseln häufiger |
| 5 | sick_days | 0.10 | Hohe Krankheitsquote deutet auf Belastung oder Unzufriedenheit hin |
| 6–10 | job_level, trainings, warnings, performance_trend, gender_m | < 0.08 | Ergänzende Einflussfaktoren mit geringerer Gewichtung |

![Top 10 Feature Importances (Gini)](data/plots/feature_engineering/feature_importance_top10.png)

---

### 🔹 Zusammenspiel der Faktoren
Die Analyse zeigt, dass **mehrere Einflussgrößen ineinandergreifen**:

- **Vergütung** wirkt als indirekte Form der Wertschätzung.  
- **Fehlende Entwicklungsmöglichkeiten** verstärken Unzufriedenheit.  
- **Steigende Krankheitstage** spiegeln Belastung und Frustration wider.  
- **Längere Betriebszugehörigkeit** führt bei fehlender Perspektive zu erhöhter Wechselbereitschaft.

Diese Kombination erklärt, warum Mitarbeiter mit mehreren dieser Merkmale ein deutlich höheres Abwanderungsrisiko aufweisen.

---

### 🔹 Modellbewertung
Das eingesetzte **Random‑Forest‑Modell** liefert eine stabile Performance mit guter Balance zwischen **Precision** und **Recall**. Die **ROC‑Kurve** zeigt eine solide Trennschärfe, und die Feature‑Importances sind konsistent über verschiedene Trainingsläufe hinweg.  
Das Modell eignet sich damit als zuverlässige Grundlage für eine datenbasierte Risikobewertung und Frühwarnsysteme im HR‑Bereich.

---

### 🔹 Fazit
Die Analyse verdeutlicht, dass Mitarbeiterabwanderung kein isoliertes Phänomen ist, sondern das Ergebnis eines komplexen Zusammenspiels aus Vergütung, Entwicklung, Belastung und Zugehörigkeit.  
Das Projekt liefert damit einen praxisnahen Ansatz, um gefährdete Mitarbeitergruppen frühzeitig zu identifizieren und gezielte Gegenmaßnahmen zu entwickeln.

## 🛠️ Lösungsansatz & Empfehlungen

Ziel der Maßnahmen ist es, die Mitarbeiterbindung zu erhöhen und die identifizierten Risikobereiche gezielt zu adressieren. Die Empfehlungen basieren direkt auf den Analyseergebnissen und den Mustern, die das Machine‑Learning‑Modell sichtbar gemacht hat.

---

### 🔹 Abteilungsspezifische Maßnahmen

#### **IT**
Die IT‑Abteilung weist über alle Joblevel hinweg die höchste Abwanderungsrate auf. Die Kombination aus hoher Arbeitsbelastung, vergleichsweise niedriger Vergütung und wenigen Entlastungsmechanismen deutet auf strukturelle Überlastung hin.

**Empfehlungen:**
- Arbeitsbelastung reduzieren (z. B. durch Neueinstellungen oder Umverteilung von Aufgaben)  
- Gehaltsstruktur überprüfen und marktgerecht anpassen  
- Entwicklungsmöglichkeiten und interne Mobilität stärken  

---

#### **Sales**
In Sales zeigt sich ein zweigeteiltes Muster:  
- **Joblevel 1–2:** hohe Unzufriedenheit, vermutlich aufgrund niedriger Boni  
- **Joblevel 3:** stabilste Gruppe  
- **Joblevel 4–5:** erfahrene Mitarbeiter mit steigenden Erwartungen an variable Vergütung

**Empfehlungen:**
- Bonusmodell für neue Mitarbeiter anpassen, um frühe Frustration zu vermeiden  
- Klare Zielstrukturen und realistische Ramp‑Up‑Phasen definieren  
- Für erfahrene Mitarbeiter: variable Vergütung überprüfen und transparenter gestalten  

---

#### **Logistics**
Die Logistik zeigt in den typischen Jobleveln (2–3) durchgehend hohe Abwanderungsraten. Da höhere Joblevel in diesem Bereich selten sind, deutet dies auf generelle Unzufriedenheit hin.

**Empfehlungen:**
- Grundvergütung erhöhen oder Zusatzleistungen einführen  
- Arbeitsbelastung durch Neueinstellungen oder Prozessoptimierung reduzieren  
- Perspektiven für Weiterentwicklung schaffen, auch ohne klassische Karriereleiter  

---

#### **Production**
Einzelne Ausreißer in höheren Jobleveln deuten auf individuelle Belastungssituationen oder unklare Rollenanforderungen hin.

**Empfehlungen:**
- Rollen und Verantwortlichkeiten überprüfen  
- Führungskräfte in diesen Bereichen gezielt einbinden  
- Belastungsspitzen identifizieren und abfedern  

---

#### **HR (Joblevel 1)**
Der Ausreißer im niedrigsten Joblevel lässt auf eine deutliche Unterbezahlung im Vergleich zu Kollegen schließen.

**Empfehlungen:**
- Gehaltsstruktur prüfen und Ungleichheiten beseitigen  
- Klare Entwicklungspfade für Einsteiger schaffen  
- Feedback‑ und Mentoring‑Strukturen stärken  

---

#### **Procurement**
Procurement zeigt die geringste Fluktuation und wirkt stabil.

**Empfehlungen:**
- Erfolgsfaktoren dieser Abteilung identifizieren und auf andere Bereiche übertragen  
- Kultur‑ und Prozessmerkmale dokumentieren  

---

### 🔹 Maßnahmen auf Mitarbeiterebene (Churn‑Score)

Das Modell ermöglicht eine frühzeitige Identifikation gefährdeter Mitarbeiter.  
Besonders relevant sind Kombinationen aus:

- niedriger Vergütung  
- steigenden Krankheitstagen  
- stagnierender Performance  
- langer Betriebszugehörigkeit  
- wenigen Entwicklungsmöglichkeiten  

**Empfehlungen:**
- Individuelle Entwicklungsgespräche priorisieren  
- Belastungssituationen früh erkennen  
- Gezielte Weiterbildungsangebote und interne Wechseloptionen anbieten  

---

### 🔹 Maßnahmen auf Unternehmensebene

Die Analyse zeigt, dass Abwanderung kein isoliertes Problem einzelner Mitarbeiter ist, sondern strukturelle Ursachen hat.

**Empfehlungen:**
- Vergütungsstrukturen überprüfen und vereinheitlichen  
- Transparente Karrierepfade definieren  
- Belastungsmanagement verbessern (z. B. durch Personalaufstockung oder Prozessoptimierung)  
- Führungskräfte stärker in Bindungsmaßnahmen einbinden  
- Datenbasierte HR‑Steuerung etablieren (regelmäßige Churn‑Analysen, Monitoring, Frühwarnsysteme)  

---

### 🔹 Rolle des Machine‑Learning‑Modells

Das Modell dient als Frühwarnsystem und unterstützt HR dabei:

- gefährdete Mitarbeitergruppen frühzeitig zu erkennen  
- Maßnahmen gezielt zu priorisieren  
- strukturelle Probleme sichtbar zu machen  
- Entscheidungen datenbasiert zu treffen  

Es ersetzt keine HR‑Expertise, sondern erweitert sie um analytische Tiefe und Objektivität.

## ⚙️ Technischer Ansatz (ML‑Pipeline)

Das Projekt folgt einer modular aufgebauten Machine‑Learning‑Pipeline, die vollständig in Python implementiert ist. Alle Schritte sind in klar getrennten Modulen organisiert und werden über `main.py` orchestriert.

---

### 🔹 1. Datenimport & Bereinigung
**Module:** `data_loading.py`, `data_cleaning.py`, `data_preparation.py`

- Einlesen der vier Rohdatensätze  
- Vereinheitlichung von Formaten (Datumsangaben, Kategorien, IDs)  
- Bereinigung fehlerhafter oder unvollständiger Einträge  
- Zusammenführung zu einem konsolidierten Datensatz (`hr_data.csv`)  
- Ableitung zusätzlicher Variablen (z. B. Tenure, Performance‑Trends)

---

### 🔹 2. Explorative Analyse (EDA)
**Module:** `eda.py`, `aggregation.py`

- Analyse von Abteilungen, Jobleveln und Fluktuationsmustern  
- Identifikation besonders betroffener Bereiche (z. B. IT, Sales, Logistics)  
- Visualisierung zentraler Muster (Churn‑Rate nach Abteilung × Joblevel)

---

### 🔹 3. Feature Engineering
**Module:** `feature_engineering.py`, `feature_engineering_visualization.py`

- Erstellung numerischer und kategorialer Features  
- Encoding von Kategorien  
- Skalierung und Transformation  
- Analyse der Feature‑Wichtigkeit (Gini‑Importances)  
- Visualisierung der Top‑Features

---

### 🔹 4. Modelltraining
**Module:** `model_training.py`

- Training eines Random‑Forest‑Klassifikationsmodells  
- Hyperparameter‑Tuning  
- Cross‑Validation  
- Speicherung des finalen Modells (`random_forest_model.pkl`)  
- Evaluierung mittels Accuracy, Precision, Recall, F1‑Score und ROC‑AUC

---

### 🔹 5. Churn‑Score & Risikobewertung
**Module:** `churn_score.py`

- Berechnung eines individuellen Churn‑Scores für jeden Mitarbeiter  
- Einordnung in Risikogruppen  
- Ableitung von Handlungsempfehlungen basierend auf Modell‑Insights

---

### 🔹 6. Automatisierte Visualisierung
**Module:** `feature_engineering_visualization.py`, `aggregation.py`

- Erstellung aller Diagramme im Ordner `data/plots/`  
- Strukturierte Ablage nach Themen (Aggregation, Feature Engineering, Modell)

---

### 🔹 7. Pipeline‑Steuerung
**Datei:** `main.py`

- Ausführung aller Schritte in definierter Reihenfolge  
- Reproduzierbare Analyse durch modularen Aufbau  
- Klare Trennung von Logik, Daten und Visualisierungen

---

Die Pipeline ist so aufgebaut, dass sie leicht erweiterbar ist — z. B. für zusätzliche Features, alternative Modelle oder neue HR‑Datenquellen.

## ▶️ Ausführung

Im Projektverzeichnis:

\`\`\`bash
python main.py
\`\`\`

Die Pipeline führt alle Schritte automatisch aus:

- Daten laden & bereinigen  
- Feature Engineering  
- Modelltraining  
- Churn‑Score‑Berechnung  
- Erstellung aller Visualisierungen im Ordner `data/plots/`  

Alle Ergebnisse werden reproduzierbar generiert und in der Projektstruktur abgelegt.
## ⚠️ Projektgrenzen & Realismus

Das Projekt basiert auf synthetischen HR‑Daten, die realistische Muster abbilden, jedoch nicht die Komplexität echter Unternehmensdaten vollständig widerspiegeln. Die Ergebnisse sind daher als analytische Demonstration zu verstehen, nicht als Bewertung eines realen Unternehmens.

Wesentliche Einschränkungen:

- **Synthetische Daten:** Trotz realistischer Struktur können einzelne Zusammenhänge stärker oder schwächer ausgeprägt sein als in echten HR‑Systemen.
- **Begrenzte Variablen:** Einige wichtige HR‑Faktoren (z. B. Teamdynamik, Führungskultur, externe Marktbedingungen) sind nicht enthalten.
- **Modellvereinfachung:** Das Random‑Forest‑Modell liefert robuste Ergebnisse, ersetzt jedoch keine umfassende HR‑Diagnose.
- **Keine kausalen Aussagen:** Die Analyse zeigt statistische Zusammenhänge, aber keine direkten Ursachen.

Trotz dieser Grenzen ermöglicht das Projekt eine praxisnahe Demonstration, wie datengetriebene HR‑Analysen aufgebaut werden können und welche Mehrwerte ein strukturiertes Machine‑Learning‑Modell im HR‑Kontext bietet.

## 🔭 Ausblick / mögliche Erweiterungen

Das Projekt bildet eine vollständige End‑to‑End‑Analyse ab, lässt sich jedoch sinnvoll erweitern, um zusätzliche Realitätsnähe und analytische Tiefe zu gewinnen.

**Mögliche Weiterentwicklungen:**

- **Weitere Modelle testen:** Vergleich mit Gradient Boosting, XGBoost oder Logit‑Modellen zur Validierung der Modellrobustheit.
- **Zeitliche Komponenten integrieren:** Analyse von Trends über mehrere Jahre (z. B. Performance‑Verlauf, Krankheitsentwicklung, Gehaltsentwicklung).
- **Erweiterte Feature‑Sets:** Einbindung zusätzlicher HR‑Variablen wie Teamzugehörigkeit, Führungsspanne, Projektlast oder interne Wechselhistorie.
- **Explainability‑Methoden:** Einsatz von SHAP‑Werten zur noch präziseren Interpretation einzelner Vorhersagen.
- **Dashboard‑Integration:** Aufbau eines interaktiven Dashboards (z. B. Streamlit), um Churn‑Risiken und Abteilungsanalysen dynamisch darzustellen.
- **Automatisiertes Monitoring:** Regelmäßige Neuberechnung des Churn‑Scores und Überwachung von Modell‑Drift bei neuen Daten.

Diese Erweiterungen würden das Projekt näher an reale HR‑Analytics‑Systeme heranführen und die Einsatzmöglichkeiten im Unternehmenskontext weiter erhöhen.

## 👤 Autor

Jan‑Ivo Oelfke  
Fachkraft für Lagerlogistik (15+ Jahre Erfahrung)
Auf dem Weg zum Data & Process Analyst