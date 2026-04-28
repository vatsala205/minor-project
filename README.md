
# Explainable Mood Variability Assessment Framework (EMVAF)
An AI-powered system for early detection of mental health risk using mood drift modeling, explainable AI, and hybrid RAG-based reasoning.
---
## 📌 Overview
Traditional mental health models treat risk detection as a classification problem.  
This project challenges that approach.
Instead of asking:
> "Is the person at risk?"
We ask:
> "How is their emotional state drifting over time?"
This system models **mood as a continuous signal**, enabling **early intervention before crisis-level detection**.
---
## 🚀 Key Features
- 📉 **Mood Drift Modeling**
  - Tracks emotional change using embedding distance from personal baseline
  - Detects subtle early-stage signals missed by classifiers
- ⚙️ **Deterministic Risk Engine**
  - Weighted scoring system based on behavioral + physiological inputs
  - Fully traceable per-variable contribution
- 🌲 **Machine Learning Models**
  - Random Forest (structured data)
  - BERT-based classifier (text + semantic understanding)
- 🧠 **Hybrid RAG + LLM Explainability**
  - Retrieves similar cases using FAISS/Chroma
  - Generates human-readable explanations using LLM
- 🕸️ **Knowledge Graph (KAG)**
  - Visualizes relationships between factors like stress, sleep, and social interaction
- 📊 **Interactive Dashboard**
  - Built with Streamlit
  - Displays risk scores, explanations, and recommendations
---
## 🧠 Core Idea: Mood Drift
Instead of static classification, we measure:
- Current state embedding → `e_t`
- Baseline embedding → `e_base`
Drift = distance between them
This allows detection of:
- gradual stress buildup
- early burnout patterns
- subtle behavioral shifts
---
## 🏗️ System Architecture
The system is built as a **5-layer modular architecture**:
1. Data Processing Layer  
2. Risk Engine  
3. ML & Deep Learning Models  
4. RAG + LLM Explainability  
5. Knowledge Graph  
See architecture diagrams in `/graphs`.
---
## ⚙️ Tech Stack
- **Backend:** Python, FastAPI
- **ML:** Scikit-learn, Random Forest
- **NLP:** BERT (HuggingFace, PyTorch)
- **RAG:** FAISS / ChromaDB
- **LLM:** Local / API-based LLM
- **Visualization:** Streamlit, Plotly
- **Graph:** NetworkX, Pyvis
---
## 📊 Results
- Recall: **0.74**
- AUC: **0.94**
Outperformed:
- Logistic Regression (0.57)
- LSTM (0.62)
- BERT (0.66)
Early drift detection significantly improved performance.  [oai_citation:0‡mood(1)-2.pdf](sediment://file_00000000b3c07208a12a95b720b5f076)
---
## 📁 Project Structure

behavioral_health_ai/
│
├── dashboard/              # Streamlit UI
├── models/                 # ML + RAG models
├── graphs/                 # Architecture diagrams
├── utils/                  # Preprocessing
├── data/                   # Dataset
├── requirements.txt
└── README.md

---
## ▶️ How to Run
```bash
# 1. Clone repo
git clone https://github.com/vatsala205/minor-project.git
# 2. Go inside
cd behavioral_health_ai
# 3. Install dependencies
pip install -r requirements.txt
# 4. Run dashboard
streamlit run dashboard/app.py

```


🧩 Use Cases

* Early detection of burnout in students/professionals
* Mental health monitoring systems
* AI-assisted clinical decision support
* Research in explainable AI for healthcare



🔐 Privacy

* No PII stored
* In-memory processing
* Only anonymized embeddings used

📈 Future Work

* Time-series transformers (Informer / Autoformer)
* Bayesian adaptive thresholds
* Automated knowledge graph extraction
* Clinical-grade deployment



👥 Authors

* Vatsala Singh
* Priyanshu Bhardwaj

SRM Institute of Science and Technology
