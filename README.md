# Temporal Aspect-Based Sentiment Analysis with RAG

This repository contains the implementation of a temporal aspect-based sentiment analysis system with retrieval-augmented generation capabilities for e-commerce review analysis. The project was developed as part of a Master's thesis in Data Science at Aalborg University (September-December 2025).

## 📋 Project Overview

The system analyzes Amazon Beauty product reviews to:
- Extract product aspects and classify sentiment polarities using transformer-based ABSA models
- Aggregate aspect-level sentiments into temporal trajectories with confidence weighting
- Validate predictions through multiple coherence metrics (Pearson correlation, coherence/divergence rates)
- Enable natural language querying via RAG architecture (FAISS + CrossEncoder + Gemini)

**Key Results:**
- 82.6% sentiment-rating coherence
- 0.785 Pearson correlation with star ratings
- 2.8s mean RAG latency
- 68,772 reviews analyzed across 16,337 products

## 🚀 Quick Start

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (optional, for faster inference)
- Google Gemini API key (for RAG functionality)

### Installation

1. **Clone the repository:**
```bash
git clone <repository-url>
cd project
```

2. **Install dependencies:**
```bash
pip install -r absa_dashboard/requirements.txt
```

3. **Download data:**
   - Place `All_Beauty.jsonl` in `data/` directory
   - Download from: [Amazon Reviews 2023](https://amazon-reviews-2023.github.io/)

4. **Set up environment variables:**
```bash
# For RAG functionality
export GOOGLE_API_KEY="your-gemini-api-key"
```

### Usage

#### 1. Run ABSA Inference

**Interactive (Jupyter):**
```bash
jupyter notebook inference.ipynb
```

**Batch Processing (Script):**
```bash
python run_inference.py --year 2020 --output absa_output/
```

#### 2. Launch Dashboard

```bash
cd absa_dashboard
streamlit run dashboard.py
```

Access the dashboard at `http://localhost:8501`

#### 3. Analyze Temporal Trajectories

```bash
jupyter notebook absa_trajectory_analysis_v2.ipynb
```

## 📁 Repository Structure

```
project/
├── README.md                          # This file
├── run_inference.py                   # Batch inference script
├── inference.ipynb                    # Interactive ABSA inference notebook
├── absa_trajectory_analysis_v2.ipynb  # Temporal trajectory analysis
├── getData_exploratory.ipynb         # Data exploration notebook
├── absa_model_comparison.ipynb       # Model evaluation notebook
│
├── data/                              # Dataset directory
│   ├── All_Beauty.jsonl              # Raw Amazon reviews
│   └── README.md                     # Dataset documentation
│
├── absa_output/                       # ABSA inference results (by month)
│   ├── Jul20/
│   ├── Aug20/
│   └── ...
│
├── absa_dashboard/                    # Interactive dashboard
│   ├── dashboard.py                  # Main Streamlit app
│   ├── config.py                     # Configuration
│   ├── preprocess_data.py            # Data preprocessing
│   ├── genai_client.py               # RAG implementation
│   ├── rag_agent.py                  # RAG agent logic
│   ├── requirements.txt              # Python dependencies
│   ├── README.md                     # Detailed dashboard docs
│   ├── data/                         # Preprocessed data (Parquet)
│   └── embeddings/                   # FAISS index
│
└── project_report/                    # LaTeX thesis
    ├── master.tex
    └── sections/
```

## 🔧 Technical Stack

**ABSA & NLP:**
- PyABSA 2.3.4 (ATEPC-LCF model)
- Transformers 4.35.0
- SentenceTransformers 2.2.2

**Vector Search & RAG:**
- FAISS 1.7.4
- Google Gemini 2.0/2.5 Flash
- CrossEncoder (ms-marco-MiniLM-L-6-v2)

**Data Processing:**
- Pandas 2.1.3
- NumPy 1.26.2
- PyArrow (Parquet)

**Visualization:**
- Streamlit 1.29.0
- Plotly 5.18.0

**ML Infrastructure:**
- PyTorch 2.1.1
- CUDA 11.8 (optional)

## 📊 Dataset

**Source:** Amazon Reviews 2023 - Beauty Category  
**Citation:** McAuley Lab, UCSD  
**Period:** September-December 2020  
**Size:** 68,772 reviews, 16,337 products  
**Verified Purchases:** 87%

See `data/README.md` for detailed dataset documentation.

## 🔬 Methodology

The project follows an engineering-oriented research methodology with five phases:

1. **Requirement Elicitation** - Translate research questions into testable requirements
2. **Requirement Specification** - Document functional/non-functional requirements (MoSCoW)
3. **Design** - Develop architectural solutions balancing accuracy, efficiency, cost
4. **Implementation** - Realize designs with concrete technologies
5. **Evaluation** - Validate against requirements through quantitative/qualitative metrics

**Research Questions:**
- **RQ1:** How can transformer-based models extract aspects and classify sentiments at scale?
- **RQ2:** How to aggregate aspect-level sentiments into temporal patterns?
- **RQ3:** How to validate predictions against ground truth?
- **RQ4:** How to enable natural language querying while maintaining factual grounding?

## 📈 Key Results

**ABSA Model Performance:**
- 82.6% coherence rate (sentiment aligns with ratings)
- 0.785 Pearson correlation (r ≥ 0.6 threshold met)
- 6.6% divergence rate (strong mismatches requiring inspection)

**Temporal Aggregation:**
- Successfully captured seasonal trends (Q4 holiday effects)
- Aspect normalization reduced lexical sparsity by 34%
- Confidence weighting improved trajectory smoothness

**RAG System:**
- 2.8s mean latency (sub-5s target met)
- 100% grounding rate (all answers cite sources)
- Operates within Gemini free tier (1M tokens/day)

## 🧪 Reproducibility

All code, data pipelines, and model configurations are version-controlled. To reproduce results:

1. Follow installation steps above
2. Run inference: `python run_inference.py --year 2020`
3. Execute evaluation notebooks in order:
   - `absa_model_comparison.ipynb` - Model validation
   - `absa_trajectory_analysis_v2.ipynb` - Temporal analysis
4. Launch dashboard: `cd absa_dashboard && streamlit run dashboard.py`

**Environment Specifications:**
- Python 3.9.18
- Ubuntu 22.04 / Windows 11
- CUDA 11.8 (optional)

See `absa_dashboard/requirements.txt` for complete dependency list with pinned versions.

## 📝 Citation

If you use this work, please cite:

```bibtex
@mastersthesis{mostafa2025temporal,
  title={Temporal Aspect-Based Sentiment Analysis with Retrieval-Augmented Generation for E-Commerce Reviews},
  author={Mostafa, Amira and Drira, Yosr},
  year={2025},
  school={Aalborg University},
  type={Master's Thesis}
}
```

## 📄 License

This project is developed for academic purposes as part of a Master's thesis at Aalborg University.

## 👥 Authors

- **Amira Mostafa** - amosta24@student.aau.dk
- **Yosr Drira** - ydrira25@student.aau.dk

**Supervisor:** [Supervisor Name], Aalborg University

## 🙏 Acknowledgments

- PyABSA team for the pretrained ATEPC-LCF model
- McAuley Lab for the Amazon Reviews 2023 dataset
- Industry expert from luxury retail for validation insights
- Open-source community (FAISS, Transformers, Streamlit)

## 📖 Additional Documentation

- **Dashboard Guide:** `absa_dashboard/README.md`
- **Dataset Details:** `data/README.md`
- **Full Thesis:** `project_report/master.pdf`
