# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_5.3-Embedding_+_Faiss_Matching-yellow)

## 🎯 Objective

> This project implements a **scalable SKU matching pipeline** by combining **embeddings**, **FAISS similarity search**, **structured filtering**, and **LLM re-validation**.

> The pipeline matches **transaction SKUs** against a **master catalog** to handle messy product descriptions, pack-size variations, and partial matches.


**Version 5.3** introduces a major upgrade: it replaces rule-based/LLM-first matching with a **vector database + retrieval system** (FAISS). It embeds all master SKUs, performs top-k retrieval for each transaction SKU, applies structured filtering, and outputs formatted match reports. Finally, a local **Mistral LLM** can be used for validation and exact match accuracy scoring.

<br>

## 🆕 Key Features

- **Master Embedding Pipeline:** Converts master SKUs into dense embeddings using nomic-embed-text.
- **FAISS Vector Index:** Stores embeddings for fast similarity search.
- *Transaction Matching:* Retrieves top-k candidate matches per transaction row.
- **Pack Size Filtering:** Numeric weight check to refine matches when available.
- **Formatted Outputs:** Groups results, aggregates top matches, and outputs cleaner reports.
- **LLM Re-validation:** Uses Mistral (via Ollama) for semantic verification of candidate matches.
- **Multi-stage Results:** Raw matches → filtered top-3 matches → formatted outputs → LLM exact match accuracy.

<br>

## 🔁 Input & Output

- Input Files
    * `Data/DataCSV/master.csv` – Master SKU reference data.
    * `Demo/*.csv` – Transaction SKUs to be matched. 
- Output Files
    * `master_index.faiss` – FAISS index of master embeddings.
    * `metadata.json` – Metadata for each master itemcode.
    * `./output/matches_<file>.csv` – Raw match results for each transaction file.
    * `./FinalMatches/final_matches_<file>.csv` – Filtered top-3 matches per transaction row.
    * `./FinalMatches/FormattedOutput/<file>.csv` – Aggregated, cleaner formatted match results.
    * `Exact.csv` – Final summary of LLM-validated exact match accuracy.

<br>

## ⚙️ Project Structure

```sh
StockKeepingUnit/
├── Data/
│   └── Concatenated Data
│		├── transaction_excluded_jun23.csv      		# Concatenated Transaction with June 23 Excluded
│		├── transaction_excluded_nov22_jun23.csv        # Concatenated Transaction with Nov 22 & June 23 Excluded        
│		├── transaction.csv             				# Concatenated Transaction
│   └── DataCSV/									
│       └── transaction/
│		    ├── # Contains all Individual Transaction Sheets 
│		├── master.csv	
│   └── Dummies/
│		├── master_dummy.xlsx                           # Synthetic Master
│		├── transaction_dummy.xlsx                      # Synthetic Transaction
│	├── master.xlsx                                     # Original Master
│	└── transaction.xlsx                                # Original Transaction
├── Demo/                                               # Testing Multiple Files at Once
│   └── final_matches_transaction.csv
├── Evaluation/
│   └── # Contains the Results from Eval.py
├── FinalMatches/
│   ├── # Contains the Results from process_transaction.py (Transaction Rows are ReiIterated with 3 Suggestions)
│   └── FormattedOutput/
│       └── # Contains the Results from transaction_formatting.py (Every Transaction Row consists of just 3 ItemCodes)
├── MaybeNeededLater/
│   ├── discarded_responses.txt                         # Additional Responsed from LLM
│	├── gpt.py                                          # Checking for Exact Match using Mistral
│	└── legacy_version.py                               # The OG VERSION
├── output/
│   └── # Contains the pre-final results from process_transaction.py (Transaction Rows are ReiIterated with 10 Suggestions)
├── Synthetic/                                          # Final LLM Step
│   ├── 1.py
│   ├── 2.py
│   ├── 3.py
│   ├── Exact.py                                        # Contains LLM based metrics
│   ├── log.log
│   ├── master.csv                                      # Segregated Master from 1.py
│   └── transaction.csv                                 # Segregated Transaction from 1.py
├── temp/                                               # The MOST IMPORTANT FOLDER strong the Metadata and Faiss Index
│   ├── master_index.faiss
│   └── metadata.json   										
├── eval.py                                             # Accuracy Evaluation using ItemCodes
├── Evaluation Summary.csv                              # Evaluation Metrics Filewise
├── master.py                                           # Processes master, prepares the Metadata and Faiss Index
├── process_transaction.py                              # Processes transaction and returns 3 Suggestions  
└── transaction_formatting.py                           # Handles 3 re-iterations for the 3 transaction rows
```

<br>

## 🚀 How It Works
<br>

📌 `master.py`

- Reads `master.csv` and selects key columns.
- Generates embeddings for each master row using **Ollama embeddings** (nomic-embed-text).
- Stores embeddings in a **FAISS index**.
- Saves metadata (`itemcode` + **attributes**) to JSON for retrieval.
<br>

📌 `process_transaction.py`

- Loads FAISS index + master metadata.
- Reads transaction CSVs from `Demo/`.
- Embeds each transaction row and retrieves top-k matches
- Applies pack-size numeric filtering (e.g., weight/unit consistency).
- Outputs raw matches (`output/`) and filtered top-3 matches (`FinalMatches/`).
<br>

📌 `transaction_formatting.py`

- Reads filtered matches from `FinalMatches/`.
- Groups by transaction columns and aggregates candidate matches.
- Outputs cleaner reports in `FinalMatches/FormattedOutput/`.
<br>

📌 `3.py`: **LLM Validation** 

- Uses **Mistral LLM** (Ollama) to semantically verify candidate matches.
- Compares transaction rows with master metadata candidates
- Outputs a final `Exact.csv` with summary metrics (match %, exact code accuracy %).
<br>

<br>

## 🛠️ Setup Guide (Windows)

 1. **Install Ollama**
 - [Download Ollama](https://ollama.com/download) and install it. It runs a local server at `http://localhost:11434`.
 
 2. **Pull the Mistral Model**
 - Open PowerShell or CMD and run: *(download size 4GB)*
 - `ollama pull mistral`

 3. **Start the Mistral Server**
 - `ollama run mistral` or `ollama serve`
 - This will launch the model in a conversational loop. For API usage, Ollama already runs a - background API server at `http://localhost:11434`.
 - You can keep ollama run mistral running in one terminal, or simply rely on the background service launched by Ollama on Windows startup.
  
 4. **Install Python Dependencies**
 - Open CMD
 - Create a Virtual Environment _(recommended)_ and run:
 - `pip install pandas numpy faiss-cpu tqdm`

> Note: Mistral model size is ~4GB and requires enough RAM (~8GB+ recommended).

<br>

## ▶️ Run the Code

```sh
# Step 1: Build master FAISS index
python master.py

# Step 2: Match transaction SKUs
python process_transaction.py

# Step 3: Format results
python transaction_formatting.py

# Step 4: Run LLM re-validation
python 3.py
```

> ⚠️ Ensure Ollama is running in the background with both nomic-embed-text and mistral models available.

<br>

## ✅ Advantages of This Version
- Fast and **scalable retrieval** via FAISS.
- Handles messy **SKU descriptions** better than rule-only systems.
- Pack-size numeric checks reduce false positives.
- Structured outputs allow for post-processing and reporting.
- Works fully **offline** with Ollama.
- LLM adds a semantic **re-validation** step for higher precision.

<br>

## 🧠 Matching Logic Highlights
|Step| Logic Used|
|:-|:-|
|Master Embedding|	`nomic-embed-text` via Ollama|
|Retrieval| FAISS L2 Similarity|
|Pack Size|	Numeric extraction + filtering|
|Transaction → Master|	Top-k candidates (default k=10, final top-3)|
|Final Matching| Aggregation of matches per transaction row|
|LLM Validation| Mistral chooses best `matched_itemcode`|

<br>

## 📅 Changelog

|Version|Changes|
|:-|:-|
|**v1**|*Rule-based + fuzzy matching*|
|**v2**|*LLM-powered item name extraction using Mistral via Ollama*|
|**v3**|*Integrated LLM + Structured SKU Matching + Fallback Suggestion Engine*|
|**v5**|*Full embedding pipeline with FAISS retrieval, pack-size filtering, formatted outputs, and LLM re-validation*|

<!-- ## 📈Model Metrics

|||Count| Percentage|
|:-|:-|:-|:-|
|**True Positive (TP)**| Correct Match| 480| 46.24%|
|**True Negative (TN)**| Correct No Match| 228| 21.97%|
|**False Negative (FN)**| Incorrect No Match| 222| 21.39%|
|**False Positive (FP)**| Incorrect Match| 108| 10.40%|
|| **Total**| 1038| 100%| -->

<br>

## 📌 Notes
- FAISS similarity is **L2 distance**; lower distance = better match.
- Pack-size filtering only applies when numeric values exist in both transaction and master.
- `Exact.csv` summarizes validation performance, but LLM evaluation may vary.
- Matching pipeline is modular — each stage (`embedding → retrieval → filtering → formatting → validation`) can be run independently.