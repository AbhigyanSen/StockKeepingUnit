# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_6-Categorical_Matching-white)

## 🎯 Objective

> This project implements a **scalable SKU matching pipeline** by combining **embeddings**, **FAISS similarity search**, **structured filtering**, and **LLM re-validation**.

> The pipeline matches **transaction SKUs** against a **master catalog** to handle messy product descriptions, pack-size variations, and partial matches.


**Version 6.0** introduces a major upgrade: instead of a single FAISS index, the system builds **per-category indexes** and stores **nested** **metadata** *(catcode → itemcode → attributes)* for better efficiency and more accurate retrieval. The pipeline retrieves candidates per transaction row, applies structured filtering, and generates formatted match reports along with **evaluation metrics**.

<br>

## 🔑 Key Features

- **Master Embedding Pipeline:** Converts master SKUs into dense embeddings using nomic-embed-text.
- **FAISS Vector Index:** Stores embeddings for fast similarity search.
- **Transaction Matching:** Retrieves top-k candidate matches per transaction row.
- **Pack Size Filtering:** Numeric weight check to refine matches when available.
- **Formatted Outputs:** Groups results, aggregates top matches, and outputs cleaner reports.
- **LLM Re-validation:** Uses Mistral (via Ollama) for semantic verification of candidate matches.
- **Multi-stage Results:** Raw matches → filtered top-3 matches → formatted outputs → LLM exact match accuracy

<br>

## 🆕 New Features

- **Category-Specific Embedding Pipeline:** Embeds master SKUs and stores them in per-catcode FAISS indexes.
- **Nested Metadata Store:** Organized as catcode → itemcode → attributes for fast lookup.
- **Transaction Matching:** Selects category-specific index for each transaction row (with fallback to all categories if needed).
- **Pack Size Filtering:** Refines top-k candidates by comparing transaction vs. master weights.
- **Formatted Outputs:** Aggregates candidate matches into cleaner CSV reports.
- **Evaluation Metrics:** Automatically computes Top-1/Top-3 accuracy, precision, recall, and error rates for the formatted outputs.

<br>

## 🔁 Input & Output

- Input Files
    * `Data/DataCSV/master.csv` – Master SKU reference data.
    * `Demo/*.csv` – Transaction SKUs to be matched. 
- Output Files
    * `./datastore/cat_indexes/index_<catcode>.faiss` – FAISS indexes for each category.
    * `./datastore/metadata.json` – Nested metadata for all SKUs grouped by category.
    * `./output/matches_<file>.csv` – Raw match results for each transaction file.
    * `./FinalMatches/final_matches_<file>.csv` – Filtered top-3 matches per transaction row.
    * `./FinalMatches/FormattedOutput/<file>.csv` – Aggregated, cleaner formatted match results.
    <!-- * `Exact.csv` – Final summary of LLM-validated exact match accuracy. -->
    * `./TransactionFormatting/Evaluation_metrics.csv` – Evaluation results across all formatted outputs.

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
- Groups by transaction row and collects top-3 candidate codes.
- Saves cleaned reports in `TransactionFormatting/FormattedOutput/`.
- Computes evaluation metrics (`Top-1`, `Top-2`, `Top-3` accuracy, precision, recall, Type I/II errors).
- Appends results to `TransactionFormatting/Evaluation_metrics.csv`.

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
# python 3.py
```

> ⚠️ Ensure Ollama is running in the background with both nomic-embed-text and mistral models available.

<br>

## ✅ Advantages of This Version

- Efficient **category-specific FAISS retrieval**.
- Nested metadata makes master lookup easier.
- Improved handling of **category mismatches** with fallback search.
- Pack-size numeric filtering reduces false positives.
- Outputs are structured and automatically **evaluated** with metrics.
- **No external API calls** – everything runs offline with Ollama.

<br>

## 🧠 Matching Logic Highlights
|Step| Logic Used|
|:-|:-|
|Master Embedding|	`nomic-embed-text` via Ollama|
|Retrieval| Category FAISS Index *(L2 Similarity)*|
|Pack Size|	Numeric extraction + filtering|
|Transaction → Master|	Top-k candidates (default k=10, final top-3)|
|Final Matching| Grouped + formatted matches per transaction row|
|Evaluation| Accuracy (Top-1/Top-3), Precision, Recall, Error rates|
<!-- |LLM Validation| Mistral chooses best `matched_itemcode`| -->

<br>

## 📅 Changelog

|Version|Changes|
|:-|:-|
|**v1**|*Rule-based + fuzzy matching*|
|**v2**|*LLM-powered item name extraction using Mistral via Ollama*|
|**v3**|*Integrated LLM + Structured SKU Matching + Fallback Suggestion Engine*|
|**v5**|*Full embedding pipeline with FAISS retrieval, pack-size filtering, formatted outputs, and LLM re-validation*|
|**v6**|*Category-specific FAISS indexes, nested metadata, automated evaluation metrics, and simplified offline pipeline*|

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
<!-- - `Exact.csv` summarizes validation performance, but LLM evaluation may vary. -->
- Matching pipeline is modular — each stage (`embedding → retrieval → filtering → formatting → validation`) can be run independently.