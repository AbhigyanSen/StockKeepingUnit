# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_3-LLM_Enhanced_with_Suggestions-white)

## 🎯 Objective

> This project aims to intelligently match and synchronize data between two CSV files:
**master.csv** and **transaction.csv**.

The **Version 3** upgrade builds on LLM-powered product name extraction by integrating it with a structured SKU matching pipeline. It adds string matching logic, structured comparison, and fallback suggestion mechanisms to handle imperfect data, all while staying fully offline using a local Mistral model served via Ollama.

<br>

## 🆕 Key Features

- Enhanced product name extraction using **Mistral** (LLM via Ollama).
- Intelligent matching across multiple attributes (company, brand, pack size, unit, etc.).
- **Fallback Suggestion Engine** for unmatched items using similarity scoring.
- Full tracking of matched, unmatched, and partially matched rows with reasons.
- Outputs results in a match report `matches.csv` and a suggestions-enhanced file `matches+suggestions.csv`.

<br>

## 🔁 Input & Output

- Input Files       *previously processed data in* `DataPreprocessing.ipynb`
    * `Labelled_Data/master.csv`
    * `Labelled_Data/transaction.csv`
- Output Files
    * `matches.csv` – Full results with match status and error reasons.
    * `matches+suggestions.csv` – Same as above with a Suggestion column populated for fallback cases.

> Unmatched rows are analyzed separately to suggest the most probable item code from master data.

<br>

## ⚙️ Project Structure

```sh
StockKeepingUnit/
├── Data/
│   ├── ActualModelResults
│		├── ACTUAL_ModelResult_2024-10__dated_2024-05-23.xlsx		# Results from Benchmarked Model
│		├── October_ACTUAL_ModelResults (labelled).xlsx				# Labelled Data provided by Client
│   └── DataCleaned/												# Dropping unnecessary columns
│		├── master_cleaned.csv							
│		├── transaction_cleaned.csv
│   ├── master.csv
│   └── transaction.csv
├── Labelled_Data/
│   ├── Data.xlsx
│   ├── master.csv
│   └── transaction.csv
├── Result_Verification/											# Model Evaluation
│   └── Version_2.3.2/											
├── main.py
├── suggestion.py
├── matches.csv
└── matches+suggestions.csv
```

<br>

## 🚀 How It Works

<br>

📌 `main.py`

- Reads in `master.csv` and `transaction.csv`
- Applies multi-layered filtering: `MANUFACTURE`, `BRAND`, `QTY`, `UNIT`, `PACKSIZE`, `PACKTYPE`
- Extracts clean product names using **Mistral LLM**
- Scores matches using semantic similarity between cleaned names
- Outputs the match status:
    * `0`: Match found
    * `1`: No match
    * `2`: Partial match (low confidence or ambiguous match)
- Annotates errors such as mismatches in MANUFACTURE, BRAND, etc.

<br>

📌 `suggestion.py`

- Reads `matches.csv`
- For rows where `MATCHED == 1` (i.e., no confident match), generates fallback suggestions
- Uses string similarity logic across multiple fields to suggest the most likely item code
- Appends a `Suggestion` column and saves as `matches+suggestions.csv`


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

> Note: Mistral model size is ~4GB and requires enough RAM (~8GB+ recommended).

<br>

## ▶️ Run the Code

```sh
# Step 1: Run main matching process
python main.py

# Step 2: Generate fallback suggestions
python suggestion.py
```

> Ensure Ollama is running in the background with Mistral loaded

<br>

## ✅ Advantages of This Version
- Combines **LLM reasoning + Rule-Based Logic** for robust matching.
- Handles messy descriptions, abbreviations, and inconsistent formatting.
- Works fully offline (no OpenAI/API keys).
- Supports match status explanation and post-match improvement with suggestions.

<br>

## 🧠 Matching Logic Highlights
|Step| Logic Used|
|:-|:-|
|Manufacturer|	Exact Match|
|Brand| Exact Match|
|Pack Size|	Normalized String Match|
|Pack Type|	Equivalence Set|
|Product Name| LLM Extraction|
|Fallback| Weighted Similarity (`Company`, `Brand`, `Itemdesc`)|

<br>

## 📅 Changelog

|Version|Changes|
|:-|:-|
|**v1**|*Rule-based + fuzzy matching*|
|**v2**|*LLM-powered item name extraction using Mistral via Ollama*|
|**v3**|*Integrated LLM + Structured SKU Matching + Fallback Suggestion Engine*|

<br>

## 📈Model Metrics

|||Count| Percentage|
|:-|:-|:-|:-|
|**True Positive (TP)**| Correct Match| 480| 46.24%|
|**True Negative (TN)**| Correct No Match| 228| 21.97%|
|**False Negative (FN)**| Incorrect No Match| 222| 21.39%|
|**False Positive (FP)**| Incorrect Match| 108| 10.40%|
|| **Total**| 1038| 100%|

<br>

## 📌 Notes
- This version focuses on imporving the model. 
- SKU matching logic from [Version 2](https://github.com/AbhigyanSen/StockKeepingUnit/tree/600083f16ad2cadd431a51c6af4794f71406492d) can later be followed in case of Fallback.
- If the product name extraction fails for any item, the error is logged in the `ProductName` field.
- Matching threshold for product name similarity is `0.85`.
- `suggestion.py` provides a best-effort match even if exact filtering fails.
