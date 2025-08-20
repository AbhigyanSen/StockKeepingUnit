# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_4-Developing_WRT_New_Format-FFFF99)

## 🎯 Objective

> This project aims to intelligently match and synchronize data between two CSV files:
**master.csv** and **transaction.csv**.

The **Version 4** is being developed wrt to the new format provided by the Client. The Data Preprocessing logic has been changed. The changes can be seen in `DataPreprocessing.ipynb` in the section `Working with Nepal Data`.

- Clarifications regarding the new format which does not have the `ItemDesc` present in the master table has to be communicated with the Client.
- The `Suggestions` and the `LLM` parts are yet to be integrated and tested with this version.

<br>

## 🔁 Input & Output

- Input Files 
<br><span style="display:inline-block; margin-top:20px; margin-bottom:0px; margin-right:0px;"></span>*for* **`DataPreprocessing.ipynb`**
    * `Data/DataCleaned/master_cleaned.csv`
    * `Data/DataCleaned/transaction_cleaned.csv`
<br><span style="display:inline-block; margin-top:-15px; margin-bottom:40px; margin-right:-40px;"></span>
*for* **`main.ipynb`**
    * `Data/master.csv`
    * `Data/transaction.csv`

- Output Files
    * `matches.csv` – Full results with match status and error reasons.

<br>

## ⚙️ Project Structure

```sh
StockKeepingUnit/
├── Data/
│   ├── DataCleaned/
│		├── master_cleaned.csv
│		├── transaction_cleaned.csv
│   ├── master.csv
│   ├── transaction.csv										
├── main.py
├── suggestion.py
└── matches_sequence.csv
```

<br>

# DO NOT REFER TO THE SECTIONS BELOW (not updated wrt Version 4)
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
