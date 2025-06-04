# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_2-LLM_Enhanced-white)

## 🎯 Objective

> The objective of this project is to match and synchronize data between two CSV files: <br> **master.csv** and **transaction.csv**. 

The **Version 2** of this project is to enhance SKU matching accuracy by **extracting clean product names** from raw item descriptions using a **local LLM** (Mistral via Ollama).
This overcomes limitations of traditional filtering and fuzzy matching by leveraging semantic understanding from language 

<br>

## 🆕 Key Features

- Uses **Mistral**, a local open-source LLM, for extracting product names from messy transactional item descriptions.
- Automates product name parsing with natural language understanding.
- Outputs a cleaned CSV with a new column `Product Name` extracted from `

<br>

## 🔁 Input & Output
🔹 **Input File:** `DataCleaned/transaction_cleaned.csv`  
_Must contain a column named_ `ITEMDESC`.
<br>
🔹 **Output File:** `ProductName.csv`  
_Includes all original columns plus an additional column_ `Product` 

<br>

## ⚙️ Project Structure

```sh
StockKeepingUnit
|-- Data
|   |-- master.csv
|   |-- transaction.csv
|   |-- ActualModelResults
|       |-- ACTUAL_ModelResult_2024-10__dated_2024-05-23.xlsx
|       |-- October_ACTUAL_ModelResults (labelled).xlsx
|-- ProductName.csv         
|-- main.py
```

<br>

## 🚀 How It Works
`main.py` uses the Mistral model served via Ollama to extract structured product names from the raw `ITEMDESC` field of the transaction file.

> Example: <br>
**Input**: `CHANDANALEPA KOHOMBA AYURVEDA SOAP/75GM/CDBOX` <br>
**Output**: `CHANDANALEPA KOHOMBA AYURVEDA SOAP`

<br>

## 🛠️ Setup Guide (Windows)

 1. **Install Ollama**
 - Download the installer from: https://ollama.com/download
 - Install it and let it run in the background. It creates a local server at `http://localhost:11434`.

 2. **Pull the Mistral Model**
 - Open PowerShell or CMD and run: *(download size 4GB)*
 - `ollama pull mistral`

 3. **Start the Mistral Server**
 - `ollama run mistral` or `ollama serve`
 - This will launch the model in a conversational loop. For API usage, Ollama already runs a - background API server at `http://localhost:11434`.
 - You can keep ollama run mistral running in one terminal, or simply rely on the background service launched by Ollama on Windows startup.

<br>

## ▶️ Run the Code

```sh
python main.py
```

This will:
- Load transaction_cleaned.csv
- Extract product names using mistral
- Save the output to ProductName.csv

<br>

## ✅ Advantages of This Version
- Works offline (no API keys needed)
- Better accuracy in parsing descriptive product fields
- Scalable and customizable with other open LLMs (e.g., llama3, gemma, etc.)

<br>

## 📌 Notes
- This version focuses solely on product name extraction. SKU matching logic from [Version 1](https://github.com/AbhigyanSen/StockKeepingUnit/tree/600083f16ad2cadd431a51c6af4794f71406492d) can later be integrated using these cleaned names.

<br>


## 🚧 Future Plans
- Integrate product name extraction with fuzzy matching logic from master.csv
- Build a hybrid model: LLM output + rule-based refinement

<br>

## 📅 Changelog

|Version|Changes|
|:-|:-|
|**v1**|*Rule-based + fuzzy matching*|
|**v2**|*LLM-powered item name extraction using Mistral via Ollama*|
