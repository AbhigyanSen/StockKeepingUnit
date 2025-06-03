# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_1-_Initial_Approach-yellow)

## 🎯 Objective
> The objective of this project is to match and synchronize data between two CSV files: <br>**master.csv** and **transaction.csv**. 

Here's a breakdown of each 
<br>


`master.csv`: Contains over 30,000 rows of data structured with fields such as
|itemcode|itemdesc|catcode|category|company|brand|packaging|flavor|color|qty|uomdesc|pack_size|launchdate.
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|

<br>

`transaction.csv`: Contains transactional data with fields like 
|PERIOD|AUDITTYPE|STORECODE|DLRCODE|ITEMCODE|CATEGORY|MANUFACTURE|BRAND|ITEMDESC|MRP|PACKSIZE|PACKTYPE|COMMENTS|IMAGE.
|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|:-|

<br>

## 🔄 Process Overview:
1. **Data Cleaning:** Both master.csv and transaction.csv undergo data cleaning to remove unnecessary columns, ensuring only relevant data is processed.
2. **Matching Logic:** 
    * Data from transaction.csv is matched against master.csv based on specific criteria: MANUFACTURE (matching company in master.csv), BRAND, PACKSIZE, PACKTYPE, and other relevant attributes.
    * Initial filtering is done to select potential matches based on exact matches of key 
    * Sorting and further refinement based on exact matches of company, brand, quantity, packtype are performed.
    * Fuzzy matching using tools like Fuzzy Wuzzy is applied to handle discrepancies in textual data.
3. **Output:**
    * A CSV file (matches.csv) is generated as output, containing matched data from transaction.csv with corresponding entries from master.csv.

<br>

## 🗂️Project Structure
```sh
StockKeepingUnit
|-- Data
    |-- master.csv
    |-- transaction.csv
    |-- ActualModelResults
        |-- ACTUAL_ModelResult_2024-10__dated_2024-05-23.xlsx
        |-- October_ACTUAL_ModelResults (labelled).xlsx
|-- demo.py
|-- main.py
|-- master_DataPreprocessing.py
|-- master_cleaned.csv
|-- matches.csv
|-- trans_DataPreprocessing.py
|-- transaction_cleaned.csv
```
<br>

## 📌 Notes:
- The project includes scripts: `master_DataPreprocessing.py`, `trans_DataPreprocessing.py` for cleaning respective CSV files before matching.
- Results from both manual checks `demo.py` and automated processes `main.py` are stored under `ActualModelResults`.

<br>

### Current Status:
This version has shown suboptimal results. Future iterations will focus on improving matching accuracy and efficiency.