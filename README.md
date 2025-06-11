# Stock Keeping Unit

![Static Badge](https://img.shields.io/badge/Version_2.3.3-Development_Version-red)

## Model Still under Development

- Use with Caution ⚠️
- **Implementing Suggestion Logic**

<br>

### Observations of the Model
The Model was tested with the first 100 rows of `transaction.csv`, which were saved as `transaction_FROMLABELLED.csv`, which was then mapped to `master.csv`. Below are the Observations:
- Upon testing for **String Matching**, it was found that: *(executing `main.py`)*
    * `MATCHED=0` **15** of **100** rows had **exact** string match with the master.
    * `MATCHED=1` **85** of **100** rows **exited** the string match due to mismatch in either of: `MANUFACTURE`,`BRAND`,`QTY`,`UNIT` or `PACKTYPE`
- Upon testing using **Mistral LLM**, it was found that: *(executing `main.py`)*
    * `MATCHED=2` Out of the **85** mismatched strings, **10** items were found similarly matching.
- The resultant file created was `matches.csv`, the `MATCHED` column shows the flags
- The **Suggestion Logic** was basically creating a copy of `matches.csv`, which consists of all the rows having `MATCHED = 1` or `2`.
    * Therefore the Suggestion Logic was implemented on **85** rows.

<br>

### Results of the Model
> The Results of the model was tested upon the 100 rows only, upon manual verification of 36 rows of **matches+suggestion.csv** with **Data.xlsx**, it was seen that only **3** rows out of **36** were found to be incorrectly mapped. <br> This disparity leads us to face the edge cases that are present. 

Row wise checking was done on 'master+suggestion.csv' and first 100 rows of `Data.xlsx` to find out the Results. 
*The results can be found in* `Result_Verification` *folder with version as* `2.3.2`

|Overall Matching Metrics|Values|
|:-|:-|
|True Matches| **90** (90.00%)| 
|False Matches| **10** (10.00%)|

<br>

|Metrics for Rows Containing '9'|Values|
|:-|:-|
|True '9' Matches| **22** (100.00%)| 
|False '9' Matches| **0** (0.00%)|

**NOTE**
- A **True Matche** depicts that the `NITEMCODE` in `Data.xlsx` is present in the corresponding row of `matches+suggestion.csv` in any of column `nitemcode` or `Suggestion`.
- A **False Match** is when the `NITEMCODE` of `Data.xlsx` is not present in the corresponding row of `matches+suggestion.csv` in any column.