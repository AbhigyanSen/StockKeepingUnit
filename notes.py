1. select 'CATEGORY' from transaction, match it exactly with the 'catcode' of the master. if match, store those rows of master (A). if not match then ERROR in the matches.csv will be CATCODE and exit
2. select 'MANUFACTURE' from transaction, match it with the 'company' of the master. if match in A, store those rows of master (B). if not match then ERROR in the matches.csv will be MANUFACTURE and exit
3. select 'BRAND' from transaction, match it with the 'brand' of the master. if match in B, store those rows of master (C) and go to step 4. IF NOT MATCH THEN DO NOT EXIT continue to next step 3.1.
    3.1. select 'PACKTYPE' from transaction, match it with the 'packtype' of the master. if match in B, store those rows of master (D). if no match then ERROR in the matches.csv will be PACKTYPE and exit
    3.2. select 'PACKSIZE' from transaction, match it with the 'qty' of the master, if match in D, store those rows of master (E). if no match then ERROR in the matches.csv will be PACKSIZE and exit.
    NOTE THAT: In packsize column of transaction the values are like '180 ML', '250ML', '20' ... etc. but in the master 'qty' column contains the interger values like '180', '250', '20' ... etc and the 'uom' column contain the values like '', 'ML', 'NO', ... etc. so you need to handle this situation by braking the 'PACKSIZE' column into its integere and string parts and then match them accordingly with 'qty' and 'uom'.

4. select 'PACKTYPE' from transaction, match it with the 'packtype' of the master. if match in C, store those rows of master (D). if no match then ERROR in the matches.csv will be PACKTYPE and exit
5. select 'PACKSIZE' from transaction, match it with the 'qty' of the master, if match in D, store those rows of master (E). if no match then ERROR in the matches.csv will be PACKSIZE and exit.
    NOTE THAT: In packsize column of transaction the values are like '180 ML', '250ML', '20' ... etc. but in the master 'qty' column contains the interger values like '180', '250', '20' ... etc and the 'uom' column contain the values like '', 'ML', 'NO', ... etc. so you need to handle this situation by braking the 'PACKSIZE' column into its integere and string parts and then match them accordingly with 'qty' and 'uom'.
    
Now, suppose that one row of transaction matches with multiple rows of the master, then u will list all the matched rows from the master. like suppose row 1 of transaction matches with rows 1,2 and 3 of master, then the output will be like:
    Transaction     Master
    Row 1           Row 1
    Row 1           Row 2
    Row 1           Row 3..
    
I do not want to handle the ITEMDESC part and LLM part right now, hence u can also ignore those. give me the code with these changes



6. now select 'ITEMDESC' drom transaction, match it with the concatenated string of 'company', 'brand', 'packtype' and 'qty' of the master of E. if match then store those rows of master (F). if no match then ERROR in the matches.csv will be ITEMDESC and exit.