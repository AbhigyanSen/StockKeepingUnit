SKU (Stock Keeping Unit):

Use Case: Merge the New Data (sl_cpg_data-new-items.csv) with the main Master Data (sl_cpg_202410_item)

Description (Example Use Case)
Suppose Amul is a company having multiple products in its inventory. Now, Amul maintains a Database (Master Table) for its inventory management to track which items are appeneded (newly produced) and which items are removed (sent to wholesellers/dealers from amul's warehouse). Again the dealers also maintain a database containing which products are pushed (added to their inventory) and popped (sold to customers/retailers).
Now, lets Suppose the case below:

|              | Category        | Product      | Abbreviation    |
| :---         |    :----:       |         ---: |            ---: |
| (master)     | AMUL            | AMUL TAAZA   | TZ              |
| (new data)   | WHOLESELLER     | AMUL TAAZA   | ATZ             |

Now, we need to track the same product in both the tables so that Amul also gets a track of which items are sent and brought by whom. Basically, its mapping of a product from end to end and SECONDLY, when the wholesellers give their PO (purchase order) to Amul, then Amul can track which product they have mentioned in their PO (as amul abbreviated its product as TZ, not necessarily the wholeseller has also tagged the product with the same abbreviation, here its ATZ. Hence a mapping of TZ to ATZ is required so that if both the tables are merged, there is no Redundant Data present)
