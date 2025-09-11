import ollama
import json
import re

t_itemdesc = "SOFTY BABY PANT/M CRAWLER/4 PICES/PPH"

m_sku_list = {
    1: "SOFTY BABY PANT-SMALL(S) PPH 4 NOS",
    2: "SOFTY BABY PANT-MEDIUM(M) PPH 4 NOS",
    3: "SOFTY BABY PANT-LARGE(L) PPH 4 NOS",
    4: "TENDER TOUCH BABY PULL UP PANTS-MEDIUM(M) PPH 5 NOS",
    5: "TENDER TOUCH BABY PULL UP PANTS-SMALL(S) PPH 5 NOS"
}

discarded_responses = []

for i in range(10):
    prompt = f"""
    We have an item description: "{t_itemdesc}"

    And the following SKU list:
    {json.dumps(m_sku_list, indent=2)}

    Find the best matching itemID from the list above.
    Return ONLY the itemID as a number (no explanation).
    """

    response = ollama.chat(
        model="mistral",
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.8, "max_tokens": 20}  # allow a bit more text
    )

    raw_output = response["message"]["content"].strip()

    # Extract first digit using regex
    match = re.search(r"\d", raw_output)
    if match:
        best_match_id = match.group(0)
        print("Best matching itemID:", best_match_id)
    else:
        best_match_id = None
        print("No valid ID found!")

    # Save discarded part if response had extra text
    if raw_output != best_match_id:
        discarded_responses.append(raw_output)

# Save discarded outputs into a file
if discarded_responses:
    with open("discarded_responses.txt", "w", encoding="utf-8") as f:
        f.write("\n\n---\n\n".join(discarded_responses))

print("Done. Discarded responses saved in discarded_responses.txt")