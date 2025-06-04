# TESTING WITH 1 INPUT ---------------------------------------------------------

# import requests

# def extract_product_name_ollama(description, model="mistral"):
#     prompt = (
#         f"Extract only the product name from the following description. "
#         f"Do not include quantity, unit, or pack type:\n\n"
#         f"{description}\n\nProduct name:"
#     )

#     response = requests.post(
#         "http://localhost:11434/api/generate",
#         json={"model": model, "prompt": prompt, "stream": False}
#     )
#     if response.status_code == 200:
#         return response.json()["response"].strip()
#     else:
#         return f"Error: {response.text}"

# # 🔍 Test it
# print(extract_product_name_ollama("CHANDANALEPA KOHOMBA AYURVEDA SOAP/75GM/CDBOX"))



# TESTING WITH THE WHOLE CSV

import pandas as pd
import requests
import time

def extract_product_name_ollama(description, model="mistral"):
    prompt = (
        f"Extract only the product name from the following description. "
        f"Do not include quantity, unit, or pack type:\n\n"
        f"{description}\n\nProduct name:"
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": False}
    )
    if response.status_code == 200:
        return response.json()["response"].strip()
    else:
        return f"Error: {response.text}"

def process_csv(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    product_names = []
    for desc in df["ITEMDESC"]:
        name = extract_product_name_ollama(desc)
        product_names.append(name)
        time.sleep(1)  # prevent overload
    df["Product Name"] = product_names
    df.to_csv(output_csv, index=False)
    print(f"Saved to {output_csv}")

# Example usage
if __name__ == "__main__":
    process_csv(R"Data\DataCleaned\transaction_cleaned.csv", "ProductName.csv")