import os
from sklearn.datasets import fetch_20newsgroups

def download_and_save_data(output_dir="data/docs"):
    # Source: [cite: 11, 12, 13, 18, 19]
    print("Downloading 20 Newsgroups dataset...")
    dataset = fetch_20newsgroups(subset='train', remove=('headers', 'footers', 'quotes'))
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Limit to 200 documents as per objective [cite: 4, 11]
    limit = 200
    print(f"Saving first {limit} documents to {output_dir}...")
    
    for i in range(limit):
        text = dataset.data[i]
        if text.strip(): # Skip empty files
            filename = f"doc_{i:03d}.txt"
            with open(os.path.join(output_dir, filename), "w", encoding="utf-8") as f:
                f.write(text)
    
    print("Data preparation complete.")

if __name__ == "__main__":
    download_and_save_data()