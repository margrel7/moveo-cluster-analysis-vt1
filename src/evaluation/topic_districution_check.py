import pandas as pd
from collections import Counter
import re
import nltk
from nltk.corpus import stopwords
nltk.download("stopwords", quiet=True)

_STOP_DE = set(stopwords.words("german"))
_STOP_EN = set(stopwords.words("english"))
STOP_WORDS = _STOP_DE | _STOP_EN

df = pd.read_csv("outputs/workshop_clusters.csv")
raw = pd.read_csv("data/raw/raw_data.csv")[["photo_id", "topic", "description"]]
df = df.merge(raw, on="photo_id", how="left")

for cluster_id in sorted(df["hdbscan_label_assigned"].unique()):
    cluster_df = df[df["hdbscan_label_assigned"] == cluster_id]
    cat = cluster_df["category"].value_counts().to_dict()
    
    all_tokens = []
    for text in cluster_df["topic"].dropna():
        text = text.lower()
        text = re.sub(r"[^a-z0-9äöüß\s]", " ", text)
        tokens = [t for t in text.split() if len(t) >= 3 and t not in STOP_WORDS]
        all_tokens.extend(tokens)
    
    top = [w for w, _ in Counter(all_tokens).most_common(6)]
    print(f"Cluster {cluster_id} (n={len(cluster_df)}) | works_well={cat.get('works_well',0)} problem={cat.get('problem',0)}")
    print(f"  Topics: {', '.join(top)}\n")

