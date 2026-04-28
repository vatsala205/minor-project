import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from utils.preprocessing import load_data

class RiskRAG:

    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.df = load_data().dropna(subset=["Risk_Level"])
        self.df["text"] = self.df.apply(self.convert_row_to_text, axis=1)

        self.embeddings = self.model.encode(
            self.df["text"].tolist(),
            convert_to_numpy=True
        )

        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dimension)
        self.index.add(self.embeddings)

    def convert_row_to_text(self, row):
        return (
            f"Workload {row['Workload_Demand']}, "
            f"Stress {row['Stress_Level']}, "
            f"Mental drain {row['Mentally_Drained']}, "
            f"Sleep quality {row['Sleep_Quality']}, "
            f"Sleep consistency {row['Sleep_Consistency']}, "
            f"Daytime sleepiness {row['Daytime_Sleepiness']}, "
            f"Wake feeling {row['Wake_Up_Feeling']}, "
            f"Social connectedness {row['Social_Connectedness']}, "
            f"Resting HR {row['Resting_HR']}."
        )

    def retrieve_similar_cases(self, query_text, top_k=5):
        query_embedding = self.model.encode([query_text])
        distances, indices = self.index.search(query_embedding, top_k)

        similar_cases = self.df.iloc[indices[0]]
        return similar_cases

    def generate_explanation(self, query_text):
        similar_cases = self.retrieve_similar_cases(query_text)

        risk_counts = similar_cases["Risk_Level"].value_counts()

        explanation = (
            f"This individual shows behavioral patterns similar to "
            f"{len(similar_cases)} historical cases. "
            f"Among them: {risk_counts.to_dict()}. "
            f"Primary contributing factors appear to be high workload, "
            f"stress intensity, and mental exhaustion severity."
        )

        return explanation, similar_cases