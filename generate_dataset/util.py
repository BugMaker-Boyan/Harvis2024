from sentence_transformers import SentenceTransformer, util
import torch


class SimilarityUtil:
    
    def __init__(self):
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
    
    def filter_based_on_similarity_threshold(self, old_examples, new_examples, threshold):
        old_sentences = [f"{example['input']} {example['output']}" for example in old_examples]
        new_sentences = [f"{example['input']} {example['output']}" for example in new_examples]
        old_embeddings = self.model.encode(old_sentences, convert_to_tensor=True)
        new_embeddings = self.model.encode(new_sentences, convert_to_tensor=True)
        cosine_scores = util.cos_sim(new_embeddings, old_embeddings)
        max_cosine_scores, _ = torch.max(cosine_scores, dim=1)
        filter_examples = []
        for i in range(len(new_sentences)):
            if max_cosine_scores[i] < threshold:
                filter_examples.append(new_examples[i])
        return filter_examples
