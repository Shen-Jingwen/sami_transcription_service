import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer
from langchain_text_splitters import SpacyTextSplitter
from sentence_transformers import SentenceTransformer
from typing import Dict, List
import torch
from qdrant_client import http, models, QdrantClient
from transformers import T5Tokenizer, T5ForConditionalGeneration
import time


class HybridVectorSearch:
    cuda_device = torch.device("cpu")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sparse_model = "naver/splade-v3"
    tokenizer = AutoTokenizer.from_pretrained(sparse_model)
    model = AutoModelForMaskedLM.from_pretrained(sparse_model).to(cuda_device)

    text_splitter = SpacyTextSplitter(chunk_size=1000)
    dense_encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

    model_name_t5 = "Falconsai/text_summarization"  # "t5-small"
    tokenizer_t5 = T5Tokenizer.from_pretrained(model_name_t5)
    model_t5 = T5ForConditionalGeneration.from_pretrained(model_name_t5).to(device)

    client = QdrantClient(url="http://localhost:6333")
    earnings_collection = "earnings_calls"


    @classmethod
    def initialize(cls):
        cls.init_collection()
        cls.check_and_add_sample_data()

    @staticmethod
    def init_collection():
        try:
            collection_info = HybridVectorSearch.client.get_collection(HybridVectorSearch.earnings_collection)
            print(f"Collection {HybridVectorSearch.earnings_collection} already exists")
        except Exception as e:
            print(f"Creating new collection: {str(e)}")
            
            # Qdrant 创建集合
            HybridVectorSearch.client.create_collection(
                collection_name=HybridVectorSearch.earnings_collection,
                vectors_config={
                    "dense_vector": models.VectorParams(
                        size=384,  # all-MiniLM-L6-v2 的向量维度
                        distance=models.Distance.COSINE
                    )
                },
                sparse_vectors_config={
                    "sparse_vector": models.SparseVectorParams(
                        index=models.SparseIndexParams(
                            on_disk=False,
                            full_scan_threshold=1000
                        )
                    )
                }
            )
            print(f"Collection {HybridVectorSearch.earnings_collection} created successfully")

    @staticmethod
    def check_and_add_sample_data():
        try:
            collection_info = HybridVectorSearch.client.get_collection(HybridVectorSearch.earnings_collection)
            if collection_info.points_count == 0:
                print("Collection is empty. Adding sample data...")
                texts = ["This is a sample transcription text.", "Another sample text for testing."]
                for i, text in enumerate(texts):
                    # 计算密集向量
                    dense_vector = HybridVectorSearch.dense_encoder.encode(text).tolist()

                    # 计算稀疏向量
                    sparse_vec, _, _ = HybridVectorSearch.compute_vector(text)
                    indices = sparse_vec.cpu().nonzero().numpy().flatten()
                    values = sparse_vec.cpu().detach().numpy()[indices]
                    sparse_vector = models.SparseVector(indices=indices, values=values)

                    # 添加点到集合
                    HybridVectorSearch.client.upsert(
                        collection_name=HybridVectorSearch.earnings_collection,
                        points=[
                            models.PointStruct(
                                id=i,
                                vector={"dense_vector": dense_vector},
                                sparse_vectors={"sparse_vector": sparse_vector},
                                payload={"conversation": text}
                            )
                        ]
                    )
                print("Sample data added successfully.")
        except Exception as e:
            print(f"Error checking or adding sample data: {e}")

    @staticmethod
    def reciprocal_rank_fusion(
        responses: List[List[http.models.ScoredPoint]], limit: int = 10
    ) -> List[http.models.ScoredPoint]:
        def compute_score(pos: int) -> float:
            ranking_constant = 2  # the constant mitigates the impact of high rankings by outlier systems
            return 1 / (ranking_constant + pos)

        scores: Dict[http.models.ExtendedPointId, float] = {}
        point_pile = {}
        for response in responses:
            for i, scored_point in enumerate(response):
                if scored_point.id in scores:
                    scores[scored_point.id] += compute_score(i)
                else:
                    point_pile[scored_point.id] = scored_point
                    scores[scored_point.id] = compute_score(i)

        sorted_scores = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        sorted_points = []
        for point_id, score in sorted_scores[:limit]:
            point = point_pile[point_id]
            point.score = score
            sorted_points.append(point)
        return sorted_points

    @staticmethod
    def summary(text: str):
        inputs = HybridVectorSearch.tokenizer_t5.encode(
            f"summarize: {text}", return_tensors="pt", max_length=1024, truncation=True
        ).to("cuda")
        summary_ids = HybridVectorSearch.model_t5.generate(
            inputs,
            max_length=512,
            min_length=100,
            length_penalty=2.0,
            num_beams=4,
            early_stopping=True,
        )
        summary = HybridVectorSearch.tokenizer_t5.decode(
            summary_ids[0], skip_special_tokens=True
        )
        return summary

    @staticmethod
    def compute_vector(text):
        tokens = HybridVectorSearch.tokenizer(text, return_tensors="pt").to(
            HybridVectorSearch.cuda_device
        )
        split_texts = []
        if len(tokens["input_ids"][0]) >= 512:
            summary = HybridVectorSearch.summary(text)
            split_texts = HybridVectorSearch.text_splitter.split_text(text)
            tokens = HybridVectorSearch.tokenizer(summary, return_tensors="pt").to(
                HybridVectorSearch.cuda_device
            )

        output = HybridVectorSearch.model(**tokens)
        logits, attention_mask = output.logits, tokens.attention_mask
        relu_log = torch.log(1 + torch.relu(logits))
        weighted_log = relu_log * attention_mask.unsqueeze(-1)
        max_val, _ = torch.max(weighted_log, dim=1)
        vec = max_val.squeeze()
        print("Computed vector:", vec) 

        # indices = vec.nonzero().numpy().flatten().tolist()
        # values = vec.detach().numpy()[vec.nonzero()].flatten().tolist()
        return vec, tokens, split_texts
        # return indices, values, split_texts

    @staticmethod
    def search(query_text: str, symbol="AMD"):
        vectors, tokens, split_texts = HybridVectorSearch.compute_vector(query_text)
        indices = vectors.cpu().nonzero().numpy().flatten()
        values = vectors.cpu().detach().numpy()[indices]

        sparse_query_vector = models.SparseVector(indices=indices, values=values)

        query_vector = HybridVectorSearch.dense_encoder.encode(query_text).tolist()
        limit = 3

        dense_request = models.SearchRequest(
            vector=models.NamedVector(name="dense_vector", vector=query_vector),
            limit=limit,
            with_payload=True,
        )
        sparse_request = models.SearchRequest(
            vector=models.NamedSparseVector(
                name="sparse_vector", vector=sparse_query_vector
            ),
            limit=limit,
            with_payload=True,
        )

        (dense_request_response, sparse_request_response) = (
            HybridVectorSearch.client.search_batch(
                collection_name=HybridVectorSearch.earnings_collection,
                requests=[dense_request, sparse_request],
            )
        )
        ranked_search_response = HybridVectorSearch.reciprocal_rank_fusion(
            [dense_request_response, sparse_request_response], limit=10
        )

        search_response = ""
        for search_result in ranked_search_response:
            search_response += search_result.payload["conversation"] + "\n"
        return ranked_search_response

    @staticmethod
    def chat_search(query: str, chat_history):
        result = HybridVectorSearch.search(query)
        chat_history.append((query, "Search Results"))
        for search_result in result[:3]:
            text = search_result.payload["conversation"]
            summary = HybridVectorSearch.summary(text) + f'\n```\n{text} \n```'
            chat_history.append((None, summary))
        return "", chat_history


    @staticmethod
    def insert_text(text: str, metadata: dict = None):
        """
        将文本插入向量数据库
        :param text: 要插入的文本内容
        :param metadata: 附加元数据(如speaker, timestamp等)
        """
        if not text or not text.strip():
            print("warning: empty data skip")
            return False
        
        try:
            dense_vector = HybridVectorSearch.dense_encoder.encode(text).tolist()
            sparse_vec, _, _ = HybridVectorSearch.compute_vector(text)
            sparse_indices = sparse_vec.nonzero().numpy().flatten().tolist()
            sparse_values = sparse_vec.detach().numpy()[sparse_vec.nonzero()].flatten().tolist() 
            
            point_id = int(time.time() * 1000)  
            
            point = models.PointStruct(
                id=point_id,
                vector={"dense_vector": dense_vector},
                sparse_vectors={
                    "sparse_vector": models.SparseVector(
                        indices=sparse_indices,
                        values=sparse_values
                    )
                },
                payload={
                    "conversation": text,
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    **(metadata or {})
                }
            )
            
            operation_info = HybridVectorSearch.client.upsert(
                collection_name=HybridVectorSearch.earnings_collection,
                points=[point]
            )
            
            print(f"insert success! ID: {point_id}, status: {operation_info.status}")
            return True
            
        except Exception as e:
            print(f"insert fail: {str(e)}")
            return False

    @staticmethod
    def _validate_sparse_vector(indices, values):
        """验证稀疏向量格式"""
        if len(indices) != len(values):
            raise ValueError("indices and values must same length")
        if not all(isinstance(v, float) for v in values):
            raise ValueError("all values must be float")
        return indices, values


    @staticmethod
    def verify_data_insertion():
        """验证数据是否成功插入"""
        try:
            latest = HybridVectorSearch.client.scroll(
                collection_name=HybridVectorSearch.earnings_collection,
                limit=1,
                with_vectors=True
            )
            if not latest:
                print("no data")
                return False
                
            print(f"latest data: {latest[0]}")
            return True
        except Exception as e:
            print(f"verify fail: {str(e)}")
            return False