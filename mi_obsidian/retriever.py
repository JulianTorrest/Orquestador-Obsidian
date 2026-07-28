import streamlit as st
import db
import graph_store
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from langchain_core.retrievers import BaseRetriever
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from typing import List

# --- Caching for expensive models and data ---

@st.cache_resource
def get_all_docs_for_bm25():
    """
    Fetches all documents from the database to create the BM25 index.
    This is cached to avoid hitting the DB on every query.
    """
    all_notes = db.get_all_notes_full_content()
    if not all_notes:
        return None, None
    
    # Create a mapping from doc_id to content for later retrieval
    doc_map = {note['id']: note for note in all_notes}
    
    # BM25 requires a list of tokenized documents
    tokenized_corpus = [note['content'].split(" ") for note in all_notes]
    
    bm25 = BM25Okapi(tokenized_corpus)
    return bm25, doc_map

@st.cache_resource
def get_cross_encoder():
    """
    Loads and caches the Cross-Encoder model for re-ranking.
    This model is more accurate than the bi-encoder for relevance scoring.
    """
    return CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# --- Hybrid Search and Re-ranking Logic ---

def hybrid_search_and_rerank(query, top_k=10):
    """
    Performs hybrid search (semantic + keyword), graph expansion, and then re-ranks the results.
    """
    bm25, doc_map = get_all_docs_for_bm25()
    cross_encoder = get_cross_encoder()

    if not bm25:
        # Fallback si no hay contexto
        return []

    # 1. Semantic Search (from db.py)
    semantic_results_df = db.search_notes(query, limit=top_k) 
    semantic_ids = set(semantic_results_df['id']) if not semantic_results_df.empty else set()

    # 2. Keyword Search (BM25)
    tokenized_query = query.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    doc_ids_from_map = list(doc_map.keys())
    
    # Combine scores with doc IDs
    bm25_results = sorted(zip(bm25_scores, doc_ids_from_map), reverse=True, key=lambda x: x[0])[:top_k]
    bm25_ids = set([doc_id for _, doc_id in bm25_results])

    # 3. Combine initial results (unique IDs)
    combined_ids = set(semantic_ids.union(bm25_ids))
    
    # 3.5 GRAPH RAG: Get neighbors for the retrieved chunks based on their source files (titles)
    titles_retrieved = {doc_map[doc_id]['title'] for doc_id in combined_ids if doc_id in doc_map}
    graph_neighbor_titles = set()
    for title in titles_retrieved:
        graph_neighbor_titles.update(graph_store.get_neighbors(title))
        
    # Add any chunk that belongs to a neighbor note
    graph_ids = set()
    if graph_neighbor_titles:
        for doc_id, doc in doc_map.items():
            if doc['title'] in graph_neighbor_titles:
                graph_ids.add(doc_id)
                
    combined_ids = list(combined_ids.union(graph_ids))

    if not combined_ids:
        return []

    # 4. Re-ranking with Cross-Encoder
    # Create pairs of [query, document_content] for the cross-encoder
    cross_inp = [[query, doc_map[doc_id]['content']] for doc_id in combined_ids]
    cross_scores = cross_encoder.predict(cross_inp)

    # Combine IDs with their new scores
    reranked_results = sorted(zip(cross_scores, combined_ids), reverse=True, key=lambda x: x[0])

    # 5. Format final results
    final_results = []
    for score, doc_id in reranked_results[:top_k]:
        doc = doc_map[doc_id]
        doc['relevance_score'] = score
        final_results.append(doc)
        
    return final_results

# Langchain Custom Retriever Wrapper
class AdvancedRetriever(BaseRetriever):
    top_k: int = 5
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        results = hybrid_search_and_rerank(query, top_k=self.top_k)
        docs = []
        for res in results:
            doc = Document(
                page_content=res['content'],
                metadata={"title": res['title'], "relevance": float(res['relevance_score'])}
            )
            docs.append(doc)
        return docs