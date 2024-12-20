import os
import requests
import json
import numpy as np
import faiss
from typing import List, Union, Dict, Optional

# PDF and Document Extraction
import PyPDF2
import docx


# Hugging Face Inference API
from huggingface_hub import HfApi

class DocumentExtractor:
    @staticmethod
    def extract_text_from_pdf(pdf_path: str) -> str:
        """Extract text from PDF file"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
                return text
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_path}: {e}")
            return ""

    @staticmethod
    def extract_text_from_docx(docx_path: str) -> str:
        """Extract text from DOCX file"""
        try:
            doc = docx.Document(docx_path)
            return "\n".join([paragraph.text for paragraph in doc.paragraphs])
        except Exception as e:
            print(f"Error extracting text from DOCX {docx_path}: {e}")
            return ""

class HuggingFaceRAGPipeline:
    def __init__(self, 
                 hf_token: str = None,
                 embedding_model='sentence-transformers/all-MiniLM-L6-v2', 
                 qa_model='deepset/roberta-base-squad2'):
        """
        Initialize RAG pipeline with Hugging Face online inference
        
        Args:
            hf_token (str): Hugging Face API token
            embedding_model (str): Embedding model for documents
            qa_model (str): Question-answering model
        """
        # Set up Hugging Face API
        self.api = HfApi()
        self.hf_token = hf_token or os.getenv('HF_TOKEN')
        
        if not self.hf_token:
            raise ValueError("Hugging Face API token is required. Set HF_TOKEN environment variable or pass token.")
        
        # Model configurations
        self.embedding_model = embedding_model
        self.qa_model = qa_model
        
        # Document storage
        self.faiss_index = None
        self.document_chunks = []
        self.chunk_metadata = []
    
    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings using Hugging Face Inference API
        
        Args:
            texts (List[str]): List of texts to embed
        
        Returns:
            numpy array of embeddings
        """
        api_url = f"https://api-inference.huggingface.co/pipeline/feature-extraction/{self.embedding_model}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(
            api_url, 
            headers=headers, 
            json={"inputs": texts, "wait_for_model": True}
        )
        
        if response.status_code != 200:
            raise Exception(f"Embedding API error: {response.text}")
        
        return np.array(response.json())
    
    def _qa_inference(self, question: str, context: str) -> str:
        """
        Perform question answering using Hugging Face Inference API
        
        Args:
            question (str): Input question
            context (str): Context for answering
        
        Returns:
            str: Generated answer
        """
        api_url = f"https://api-inference.huggingface.co/models/{self.qa_model}"
        headers = {
            "Authorization": f"Bearer {self.hf_token}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "inputs": {
                "question": question,
                "context": context
            },
            "wait_for_model": True
        }
        
        response = requests.post(api_url, headers=headers, json=payload)
        
        if response.status_code != 200:
            raise Exception(f"QA API error: {response.text}")
        
        result = response.json()
        return result.get('answer', 'No answer found')
    
    def preprocess_documents(self, 
                              documents: List[str], 
                              chunk_size: int = 200, 
                              overlap: int = 50):
        """
        Preprocess documents into chunks and create embeddings
        
        Args:
            documents (List[str]): List of document texts
            chunk_size (int): Number of tokens per chunk
            overlap (int): Tokens to overlap between chunks
        """
        all_chunks = []
        all_embeddings = []
        all_metadata = []
        
        for doc_index, doc in enumerate(documents):
            # Simple chunking (you might want to use a more sophisticated tokenization)
            chunks = [
                doc[i:i+chunk_size] 
                for i in range(0, len(doc), chunk_size - overlap)
            ]
            
            # Generate embeddings
            chunk_embeddings = self._get_embeddings(chunks)
            
            # Create metadata
            metadata = [
                {
                    'document_index': doc_index, 
                    'start_char': i * (chunk_size - overlap),
                    'length': len(chunk)
                } 
                for i, chunk in enumerate(chunks)
            ]
            
            all_chunks.extend(chunks)
            all_embeddings.append(chunk_embeddings)
            all_metadata.extend(metadata)
        
        # Combine embeddings
        embeddings = np.vstack(all_embeddings)
        
        # Create FAISS index
        dimension = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dimension)
        self.faiss_index.add(embeddings)
        
        # Store chunks and metadata
        self.document_chunks = all_chunks
        self.chunk_metadata = all_metadata
    
    def retrieve_context(self, 
                         query: str, 
                         top_k: int = 3) -> List[Dict[str, Union[str, Dict]]]:
        """
        Retrieve most relevant document chunks with metadata
        
        Args:
            query (str): Input query
            top_k (int): Number of top chunks to retrieve
        
        Returns:
            List of dictionaries with chunk text and metadata
        """
        if self.faiss_index is None:
            raise ValueError("No documents preprocessed. Call preprocess_documents first.")
        
        # Encode query
        query_embedding = self._get_embeddings([query])
        
        # Search FAISS index
        distances, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Prepare context results
        context_results = []
        for idx in indices[0]:
            context_results.append({
                'text': self.document_chunks[idx],
                'metadata': self.chunk_metadata[idx]
            })
        
        return context_results
    
    def answer_question(self, 
                        query: str, 
                        context: Optional[str] = None,
                        max_answer_length: int = 100) -> Dict[str, Union[str, List[Dict]]]:
        """
        Generate answer using retrieved or provided context
        
        Args:
            query (str): Input question
            context (str, optional): Specific context to use
            max_answer_length (int): Maximum answer length
        
        Returns:
            Dictionary with answer and context
        """
        # If no context provided, retrieve context
        if context is None:
            contexts = self.retrieve_context(query)
            context = contexts[0]['text']
        else:
            contexts = []
        
        # Generate answer using QA API
        answer = self._qa_inference(query, context)
        
        return {
            'answer': answer,
            'context': context,
            'contexts': contexts
        }
    
    def __call__(self, documents: List[str], query: str) -> Dict[str, Union[str, List[Dict]]]:
        """
        Complete RAG pipeline execution
        
        Args:
            documents (List[str]): List of document texts
            query (str): Input query
        
        Returns:
            Dictionary with answer and context details
        """
        self.preprocess_documents(documents)
        return self.answer_question(query)

def process_files(file_paths: Union[str, List[str]]) -> List[str]:
    """
    Process multiple file types and extract text
    
    Args:
        file_paths (str or List[str]): Path or list of paths to documents
    
    Returns:
        List[str]: Extracted texts from documents
    """
    if isinstance(file_paths, str):
        file_paths = [file_paths]
    
    documents = []
    for file_path in file_paths:
        # Determine file type and extract text
        if file_path.lower().endswith('.pdf'):
            text = DocumentExtractor.extract_text_from_pdf(file_path)
        elif file_path.lower().endswith('.docx'):
            text = DocumentExtractor.extract_text_from_docx(file_path)
        else:
            # Assume it's a text file
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
        
        documents.append(text)
    
    return documents

def main():
    
    
    # Initialize Enhanced RAG Pipeline
    rag_pipeline = HuggingFaceRAGPipeline()
    
    # File paths (replace with your document paths)
    file_paths = ['document1.pdf', 'document2.docx']
    
    # Process documents
    documents = process_files(file_paths)
    
    # Example query
    query = "What are the main points?"
    
    # Get answer
    result = rag_pipeline(documents, query)
    
    # Print results
    print("Answer:", result['answer'])
    print("\nContext:", result['context'])
    print("\nAdditional Contexts:")
    for context in result.get('contexts', []):
        print(context['text'])

if __name__ == "__main__":
    main()