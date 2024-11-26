import streamlit as st
import qdrant_client
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import numpy as np
from qdrant_client import QdrantClient

from sklearn.metrics.pairwise import cosine_similarity

class Agent2:
    def __init__(self, qdrant_client, model_name="all-MiniLM-L6-v2"):
        # Initialize Qdrant client and the transformer model for semantic search
        self.client = qdrant_client
        self.model = SentenceTransformer(model_name)

    def save_content_to_qdrant(self, content, company_name):
        """
        Save the content into Qdrant for later retrieval.
        """
        # Convert the content to embeddings
        content_embeddings = self.model.encode(content)

        # Create or get the collection for the company
        collection = self.client.get_collection(name=company_name)

        # Save the embeddings to Qdrant
        self.client.upload_collection(
            collection_name=company_name,
            vectors=content_embeddings,
            payload=[{"company": company_name}],
            ids=[str(i) for i in range(len(content))]
        )

    def query_content(self, query):
        """
        Perform semantic search using the query.
        """
        # Retrieve all collections
        collections_response = self.client.get_collections()
        collections = collections_response.collections  # Accessing collections using the attribute

        results = []

        # Iterate over collections and perform semantic search
        for collection_name in collections:
            collection = self.client.get_collection(name=collection_name)
            
            # Get all documents in the collection
            documents = collection.get(include=["documents", "embeddings"])["documents"]
            embeddings = collection.get(include=["embeddings"])["embeddings"]

            # Convert the query to an embedding
            query_embedding = self.model.encode(query)
            
            # Compute cosine similarities between the query and stored embeddings
            similarities = cosine_similarity([query_embedding], embeddings)[0]
            
            # Get the most relevant document
            most_relevant_index = np.argmax(similarities)
            most_relevant_document = documents[most_relevant_index]
            
            # Store the result along with the relevance score
            results.append({
                "company": collection_name,
                "document": most_relevant_document,
                "similarity_score": similarities[most_relevant_index]
            })

        return results

    def analyze_industry_trends_and_standards(self, industry):
        """
        Analyze industry trends and standards within the company's sector related to AI, ML, and automation.
        """
        # Placeholder for actual industry analysis logic.
        industry_trends = {
            "Automotive": "AI is revolutionizing the automotive industry with autonomous driving, predictive maintenance, and smart manufacturing.",
            "Manufacturing": "AI and ML are optimizing production lines, enabling predictive maintenance, and enhancing supply chain management.",
            "Finance": "ML algorithms are being used for fraud detection, investment strategies, and customer service chatbots.",
            "Retail": "Retailers are leveraging AI to optimize inventory, enhance customer experiences, and offer personalized product recommendations.",
            "Healthcare": "AI is being used in healthcare for diagnostics, drug discovery, and personalized treatment plans."
        }
        return industry_trends.get(industry, "Industry trends not available.")

    def propose_use_cases(self, company_info, industry):
        """
        Propose relevant use cases where the company can leverage GenAI, LLMs, and ML technologies.
        """
        use_cases = {
            "Automotive": [
                "Implementing predictive maintenance using machine learning to reduce downtime.",
                "Leveraging AI for autonomous driving features and driver assistance systems.",
                "Using AI to optimize production and reduce energy consumption in manufacturing."
            ],
            "Manufacturing": [
                "Using machine learning to predict equipment failures and reduce maintenance costs.",
                "Implementing AI for quality control and defect detection during production.",
                "Using automation and AI to streamline supply chain and inventory management."
            ],
            "Finance": [
                "Utilizing AI for fraud detection and prevention in transactions.",
                "Leveraging ML for credit risk assessment and personalized financial products.",
                "Implementing chatbots for 24/7 customer service and support."
            ],
            "Retail": [
                "Using AI for personalized product recommendations based on customer data.",
                "Implementing machine learning for dynamic pricing based on demand and inventory.",
                "Using AI to optimize supply chains and reduce operational costs."
            ],
            "Healthcare": [
                "Leveraging AI for diagnostics and personalized treatment recommendations.",
                "Using machine learning for drug discovery and clinical trial optimization.",
                "Implementing AI for remote patient monitoring and telemedicine services."
            ]
        }

        return use_cases.get(industry, ["No specific use cases identified."])

    def run(self, company_name, query):
        """
        Main function to handle both tasks: Industry analysis and use case generation.
        """
        # Query content stored in Qdrant
        search_results = self.query_content(query)
        
        if search_results:
            # Show the most relevant content to the user
            for result in search_results:
                st.write(f"**Company:** {result['company']}")
                st.write(f"**Content:** {result['document']}")
                st.write(f"**Relevance Score:** {result['similarity_score']:.2f}")
        else:
            st.warning("No relevant content found in the database.")
        
        # Analyze industry trends and standards based on the company
        industry = "Automotive"  # Placeholder: You can dynamically get the industry based on content
        industry_trends = self.analyze_industry_trends_and_standards(industry)
        st.write(f"### Industry Trends for {industry}:")
        st.write(industry_trends)

        # Propose use cases based on the company's industry
        use_cases = self.propose_use_cases(company_name, industry)
        st.write(f"### Proposed Use Cases for {company_name}:")
        for use_case in use_cases:
            st.write(f"- {use_case}")

# Streamlit chatbot UI for interaction
def main():
    st.title("Agent 2: Market Standards & Use Case Generation")
    
    # Input fields
    company_name = st.text_input("Enter the company's name:")
    user_query = st.text_input("Ask a question about the company or industry:")
    
    # Initialize Qdrant client
    client = QdrantClient(
    url="https://d44f01e2-4c44-4180-910f-5d7c59c48d9e.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="Xz9uw1Rczffbut96Qjmh1pVe0UUh8FAo2OdWziNtQEvmkAlcAOSQ3A",
)
    
    # Instantiate Agent 2
    agent = Agent2(client)
    
    # Button to handle search and analysis
    if st.button("Analyze and Generate Use Cases"):
        if company_name and user_query:
            # Run Agent 2 tasks
            agent.run(company_name, user_query)
        else:
            st.warning("Please provide both a company name and a query.")

if __name__ == "__main__":
    main()

