import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://d44f01e2-4c44-4180-910f-5d7c59c48d9e.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="1ojtpKX2PcRJgfYO4xFYIARv2EACnpFAGPaMyziMUsF4QhqZQty4dg",
)

# Initialize the embedding model (Sentence Transformer or any other model of your choice)
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models for better accuracy

class BrowserAgent:
    def __init__(self, industry_keywords=None):
        if industry_keywords is None:
            self.industry_keywords = {
                "Automotive": ["vehicle", "car", "automotive", "engine", "transport", "auto", "mobility", "automobile"],
                "Manufacturing": ["factory", "production", "manufacture", "assembly", "machinery", "plant", "industrial", "phone", "electronics"],
                "Finance": ["bank", "investment", "financial", "stock", "insurance", "capital", "funding", "economy"],
                "Retail": ["store", "shopping", "retail", "e-commerce", "sale", "marketplace", "consumer", "brand", "shop"],
                "Healthcare": ["hospital", "health", "medical", "clinic", "pharma", "biotech", "medicine", "care", "wellness"],
                "Software": ["software", "developer", "programming", "IT", "technology", "cloud", "app", "platform", "coding", "database", "SaaS", "tech", "enterprise", "solutions", "services"],
                "Telecommunications": ["telecom", "internet", "mobile", "communication", "network", "broadband", "wireless", "cellular", "5G", "satellite"],
                "Energy": ["energy", "oil", "gas", "renewable", "electricity", "power", "solar", "wind", "nuclear", "battery"],
                "Education": ["education", "school", "college", "university", "learning", "teaching", "training", "courses", "degree", "study"],
                "Entertainment": ["entertainment", "movie", "film", "theater", "music", "gaming", "media", "art", "show"],
                "Hospitality": ["hotel", "restaurant", "resort", "tourism", "travel", "accommodation", "hospitality", "service", "catering"],
                "Real Estate": ["real estate", "property", "housing", "building", "commercial", "residential", "development", "construction", "estate"],
                "Other": []
            }
        else:
            self.industry_keywords = industry_keywords

    def search_for_company_info(self, company_name, retries=3, delay=2):
        if company_name.startswith("http"):
            search_url = company_name
        else:
            search_url = f"https://www.google.com/search?q={urllib.parse.quote(company_name)}"
        
        attempt = 0
        while attempt < retries:
            try:
                response = requests.get(search_url, headers={"User-Agent": "Mozilla/5.0"})
                response.raise_for_status()

                if company_name.startswith("http"):
                    soup = BeautifulSoup(response.text, "html.parser")
                    content = self.extract_relevant_text(soup)
                    company_name = self.extract_company_name(soup)
                else:
                    soup = BeautifulSoup(response.text, "html.parser")
                    link = soup.find("a", href=True)
                    if link:
                        company_url = link["href"]
                        response = requests.get(company_url, headers={"User-Agent": "Mozilla/5.0"})
                        soup = BeautifulSoup(response.text, "html.parser")
                        content = self.extract_relevant_text(soup)
                        company_name = self.extract_company_name(soup)
                    else:
                        content = "No valid URL found for the company."
                        company_name = "Unknown"

                return content, company_name
            except requests.exceptions.RequestException as e:
                attempt += 1
                if attempt >= retries:
                    st.error(f"Error in browsing: {e}")
                    return None, "Unknown"
                time.sleep(delay)  # wait before retrying

    def extract_relevant_text(self, soup):
        paragraphs = soup.find_all("p")
        headers = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
        content = " ".join([para.get_text() for para in paragraphs if para.get_text()])
        content += " ".join([header.get_text() for header in headers if header.get_text()])
        return content

    def extract_company_name(self, soup):
        title = soup.title
        if title:
            title_text = title.get_text()
            return title_text.strip()
        return "Unknown Company"

    def classify_industry(self, content):
        industry_scores = {industry: 0 for industry in self.industry_keywords}

        for industry, keywords in self.industry_keywords.items():
            for keyword in keywords:
                if keyword.lower() in content.lower():
                    industry_scores[industry] += 1

        sorted_industries = sorted(industry_scores.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_industries[0][1] > 0:
            return sorted_industries[0][0]
        return "Other"

    def extract_key_offerings(self, content):
        offerings_keywords = ["service", "product", "solution", "platform", "technology", "feature"]
        strategic_focus_keywords = ["vision", "mission", "goal", "strategy", "focus", "objective"]

        offerings = [sentence for sentence in content.split(". ") if any(word in sentence.lower() for word in offerings_keywords)]
        strategy = [sentence for sentence in content.split(". ") if any(word in sentence.lower() for word in strategic_focus_keywords)]

        return offerings, strategy

    def save_to_qdrant(self, content, company_name):
        # Convert content to embedding vectors
        vectors = model.encode([content])

        # Create a valid UUID for the point ID
        point_id = str(uuid.uuid4())  # Generates a unique UUID

        # Save to Qdrant
        qdrant_client.upsert(
            collection_name="company_info", 
            points=[{
                "id": point_id,  # Use UUID as the point ID
                "vector": vectors[0],
                "payload": {"company_name": company_name, "content": content}
            }]
        )

    def run(self, company_name):
        content, company_name = self.search_for_company_info(company_name)
        if content:
            industry = self.classify_industry(content)
            offerings, strategy = self.extract_key_offerings(content)

            # Save the content and offerings to Qdrant
            self.save_to_qdrant(content, company_name)
            
            return industry, company_name, offerings, strategy
        else:
            return "Unknown", "Unknown", [], []

    def retrieve_similar_info(self, query):
        query_vector = model.encode([query])

        # Search Qdrant for the most similar content
        results = qdrant_client.search(
            collection_name="company_info",
            query_vector=query_vector[0],
            limit=3  # Number of similar results to retrieve
        )

        similar_info = []
        for result in results:
            similar_info.append(result.payload['content'])
        
        return similar_info


# Streamlit app
def main():
    st.title("Company Industry Classification and Key Offerings Agent")

    # Input field for the company name
    company_name = st.text_input("Enter the company's name or website URL:")

    # Instantiate the agent
    agent = BrowserAgent()

    if st.button("Classify Industry and Extract Offerings"):
        if company_name:
            industry, company_name, offerings, strategy = agent.run(company_name)
            st.write(f"**Company Name**: {company_name}")
            st.write(f"**Classified Industry**: {industry}")
            st.write("**Key Offerings**:")
            for offering in offerings:
                st.write(f"- {offering}")
            st.write("**Strategic Focus Areas and Vision**:")
            for strat in strategy:
                st.write(f"- {strat}")
        else:
            st.warning("Please enter a valid company name or website URL.")

if __name__ == "__main__":
    main()
