import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from sentence_transformers import SentenceTransformer
import qdrant_client
from qdrant_client.models import VectorParams, Distance

class BrowserAgent1:
    def __init__(self, industry_keywords=None):
        # Define industry keywords for classification purposes
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

        # Initialize embedding model for generating vectors
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

        # Set up Qdrant client
        self.qdrant_client = qdrant_client.QdrantClient("localhost")  # Change to the actual Qdrant server URL if hosted remotely
        self.collection_name = "company_info"
        
        # Create a Qdrant collection if it doesn't exist
        self._create_collection()

    def _create_collection(self):
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vector_params=VectorParams(size=384, distance=Distance.COSINE),  # Adjust the vector size to match SentenceTransformer output
            )
        except Exception as e:
            print(f"Collection already exists or error: {e}")

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

                # If it's a valid company URL, parse its content directly
                if company_name.startswith("http"):
                    soup = BeautifulSoup(response.text, "html.parser")
                    content = self.extract_relevant_text(soup)
                    company_name = self.extract_company_name(soup)
                else:
                    # If it's a search result, find the URL and scrape the content
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
                time.sleep(delay)

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

    def extract_offerings_and_strategy(self, content):
        offerings_keywords = ["service", "product", "solution", "platform", "technology", "feature"]
        strategic_focus_keywords = ["vision", "mission", "goal", "strategy", "focus", "objective"]

        offerings = [sentence for sentence in content.split(". ") if any(word in sentence.lower() for word in offerings_keywords)]
        strategy = [sentence for sentence in content.split(". ") if any(word in sentence.lower() for word in strategic_focus_keywords)]

        return offerings, strategy

    def store_in_qdrant(self, content, company_name):
        vector = self.embedding_model.encode(content)
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=[
                {
                    "id": company_name,  # Use company_name as the point ID
                    "vector": vector.tolist(),
                    "payload": {"company_name": company_name}
                }
            ]
        )

    def run(self, company_name):
        content, company_name = self.search_for_company_info(company_name)
        if content:
            self.store_in_qdrant(content, company_name)
            industry = self.classify_industry(content)
            offerings, strategy = self.extract_offerings_and_strategy(content)
            return industry, company_name, offerings, strategy
        else:
            return "Unknown", "Unknown", [], []

# Streamlit app
def main():
    st.title("Agent 1: Company Information & Industry Classification")

    company_name = st.text_input("Enter the company's name or website URL:")
    agent = BrowserAgent1()

    if st.button("Classify Industry and Extract Information"):
        if company_name:
            industry, company_name, offerings, strategy = agent.run(company_name)
            st.write(f"*Company Name*: {company_name}")
            st.write(f"*Classified Industry*: {industry}")
            st.write("*Key Offerings:*")
            for offering in offerings:
                st.write(f"- {offering}")
            st.write("*Strategic Focus Areas:*")
            for focus in strategy:
                st.write(f"- {focus}")
        else:
            st.warning("Please enter a valid company name or website URL.")

if __name__ == "__main__":
    main()
