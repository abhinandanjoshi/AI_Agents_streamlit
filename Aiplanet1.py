import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import uuid
import os
import kaggle

# Initialize Qdrant client
qdrant_client = QdrantClient(
    url="https://d44f01e2-4c44-4180-910f-5d7c59c48d9e.us-east4-0.gcp.cloud.qdrant.io:6333", 
    api_key="1ojtpKX2PcRJgfYO4xFYIARv2EACnpFAGPaMyziMUsF4QhqZQty4dg",
)

# Initialize the embedding model (Sentence Transformer or any other model of your choice)
model = SentenceTransformer('all-MiniLM-L6-v2')  # You can use other models for better accuracy

# Browser Agent for Researching Industry/Company Info
class BrowserAgent:
    def __init__(self, industry_keywords=None):
        if industry_keywords is None:
            self.industry_keywords = {
                "Automotive": ["vehicle", "car", "automotive", "engine", "transport", "auto", "mobility", "automobile"],
                "Manufacturing": ["factory", "production", "manufacture", "assembly", "machinery", "plant", "industrial", "phone", "electronics"],
                "Finance": ["bank", "investment", "financial", "stock", "insurance", "capital", "funding", "economy"],
                "Retail": ["store", "shopping", "retail", "e-commerce", "sale", "marketplace", "consumer", "brand", "shop"],
                "Healthcare": ["hospital", "health", "medical", "clinic", "pharma", "biotech", "medicine", "care", "wellness"],
                # More industries can be added here...
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
                        if not company_url.startswith("http"):
                            company_url = "https://www.google.com" + company_url  # Correcting relative URLs
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


# Market Trends and Use Case Generation Agent
class MarketStandardsUseCaseAgent:
    def analyze_industry_trends(self, industry):
        trends = {
            "Automotive": "AI and ML are being leveraged to enhance autonomous driving, predictive maintenance, and vehicle personalization.",
            "Manufacturing": "Industry 4.0 involves the use of AI and robotics for automation, predictive maintenance, and quality control.",
            "Finance": "AI and ML are improving fraud detection, personalized banking services, and algorithmic trading.",
            "Retail": "AI is revolutionizing inventory management, personalized marketing, and customer support through chatbots.",
            "Healthcare": "ML algorithms are being used for early diagnosis, treatment recommendations, and drug discovery.",
        }
        return trends.get(industry, "No data available for this industry.")

    def propose_use_cases(self, company_name, industry):
        use_cases = {
            "Automotive": [
                "Autonomous Vehicles: AI for real-time decision-making, object detection, and path planning; ML for training models on driving data.",
                "Predictive Maintenance: ML for predicting equipment failures.",
                "Personalized Customer Experience: GenAI for tailored recommendations and virtual assistants."
            ],
            "Manufacturing": [
                "Quality Control: AI for visual inspection.",
                "Predictive Maintenance: ML for predicting equipment failures.",
                "Supply Chain Optimization: ML for optimizing logistics and inventory management.",
                "Robotic Process Automation: AI and ML for automating repetitive tasks."
            ],
            # More industries can be added here...
        }
        return use_cases.get(industry, [])


# Resource Asset Collection Agent
class ResourceAssetCollectionAgent:
    def collect_kaggle_resources(self, industry):
        try:
            # List the datasets with a specific search term, sort them by 'hottest'
            datasets = kaggle.api.datasets_list(search=industry, sort_by="hottest", file_type="csv")
            
            resource_links = []
            for dataset in datasets:
                resource_links.append(f"- [Dataset: {dataset.title}]({dataset.url})")
            
            # Save resources in markdown file
            with open("kaggle_resources.md", "w") as f:
                f.write(f"# Kaggle Resources for {industry}\n")
                for link in resource_links:
                    f.write(link + "\n")
            
            return "Kaggle resources have been saved to 'kaggle_resources.md'."
        except Exception as e:
            return f"Error while collecting Kaggle datasets: {str(e)}"
    
    def propose_genai_solutions(self):
        solutions = [
            "Document Search: GenAI-powered search functionality to help find documents quickly.",
            "Automated Report Generation: AI for generating business reports automatically.",
            "AI-Powered Chat Systems: Chatbots for customer support, data retrieval, and interaction."
        ]
        return solutions


# Example usage of agents:
browser_agent = BrowserAgent()
market_standards_agent = MarketStandardsUseCaseAgent()
resource_asset_agent = ResourceAssetCollectionAgent()

# Step 1: Research Industry or Company
industry, company_name, offerings, strategy = browser_agent.run("Tesla")

# Step 2: Analyze Market Trends and Propose Use Cases
trends = market_standards_agent.analyze_industry_trends(industry)
use_cases = market_standards_agent.propose_use_cases(company_name, industry)

# Step 3: Collect Kaggle Resources
kaggle_resources = resource_asset_agent.collect_kaggle_resources(industry)

# Output results
st.write(f"Company: {company_name}")
st.write(f"Industry: {industry}")
st.write("Offerings:")
st.write(offerings)
st.write("Strategic Focus:")
st.write(strategy)
st.write(f"Industry Trends: {trends}")
st.write(f"Proposed Use Cases: {use_cases}")
st.write(kaggle_resources)
