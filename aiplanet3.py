import streamlit as st
import requests
from bs4 import BeautifulSoup
import urllib.parse
import time
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer
import kaggle

# Qdrant client setup
qdrant_client = QdrantClient(
    url="https://d44f01e2-4c44-4180-910f-5d7c59c48d9e.us-east4-0.gcp.cloud.qdrant.io:6333",
    api_key="1ojtpKX2PcRJgfYO4xFYIARv2EACnpFAGPaMyziMUsF4QhqZQty4dg",
)

# Initialize the embedding model (Sentence Transformer or any other model of your choice)
model = SentenceTransformer('all-MiniLM-L6-v2')

# Define Browser Agent Class
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

# Define Market Standards & Use Case Generation Agent Class
class MarketStandardsUseCaseAgent:
    def analyze_industry_trends(self, industry):
        # Simulate fetching industry trends based on the industry
        return f"Trends for {industry} industry."

    def propose_use_cases(self, industry):
        # Simulate proposing use cases based on the industry
        return [
            f"Use Case 1 for {industry}",
            f"Use Case 2 for {industry}",
            f"Use Case 3 for {industry}",
        ]

# Define Resource Asset Collection Agent Class
class ResourceAssetCollectionAgent:
    def collect_kaggle_resources(self, industry):
        try:
            datasets = kaggle.api.dataset_list(search=industry)
            resource_links = []
            for dataset in datasets:
                dataset_title = dataset.title
                dataset_url = f"https://www.kaggle.com/datasets/{dataset.ref}"
                resource_links.append(f"- [{dataset_title}]({dataset_url})")

            markdown_content = f"# Kaggle Resources for {industry}\n\n" + "\n".join(resource_links)

            with open("kaggle_resources.md", "w", encoding="utf-8") as f:
                f.write(markdown_content)

            st.markdown(f"[Download Kaggle Resources Markdown](kaggle_resources.md)")

            return markdown_content
        except Exception as e:
            return f"Error collecting Kaggle datasets: {str(e)}"

# Initialize agents
browser_agent = BrowserAgent()
market_standards_agent = MarketStandardsUseCaseAgent()
resource_asset_agent = ResourceAssetCollectionAgent()

# Streamlit App
st.title("Multi-Agent Architecture for Market Research & Use Case Generation")

# Step 1: Browser Agent - Company Research
st.subheader("1. Browser Agent - Company Research")
company_name_input = st.text_input("Enter a company name:")

if st.button("Run Browser Agent"):
    if "company_output" not in st.session_state:
        content, company_name = browser_agent.search_for_company_info(company_name_input)
        industry = browser_agent.classify_industry(content)

        st.session_state["company_output"] = {"company_name": company_name, "industry": industry, "content": content}

if "company_output" in st.session_state:
    company_output = st.session_state["company_output"]
    st.write(f"Company Name: {company_output['company_name']}")
    st.write(f"Industry: {company_output['industry']}")
    st.write(company_output["content"])

# Step 2: Market Standards & Use Case Generation Agent
st.subheader("2. Market Standards & Use Case Generation Agent")

if "company_output" in st.session_state:
    if st.button("Run Market Standards & Use Case Agent") and "market_output" not in st.session_state:
        industry_trends = market_standards_agent.analyze_industry_trends(st.session_state["company_output"]["industry"])
        use_cases = market_standards_agent.propose_use_cases(st.session_state["company_output"]["industry"])
        st.session_state["market_output"] = {"industry_trends": industry_trends, "use_cases": use_cases}

    if "market_output" in st.session_state:
        market_output = st.session_state["market_output"]
        st.write("Industry Trends:")
        st.write(market_output["industry_trends"])
        st.write("Proposed Use Cases:")
        st.write("\n".join(market_output["use_cases"]))

# Step 3: Resource Asset Collection Agent
st.subheader("3. Resource Asset Collection Agent")

if "company_output" in st.session_state:
    if st.button("Run Resource Asset Collection Agent") and "resource_output" not in st.session_state:
        kaggle_data = resource_asset_agent.collect_kaggle_resources(st.session_state["company_output"]["industry"])
        st.session_state["resource_output"] = kaggle_data

    if "resource_output" in st.session_state:
        st.write(st.session_state["resource_output"])
