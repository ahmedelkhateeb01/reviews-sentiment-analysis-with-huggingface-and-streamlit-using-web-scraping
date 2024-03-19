import requests
import pandas as pd
import streamlit as st

from bs4 import BeautifulSoup
from transformers import pipeline



@st.cache_data()
def extract(page_num):

    all_review_data = []

    for page in range(1, page_num + 1):
        url = f"https://www.trustpilot.com/review/listwithclever.com?page={page}"
        response = requests.get(url)

        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            
            review_elements = soup.find_all("div", class_="styles_cardWrapper__LcCPA styles_show__HUXRb styles_reviewCard__9HxJJ")
            
            for review in review_elements:

                reviewer_name_elem = review.find("span", class_="typography_heading-xxs__QKBS8")
                reviewer_name = reviewer_name_elem.text.strip() if reviewer_name_elem else ""
                
                reviewer_reviews_count_elem = review.find("span", class_="typography_body-m__xgxZ_")
                reviewer_reviews_count = reviewer_reviews_count_elem.text.strip() if reviewer_reviews_count_elem else ""
                
                reviewer_country_elem = review.find("div", class_="typography_body-m__xgxZ_ typography_appearance-subtle__8_H2l styles_detailsIcon__Fo_ua").find("span")
                reviewer_country = reviewer_country_elem.text.strip() if reviewer_country_elem else ""
                
                review_header_elem = review.find("div", class_="styles_reviewHeader__iU9Px")
                review_rating = int(review_header_elem['data-service-review-rating']) if review_header_elem else None
                
                review_title_elem = review.find("h2", class_="typography_heading-s__f7029")
                review_title = review_title_elem.text.strip() if review_title_elem else ""
                
                review_text_elem = review.find("p", class_="typography_body-l__KUYFJ")
                review_text = review_text_elem.text.strip() if review_text_elem else ""
                
                date_of_experience_elem = review.find("p", class_="typography_body-m__xgxZ_")
                date_of_experience = date_of_experience_elem.text.strip().replace("Date of experience:", "") if date_of_experience_elem else ""
                
                
                all_review_data.append({"Reviewer Name": reviewer_name,
                                        "Reviewer's Reviews Count": reviewer_reviews_count,
                                        "Reviewer Country": reviewer_country,
                                        "Review Rating": review_rating,
                                        "Review Title": review_title,
                                        "Review Text": review_text,
                                        "Date of Experience": date_of_experience})
        else:
            print(f"Failed to fetch page {page_num}. Status code:", response.status_code)

            
    df = pd.DataFrame(all_review_data)

    return df



@st.cache_resource()
def get_inference(df):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)

    with st.spinner("Analyzing sentiment..."):
        sentiments = []
        for review in df["Review Text"]:
            result = classifier(review)[0][0]['label']
            sentiments.append(result)

    df["Sentiment"] = sentiments

    return df