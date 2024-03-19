import streamlit as st
import plotly.express as px
import base64

from functions import *




with open("ramadan.gif", "rb") as file:
    img_url = base64.b64encode(file.read()).decode("utf-8")


img_width = 225

st.markdown(
    f'<img src="data:image/jpg;base64,{img_url}" alt="ramadan" style="width: {img_width}px; float: right; margin-right: 32%;">',
    unsafe_allow_html=True,
)


st.title("Sentiment Analysis ❤️ with HuggingFace🤗")


st.divider()


image = 'sentiment.jpg'
st.image(image, use_column_width=True)


st.divider()


st.header("Data Overview🔍")

n_of_pages = st.number_input("Specify the number of pages from which the data will be extracted", value=2, min_value=2, max_value=50)

df = extract(n_of_pages)


drop_null = st.checkbox('Drop Null Values', value=True)

if drop_null:
    df.dropna(inplace=True)


st.write(df)


st.divider()


st.header("Data after applying sentiment analysis❤️")

df_with_inferences = get_inference(df)


filtered_data = df_with_inferences.copy()

selected_sentiment = st.multiselect("Select Sentiment", filtered_data['Sentiment'].unique())

apply_filter = st.button("Apply Filter")

if apply_filter:

    if selected_sentiment:
        filtered_data = filtered_data[filtered_data['Sentiment'].isin(selected_sentiment)]


    st.write(filtered_data)
else:
    st.write(df_with_inferences)


st.divider()


sentiment_counts = filtered_data["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ['Sentiment', 'Count']


fig = px.bar(sentiment_counts, x='Count', y='Sentiment', orientation='h', color='Sentiment',
             title='Sentiment Analysis')
fig.update_layout(xaxis_title='Count', yaxis_title='Sentiment')

st.plotly_chart(fig, theme="streamlit", use_container_width=True)


st.divider()


st.title("Contact Me 📧")

name = "Abdullah Khaled"
email = "dev.abdullah.khaled@gmail.com"
phone = '+201557504902'

st.write(f"Name: {name}")
st.write(f"Email: {email}")
st.write(f"Phone: {phone}")

