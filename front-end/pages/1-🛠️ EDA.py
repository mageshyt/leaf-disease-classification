import pandas as pd
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns

# import pandas_profiling as pp
# from ydata_profiling import ProfileReport

# from streamlit_pandas_profiling import st_profile_report


st.info("This is a demo application written to show how to our deeplearing model can be used to predict the disease of a leaf.")

st.title("Explore Data Analysis 📊")

st.image("https://pianalytix.com/wp-content/uploads/2020/11/Exploratory-Data-Analysis.jpg")

leaf_disease_df = pd.read_csv("./assets/csv/LeafDisease-2.csv")

# Display the dataframe
st.data_editor(leaf_disease_df)

# 1.we will show much plant has disease and how much plant is healthy
st.subheader(
    "1.we will show much plant has disease and how much plant is healthy")

leaf_type = leaf_disease_df['leaf_type'].value_counts()

st.write(leaf_type)
# plot them
st.bar_chart(leaf_type)

st.divider()

# 2 . we will see the class distribution of the dataset
st.subheader("2.we will see the class distribution of the dataset")

st.write(leaf_disease_df['leaf_family'].value_counts())
# profile = ProfileReport(leaf_disease_df, title="Profiling Report")


# plot the class distribution of the dataset

st.write("plot the class distribution of the dataset")

st.bar_chart(leaf_disease_df['leaf_family'].value_counts())

st.markdown("**Note:** The Dataset we chosen is correctly balanced so we don't need to do any balancing techniques")
st.divider()
# st_profile_report(profile)

# 3. Family and Disease Relationship:

st.subheader("3. Family and Disease Relationship:")

family_disease_df = leaf_disease_df.groupby(
    ['leaf_family', 'leaf_type']).size().reset_index(name='counts')

st.write(family_disease_df)
st.divider()


# Which disease is the most common in the dataset?

st.subheader("4. Which disease is the most common in the dataset?")
most_common_disease = leaf_disease_df[leaf_disease_df['leaf_type']
                                      == 'diseased']['leaf_family'].value_counts().idxmax()


st.write()
# make a pie chart using plt

# Set Seaborn style
sns.set(style="whitegrid")

# Create a pie chart using Seaborn
plt.figure(figsize=(10, 10))
leaf_family_counts = leaf_disease_df[leaf_disease_df['leaf_type']
                                     == 'diseased']['leaf_family'].value_counts()
sns.set_palette("pastel")  # Set a pastel color palette
plt.pie(leaf_family_counts, labels=leaf_family_counts.index,
        autopct='%1.1f%%', startangle=140)
plt.title("Disease Distribution")

# Display the pie chart using Streamlit
st.pyplot(plt)

st.markdown(
    f"**{most_common_disease}** is the most common disease in the dataset")

st.divider()
