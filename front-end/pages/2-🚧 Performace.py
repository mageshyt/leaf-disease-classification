import streamlit as st
import pandas as pd

st.set_page_config(page_title="Performance Measures", layout="wide")
st.title('Performance Measures 🚧')
st.image('https://miro.medium.com/v2/resize:fit:1400/1*g9x71-cqZ5im3EaFxwyf9Q.jpeg')
# load the performance metrics df

performance_df = pd.read_csv("./assets/csv/performance_metrics.csv")

# 1. Precision, Recall, and F1 Score of the Machine Learning Models

st.subheader(
    "1. Precision, Recall, and F1 Score of the Machine Learning Models")

# Display the data-frame
st.dataframe(performance_df)

# plot the performance

st.write("plot the performance ")

st.bar_chart(performance_df, x='Model', y=[
             'Accuracy', 'Precision', 'Recall', 'F1 Score'])


st.markdown("**Note:** The model with the highest F1 score is the best model.")


st.success("Based on the evaluation results for 10% of the data, it is evident that MobileNet_v2 outperforms the other models in terms of accuracy, precision, recall, and F1 score. Here's a brief conclusion:.", icon="🔥")

st.divider()
# explain the results

st.subheader("2. Explain the results")


# Provide explanations for the performance of each model

# 1. MobileNet_v2 Model
st.markdown("##### 1. MobileNet_v2 Model :-  ")

st.write(" MobileNet_v2, a lightweight convolutional neural network, demonstrates excellent performance "
         "due to its efficiency and ability to capture important features in leaf images.")

# plot the performance of  MobileNet_v2

st.write("plot the performance of  MobileNet_v2")
mobile_net_model = performance_df.loc[performance_df['Model']
                                      == 'MobileNet_v2']
st.write(mobile_net_model)
columns = ['Accuracy', 'Precision', 'Recall', 'F1 Score']


st.bar_chart(mobile_net_model.iloc[0][columns])


# 1.Resnet50v2 Model
st.markdown("##### 2. Resnet50v2 Model :-  ")

st.write("Resnet50v2 also performs well, but it may be slightly heavier compared to MobileNet_v2.")


st.write("plot the performance of  Resnet50v2")

resnet50v2_model = performance_df.loc[performance_df['Model'] == 'Resnet50v2']

st.write(resnet50v2_model)

# plot the performance of  Resnet50v2

st.bar_chart(resnet50v2_model.iloc[0][columns])


# 3. EfficientNet Model
st.markdown("##### 3. EfficientNet Model:")
st.write("EfficientNet, known for its scalability, achieves high accuracy with less computational complexity.")

# Plot the performance of EfficientNet
efficientnet_model = performance_df.loc[performance_df['Model']
                                        == 'EfficientNet']
st.write(efficientnet_model)

# Bar chart for EfficientNet performance
st.bar_chart(efficientnet_model.iloc[0][columns])

# 4. OwnCnn_model
st.markdown("##### 4. OwnCnn_model:")
st.write("The custom CNN model shows moderate performance. Fine-tuning and additional training may improve results.")

# Plot the performance of OwnCnn_model
own_cnn_model = performance_df.loc[performance_df['Model'] == 'OwnCnn_model']
st.write(own_cnn_model)

# Bar chart for OwnCnn_model performance
st.bar_chart(own_cnn_model.iloc[0][columns])

# 5. non_cnn_model
st.markdown("##### 5. non_cnn_model:")
st.write("The non-CNN model lags behind significantly, indicating the importance of convolutional neural networks for image classification tasks.")

# Plot the performance of non_cnn_model
non_cnn_model = performance_df.loc[performance_df['Model'] == 'non_cnn_model']
st.write(non_cnn_model)

# Bar chart for non_cnn_model performance
st.bar_chart(non_cnn_model.iloc[0][columns])

st.markdown("**🌐 Conclusion:** Based on these findings, it is recommended to proceed with training MobileNet_v2 on the full-scale dataset, as it consistently outperforms other models.")
