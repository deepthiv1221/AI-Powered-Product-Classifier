import streamlit as st
import joblib, cv2, numpy as np

# Load saved files
model = joblib.load('product_classifier_rf.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')
encoder = joblib.load('label_encoder.pkl')  # Load the saved LabelEncoder

# Title of the app
st.title("📦 AI-Powered Product Classifier")

# Upload file (image) and input description
uploaded_file = st.file_uploader("Upload a product image...", type=["jpg", "png"])
description = st.text_input("Enter product description:")

if uploaded_file and description:
    # Read image bytes
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    # Preprocess image (match your training size)
    img = cv2.resize(img, (32, 32)) / 255.0
    img_feat = img.flatten().reshape(1, -1)
    
    # Preprocess text
    txt_feat = vectorizer.transform([description]).toarray()
    
    # Combine image and text features
    features = np.hstack([txt_feat, img_feat])
    pred = model.predict(features)[0]
    
    # Decode the predicted label back to the category name
    decoded_pred = encoder.inverse_transform([pred])[0]
    
    # Calculate prediction confidence (optional, using predict_proba)
    probs = model.predict_proba(features)[0]
    top_idx = np.argmax(probs)
    confidence = probs[top_idx]  # Confidence of the top prediction

    # Create a layout with columns
    col1, col2 = st.columns(2)
    with col1:
        # Display uploaded image
        st.image(img, caption="Uploaded Image", use_container_width=True)

    with col2:
        # Display prediction and confidence
        st.write(f"**Predicted Category:** {decoded_pred}")
        st.write(f"**Confidence:** {confidence:.2%}")
