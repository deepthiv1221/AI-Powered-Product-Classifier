# 🧠 AI-Powered Product Classifier

This project uses AI to automatically classify fashion products based on both:
- 🖼️ **Product images** (via computer vision)
- 📝 **Product descriptions** (via natural language processing)

It’s a multimodal AI system built using Python, OpenCV, NLP, and a Random Forest classifier — perfect for e-commerce platforms needing automated catalog tagging.

---

📦 Dataset (Images + Metadata)
Due to GitHub's file size limits, the dataset is not included in this repository.
https://www.kaggle.com/datasets/paramaggarwal/fashion-product-images-small
link to download the dataset

🔽 Download Manually:

📁 After downloading:
Extract images.zip into data/images/
Place styles.csv into data/

Folder structure should look like:
automated-product-classifier/
│
├── app.py
├── product_classifier_rf.pkl
├── tfidf_vectorizer.pkl
├── label_encoder.pkl
├── requirements.txt
└── data/
    ├── images/
    │   ├── 1234.jpg
    │   ├── ...
    └── styles.csv

🧪 Features
✅ Extracts image features using OpenCV

✅ Extracts text features using TF-IDF from product descriptions

✅ Combines both to classify products into categories

✅ Streamlit frontend for real-time predictions

✅ Trained on Myntra product catalog dataset

🛠 Tech Stack
Tool	        Role
Python	      Core programming
Streamlit	    Web app interface
OpenCV	      Image preprocessing
scikit-learn	ML model + vectorizer
pandas	      Data handling
joblib	      Model persistence

🔄 How to Run
# 1. Create a virtual environment (optional but recommended)
python -m venv venv
# Activate:
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the Streamlit app
streamlit run app.py

## 🚀 Live Demo

📍 You can launch the app using Streamlit (locally):

```bash
streamlit run app.py

📷 Sample Output
Upload an image + enter product description
➡️ Model predicts: "Tshirts"
✅ Confidence: 92%


👩‍💻 Author
Deepthi V
GitHub: @deepthiv1221


