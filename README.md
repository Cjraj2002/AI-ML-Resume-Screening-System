# AI-ML-Resume-Screening-System

🧑‍💻 AI Resume Tracker (OCR + ML)

	This project extracts text from resumes using Tesseract OCR, processes it with NumPy, and applies a Machine Learning model to predict whether a candidate is a "Good Fit" or "Not Fit" for a given role.

	Built with Streamlit for an interactive UI, this tool demonstrates the integration of OCR + NLP preprocessing + ML classification in a real-world HR use case.

📂 Project Structure

	resume_tracker/
	│── app.py              # Streamlit UI + backend logic
	│── model.py      # Script to train ML model
	│── model.pkl           # Saved ML model
	│── utils.py            # OCR + preprocessing functions
	│── requirements.txt    # Python dependencies
	│── sample_resumes/     # Example resumes (training + testing)

⚡ Features

	📄 Resume Upload – Upload resumes in image/PDF format.

	🔎 OCR Extraction – Extracts text using Tesseract OCR.

	🧹 Text Preprocessing – Cleans and tokenizes resume content with NumPy + custom pipeline.

	🤖 ML Model Prediction – Uses Logistic Regression / Linear Regression to classify as:

	✅ Good Fit

	❌ Not Fit

	📊 Confidence Score – Displays prediction probability.

	🎨 Streamlit UI – Simple, user-friendly interface.


🛠️ Tech Stack

	Languages: Python

	Libraries: NumPy, Pandas, Scikit-learn, PyTesseract, Streamlit

	ML Models: Logistic Regression, Linear Regression (extendable to others)

	Tools: Git, VS Code, Jupyter Notebook


⚙️ Installation

1. Clone the repo

 	git clone https://github.com/yourusername/AI-ML-Resume-Screening-System.git
	cd AI-ML-Resume-Screening-System

2. Create virtual environment (optional but recommended)
	python3 -m venv venv
	source venv/bin/activate   # for Linux/Mac
	venv\Scripts\activate      # for Windows

3. Install dependencies
	pip install -r requirements.txt

4. Run the Streamlit app
	streamlit run app.py

5. 📊 Model Training

	Place training resumes inside sample_resumes/.
	python model_train.py

🧪 Example Usage

	Upload a resume (JPEG/PDF).

	OCR extracts skills like Python, Django, SQL, ML.

	Model applies weights → predicts Good Fit (85%) or Not Fit (40%).

🚧 Future Improvements

	Add support for PDF parsing (PyPDF2, pdfplumber).

	Extend ML model with NLP embeddings (TF-IDF, BERT).

	Build Dashboard for recruiters to track multiple candidates.

	Deploy on Streamlit Cloud / AWS / Azure.


👨‍💻 Author

	Chelin Jebastin Raj
	📍 Bengaluru, India
