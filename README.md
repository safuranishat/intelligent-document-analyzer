# 🧠 DocuMind – Intelligent Document Analyzer

### Understand Documents Beyond Their Words.

[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Framework-Streamlit-red.svg)](https://streamlit.io/)
[![NLP](https://img.shields.io/badge/Domain-Natural%20Language%20Processing-purple.svg)](https://en.wikipedia.org/wiki/Natural_language_processing)
[![Status](https://img.shields.io/badge/Status-Deployed-success.svg)](https://intelligent-document-analyzer-ec6vseuj2xmyk6qpb4ozam.streamlit.app/)

## 🌐 Live Application

**Try DocuMind here:**

https://intelligent-document-analyzer-ec6vseuj2xmyk6qpb4ozam.streamlit.app/

---

## 📖 About the Project

Documents contain valuable information, but understanding them is not always easy.

Reading lengthy documents, identifying important information, finding relationships between words, recognizing entities, and understanding sentence structures can take considerable time and effort.

What if we could use Natural Language Processing to make this process easier?

That question inspired **DocuMind – Intelligent Document Analyzer**.

DocuMind is an NLP-powered web application that analyzes uploaded documents and provides meaningful insights into their structure, language, grammar, syntax, and semantics.

It transforms unstructured document text into organized and understandable information through multiple NLP techniques.

---

## 🎓 Academic Background

Natural Language Processing (NLP) is one of the subjects included in our **B.Tech Computer Science and Engineering – 3rd Year curriculum**.

As part of learning this subject, we were assigned a practical project to implement and demonstrate the concepts and techniques covered in our coursework.

Instead of implementing individual NLP techniques as separate programs, I wanted to bring multiple concepts together into one functional application.

This led to the development of DocuMind.

The project provided an opportunity to understand how theoretical NLP concepts can be applied to a practical problem through Python, NLP libraries, data visualization, and web application development.

---

## 🚨 Problem Statement

Understanding large and unstructured documents manually can be time-consuming and challenging.

Some common difficulties include:

- Identifying important information from lengthy documents.
- Understanding the grammatical structure of sentences.
- Recognizing people, organizations, locations, dates, and other entities.
- Identifying frequently occurring words and phrases.
- Understanding relationships between different words in a sentence.
- Determining the intended meaning of ambiguous words based on context.
- Extracting meaningful information without reading the entire document repeatedly.

### Proposed Solution

DocuMind addresses these challenges by providing a centralized document analysis platform that applies different Natural Language Processing techniques to extract, organize, and visualize linguistic information.

Users can upload a document and explore its analysis through an interactive dashboard.

---

## 🎯 Project Objectives

The main objectives of DocuMind are:

1. To automate the extraction of text from different document formats.
2. To analyze document structure and identify important sections.
3. To perform linguistic and morphological analysis.
4. To identify named entities using Named Entity Recognition.
5. To understand grammatical relationships through dependency parsing.
6. To analyze word patterns using N-gram language models.
7. To generate an extractive summary of the document.
8. To estimate document readability.
9. To extract basic semantic relationships.
10. To demonstrate Word Sense Disambiguation using contextual information.

---

## ✨ Key Features

### 1. 📄 Multi-Format Document Upload
- Upload PDF, DOCX, and TXT files.
- Extract text automatically.
- View the complete extracted document.
- Download extracted text.

### 2. 🏗️ Document Structure Analysis
- Detect document headings.
- Identify paragraphs.
- Display paragraph information.
- Explore the extracted document structure.

### 3. 🔤 Linguistic Analysis
- Perform Part-of-Speech (POS) tagging.
- Visualize POS distribution through interactive charts.
- Display full forms and meanings of POS tags.
- Perform morphological analysis.
- Extract frequently occurring keywords.

### 4. 🏷️ Named Entity Recognition (NER)
Identify important entities such as:
- Person names
- Organizations
- Locations
- Dates
- Money
- Other recognized entity categories

Display recognized entities in organized tables and charts.

### 5. 🌳 Syntax Analysis
- Perform dependency parsing.
- Visualize grammatical relationships between words.
- Identify the root word of a sentence.
- Display token-level grammatical information.
- Analyze parts of speech and dependency labels.

### 6. 📚 N-gram Language Models

Explore word patterns through three types of N-grams:

- **Unigram:** Analysis of individual words.
- **Bigram:** Analysis of consecutive two-word combinations.
- **Trigram:** Analysis of consecutive three-word combinations.

Also includes word frequency analysis and visualizations.

### 7. 📝 Smart Summary & Readability
- Generate an extractive summary using word-frequency-based sentence scoring.
- Identify important sentences.
- Calculate an estimated Flesch Reading Ease score.
- Provide a basic interpretation of document readability.

### 8. 🧠 Semantic Analysis
- Extract basic Subject–Action–Object relationships.
- Identify sentence subjects.
- Identify actions and objects.
- Display extracted semantic relationships in a structured table.

### 9. 🧩 Word Sense Disambiguation (WSD)
- Select a word from the uploaded document.
- Analyze the sentence in which the word appears.
- Identify a possible meaning using the Lesk algorithm.
- Retrieve definitions and examples from WordNet.
- Display related synonyms.

---

## ⚙️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| Streamlit | Interactive web application |
| spaCy | NLP processing, POS tagging, NER and dependency parsing |
| NLTK | Word Sense Disambiguation and WordNet |
| PyMuPDF | PDF text extraction |
| python-docx | DOCX text extraction |
| Pandas | Data processing and tabular representation |
| Plotly | Interactive data visualizations |
| Regular Expressions | Text processing and pattern detection |

---

## 🔄 How the Application Works

The application follows a simple document-processing workflow:

**Step 1 – Document Upload**

The user uploads a PDF, DOCX, or TXT document.

**Step 2 – Text Extraction**

The application extracts readable text from the uploaded document.

**Step 3 – NLP Processing**

The extracted text is processed using spaCy and NLTK.

**Step 4 – Linguistic Analysis**

Different NLP techniques are applied to identify grammatical, structural, and linguistic patterns.

**Step 5 – Semantic Analysis**

The application explores basic relationships and contextual word meanings.

**Step 6 – Visualization**

The results are presented through interactive tables, charts, and visual representations.

**Step 7 – Export**

Users can download extracted text and an analysis report.

---

## 🏛️ System Architecture

```text
             USER
               |
               v
       DOCUMENT UPLOAD
        PDF / DOCX / TXT
               |
               v
        TEXT EXTRACTION
               |
               v
       NLP PROCESSING ENGINE
          /          \
         /            \
        v              v
      spaCy           NLTK
        |               |
        v               v
  Linguistic         Word Sense
   Analysis         Disambiguation
        |
        v
  +---------------------------+
  |     DOCUMENT ANALYSIS     |
  +---------------------------+
  | Document Structure        |
  | POS Tagging               |
  | Named Entity Recognition  |
  | Dependency Parsing        |
  | N-gram Analysis           |
  | Smart Summary             |
  | Readability Analysis      |
  | Semantic Analysis         |
  +---------------------------+
               |
               v
       INTERACTIVE DASHBOARD
               |
               v
        RESULTS & REPORT
```

---

## 🚀 Getting Started

Follow these steps to run DocuMind locally.

### Prerequisites

- Python 3.12
- VS Code or another Python IDE
- Internet connection for initial package installation

### 1. Clone the Repository

```bash
git clone https://github.com/safuranishat/intelligent-document-analyzer.git
```

### 2. Navigate to the Project Folder

```bash
cd intelligent-document-analyzer
```

### 3. Create a Virtual Environment

```bash
python3 -m venv venv
```

### 4. Activate the Virtual Environment

For Linux:

```bash
source venv/bin/activate
```

For Windows:

```bash
venv\Scripts\activate
```

### 5. Install Dependencies

```bash
pip install -r requirements.txt
```

### 6. Download the spaCy English Language Model

```bash
python -m spacy download en_core_web_sm
```

### 7. Download NLTK WordNet Resources

```bash
python -m nltk.downloader wordnet omw-1.4
```

### 8. Run the Application

```bash
python -m streamlit run app.py
```

The application will open in your browser.

---

## 📂 Project Structure

```text
intelligent-document-analyzer/
│
├── app.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Project dependencies
│
└── README.md
    └── Project documentation
```

---

## 🌍 Deployment

DocuMind is deployed using **Streamlit Community Cloud**.

The application is hosted online and can be accessed through a public URL without requiring users to install Python or configure a local development environment.

**Live Application:**

https://intelligent-document-analyzer-ec6vseuj2xmyk6qpb4ozam.streamlit.app/

---

## ⚠️ Current Scope and Limitations

DocuMind demonstrates several practical NLP techniques, but some of its outputs are simplified implementations.

- The generated summary is extractive and uses word-frequency-based sentence scoring.
- Readability is estimated using a simplified syllable-counting approach.
- Semantic relationship extraction is dependency-based and heuristic.
- Word Sense Disambiguation uses the simplified Lesk algorithm and WordNet, so the predicted meaning may not always match the intended context.
- Document extraction works best with text-based PDFs; scanned image-only PDFs may require OCR.

These limitations provide opportunities for further improvement.

---

## 🔮 Future Enhancements

Some possible improvements include:

- Adding Optical Character Recognition (OCR) for scanned documents.
- Supporting additional languages.
- Improving semantic analysis using advanced NLP models.
- Adding more advanced summarization techniques.
- Introducing document comparison and similarity analysis.
- Supporting larger documents and batch processing.
- Improving the accuracy of contextual word sense identification.

---

## 📚 Learning Outcomes

Through this project, I explored and applied several concepts from Natural Language Processing and Python development, including:

- Text preprocessing and tokenization.
- Part-of-Speech tagging.
- Lemmatization and morphological analysis.
- Named Entity Recognition.
- Dependency parsing.
- N-gram language models.
- Text summarization.
- Readability analysis.
- Semantic relationship extraction.
- Word Sense Disambiguation.
- Data visualization.
- Building and deploying an interactive Python web application.

Most importantly, this project helped me connect theoretical concepts learned in my B.Tech NLP coursework with a practical application.

---

## 👩‍💻 Author

**Safura Nishat**

B.Tech – Computer Science and Engineering  
3rd Year Student

GitHub: https://github.com/safuranishat

LinkedIn: https://www.linkedin.com/in/safura-nishat-850196422/

---

## ❤️ Acknowledgement

This project was developed as part of the Natural Language Processing coursework in my B.Tech 3rd year.

It represents an effort to explore different NLP techniques and bring them together into one practical, interactive application.

Building DocuMind was a valuable learning experience that helped me understand how Natural Language Processing can be used to make information extraction and document understanding more accessible.

---

**DocuMind – Turning Unstructured Documents into Meaningful Insights.** 🚀
