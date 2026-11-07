import nltk
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
import streamlit as st
import spacy
import streamlit.components.v1 as components
from spacy import displacy
import pymupdf
import re
import math
import pandas as pd
import plotly.express as px

@st.cache_resource
def download_wordnet():
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)

download_wordnet()

from collections import Counter
from itertools import islice
from io import BytesIO
from docx import Document


# -----------------------------------
# PAGE CONFIGURATION
# -----------------------------------

st.set_page_config(
    page_title="Intelligent Document Analyzer",
    page_icon="📄",
    layout="wide"
)


# -----------------------------------
# CUSTOM DESIGN
# -----------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #0B1020;
    color: #FFFFFF;
}

h1 {
    font-size: 38px !important;
    color: #FFFFFF !important;
}

h2 {
    font-size: 29px !important;
    color: #FFFFFF !important;
}

h3 {
    font-size: 23px !important;
    color: #DCE6FF !important;
}

/* Extracted document text */

[data-testid="stTextArea"] textarea {
    font-size: 19px !important;
    line-height: 1.8 !important;
    color: #FFFFFF !important;
    background-color: #151B2E !important;
    padding: 18px !important;
}

/* Text area labels */

[data-testid="stTextArea"] label {
    font-size: 19px !important;
    font-weight: 600 !important;
}

/* Metric cards */

[data-testid="stMetric"] {
    background-color: #151B2E;
    padding: 20px;
    border-radius: 12px;
    border: 1px solid #293653;
}

/* Tabs */

.stTabs [data-baseweb="tab"] {
    font-size: 16px;
    font-weight: 600;
}

/* Buttons */

.stButton button {
    border-radius: 8px;
}

/* Sidebar navigation */

section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111A30 0%, #0B1020 100%);
    border-right: 1px solid #293653;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p {
    color: #DCE6FF;
    font-size: 12px;
    line-height: 1.25;
    margin-bottom: 3px;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3 {
    font-size: 17px !important;
    margin-top: 5px;
    margin-bottom: 7px;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
}

.nav-item {
    background: #151F35;
    border: 1px solid #293653;
    border-radius: 7px;
    padding: 5px 8px;
    margin: 3px 0;
    color: #EAF0FF;
    font-size: 12px;
    line-height: 1.15;
}

.nav-number {
    color: #8FB3FF;
    font-weight: 700;
    margin-right: 6px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------------
# LOAD NLP MODEL
# -----------------------------------

@st.cache_resource
def load_model():
    return spacy.load("en_core_web_sm")


nlp = load_model()


# -----------------------------------
# DOCUMENT EXTRACTION
# -----------------------------------

def extract_text(file):

    extension = file.name.split(".")[-1].lower()

    if extension == "pdf":

        pdf = pymupdf.open(
            stream=file.getvalue(),
            filetype="pdf"
        )

        text = "\n".join(
            page.get_text()
            for page in pdf
        )

        pdf.close()

        return text

    elif extension == "docx":

        doc = Document(
            BytesIO(file.getvalue())
        )

        return "\n".join(
            paragraph.text
            for paragraph in doc.paragraphs
        )

    elif extension == "txt":

        return file.getvalue().decode(
            "utf-8",
            errors="replace"
        )

    return ""


# -----------------------------------
# DOCUMENT STRUCTURE
# -----------------------------------

def analyze_structure(text):

    lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip()
    ]

    headings = []

    for line in lines:

        if (
            len(line) < 100
            and (
                line.isupper()
                or re.match(
                    r"^(chapter|section|[0-9]+[.)])",
                    line,
                    re.I
                )
            )
        ):
            headings.append(line)

    paragraphs = [
        p.strip()
        for p in re.split(r"\n\s*\n", text)
        if p.strip()
    ]

    return headings, paragraphs


# -----------------------------------
# KEYWORD EXTRACTION
# -----------------------------------

def extract_keywords(doc):

    words = [
        token.lemma_.lower()
        for token in doc
        if token.is_alpha
        and not token.is_stop
        and len(token.text) > 2
    ]

    return Counter(words).most_common(15)


# -----------------------------------
# N-GRAM ANALYSIS
# -----------------------------------

def get_ngrams(words, n):

    if len(words) < n:
        return []

    grams = zip(
        *(islice(words, i, None)
          for i in range(n))
    )

    return Counter(grams).most_common(10)


# -----------------------------------
# READABILITY ANALYSIS
# -----------------------------------

def count_syllables(word):

    word = word.lower()

    vowels = "aeiou"
    count = 0
    previous = False

    for char in word:

        current = char in vowels

        if current and not previous:
            count += 1

        previous = current

    if word.endswith("e") and count > 1:
        count -= 1

    return max(count, 1)


def readability_score(doc):

    sentences = list(doc.sents)

    words = [
        token.text
        for token in doc
        if token.is_alpha
    ]

    if not sentences or not words:
        return None

    syllables = sum(
        count_syllables(word)
        for word in words
    )

    score = (
        206.835
        - 1.015 * (len(words) / len(sentences))
        - 84.6 * (syllables / len(words))
    )

    return round(score, 2)


# -----------------------------------
# SMART SUMMARY
# -----------------------------------

def generate_summary(doc):

    sentences = list(doc.sents)

    if len(sentences) <= 3:
        return [s.text.strip() for s in sentences]

    frequencies = Counter(
        token.lemma_.lower()
        for token in doc
        if token.is_alpha and not token.is_stop
    )

    ranked = []

    for sentence in sentences:

        score = sum(
            frequencies[token.lemma_.lower()]
            for token in sentence
            if token.is_alpha and not token.is_stop
        )

        ranked.append(
            (score, sentence.text.strip())
        )

    top = sorted(
        ranked,
        reverse=True
    )[:min(5, len(ranked))]

    return [
        sentence
        for score, sentence in top
    ]


# -----------------------------------
# HEADER
# -----------------------------------

st.title("📄 Intelligent Document Analyzer")

st.markdown(
    "### Understand Documents Beyond Their Words"
)

st.caption(
    "An NLP-powered platform for structural, "
    "linguistic, syntactic and semantic analysis."
)

st.divider()


# -----------------------------------
# SIDEBAR
# -----------------------------------

st.sidebar.markdown(
    """
    <div style="padding: 8px 0 14px 0;">
        <div style="font-size: 25px; font-weight: 800; color: #FFFFFF;">
            🧠 DocuMind
        </div>
        <div style="font-size: 12px; color: #9FB3D9; letter-spacing: 1px;">
            INTELLIGENT DOCUMENT ANALYZER
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

st.sidebar.markdown("### 🧭 Navigation")

navigation_items = [
    ("01", "Document Overview"),
    ("02", "Document Text"),
    ("03", "Document Structure"),
    ("04", "Linguistic Analysis"),
    ("05", "Named Entity Recognition"),
    ("06", "Syntax Analysis"),
    ("07", "Language Models"),
    ("08", "Smart Summary & Readability"),
    ("09", "Semantic Analysis"),
    ("10", "Word Sense Disambiguation"),
]

for number, label in navigation_items:
    st.sidebar.markdown(
        f'<div class="nav-item"><span class="nav-number">{number}</span>{label}</div>',
        unsafe_allow_html=True
    )

st.sidebar.divider()
st.sidebar.caption("⚙️ Engine: spaCy + WordNet")
st.sidebar.caption("📁 PDF  •  DOCX  •  TXT")


# -----------------------------------
# DOCUMENT UPLOAD
# -----------------------------------

st.header("📤 Upload Your Document")

uploaded_file = st.file_uploader(
    "Choose your document",
    type=["pdf", "docx", "txt"]
)


if uploaded_file:

    text = extract_text(uploaded_file)

    if not text.strip():

        st.error("No readable text found.")

        st.stop()

    with st.spinner("Analyzing your document..."):

        doc = nlp(text)

        sentences = list(doc.sents)

        words = [
            token for token in doc
            if token.is_alpha
        ]

        headings, paragraphs = analyze_structure(text)

        keywords = extract_keywords(doc)

        entities = [
            (ent.text, ent.label_)
            for ent in doc.ents
        ]

        summary = generate_summary(doc)

        readability = readability_score(doc)

        pos_counts = Counter(
            token.pos_
            for token in doc
            if token.is_alpha
        )

        word_list = [
            token.text.lower()
            for token in doc
            if token.is_alpha
        ]

        word_frequency = Counter(word_list)

        bigrams = get_ngrams(word_list, 2)

        trigrams = get_ngrams(word_list, 3)


    st.success(
        f"Successfully analyzed: {uploaded_file.name}"
    )

    # -----------------------------------
    # DASHBOARD METRICS
    # -----------------------------------

    st.header("📊 Document Overview")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric(
        "Total Words",
        len(words)
    )

    c2.metric(
        "Sentences",
        len(sentences)
    )

    c3.metric(
        "Paragraphs",
        len(paragraphs)
    )

    c4.metric(
        "Named Entities",
        len(entities)
    )

    c5, c6, c7, c8 = st.columns(4)

    c5.metric(
        "Unique Words",
        len(set(word_list))
    )

    c6.metric(
        "Headings",
        len(headings)
    )

    c7.metric(
        "Characters",
        len(text)
    )

    c8.metric(
        "Reading Time",
        f"{max(1, math.ceil(len(words)/200))} min"
    )

    st.divider()


    # -----------------------------------
    # ANALYSIS TABS
    # -----------------------------------

    tabs = st.tabs([
        "📖 Document Text",
        "🏗️ Structure",
        "🔤 Linguistic Analysis",
        "🔍 Named Entities",
        "🌳 Syntax",
        "📈 Language Models",
        "✨ Smart Summary",
        "🧠 Semantic Analysis",
        "🧩 Word Sense Disambiguation"
    ])

    # -----------------------------------
    # TAB 1: DOCUMENT TEXT
    # -----------------------------------

    with tabs[0]:

        st.header("Extracted Document Text")

        st.text_area(
            "Full Document Content",
            text,
            height=500
        )

        st.download_button(
            "⬇️ Download Extracted Text",
            text,
            file_name="extracted_document.txt"
        )


    # -----------------------------------
    # TAB 2: STRUCTURE
    # -----------------------------------

    with tabs[1]:

        st.header("Document Structure Analysis")

        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Detected Headings")

            if headings:

                for heading in headings:
                    st.markdown(f"- {heading}")

            else:

                st.info("No clear headings detected.")

        with col2:

            st.subheader("Paragraph Information")

            st.metric(
                "Total Paragraphs",
                len(paragraphs)
            )

            for i, paragraph in enumerate(
                paragraphs[:10], 1
            ):

                with st.expander(
                    f"Paragraph {i}"
                ):

                    st.write(paragraph)


    # -----------------------------------
    # TAB 3: LINGUISTIC ANALYSIS
    # -----------------------------------

    with tabs[2]:

        st.header("Linguistic Analysis")

        st.subheader("Parts of Speech Distribution")

        pos_df = pd.DataFrame(
            pos_counts.items(),
            columns=["POS Tag", "Count"]
        )

        if not pos_df.empty:

            fig = px.pie(
                pos_df,
                names="POS Tag",
                values="Count",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            pos_full_forms = {
                "ADJ": ("Adjective", "Describes a noun or pronoun"),
                "ADP": ("Adposition", "Shows relationships such as in, on, at, or by"),
                "ADV": ("Adverb", "Modifies a verb, adjective, or another adverb"),
                "AUX": ("Auxiliary Verb", "Helps the main verb express tense, mood, or voice"),
                "CCONJ": ("Coordinating Conjunction", "Connects words or phrases, such as and, but, or"),
                "DET": ("Determiner", "Introduces or specifies a noun, such as the, a, or this"),
                "INTJ": ("Interjection", "Expresses a sudden feeling or reaction"),
                "NOUN": ("Noun", "Names a person, place, thing, or idea"),
                "NUM": ("Numeral", "Represents a number or quantity"),
                "PART": ("Particle", "A function word associated with another word or phrase"),
                "PRON": ("Pronoun", "Replaces or refers to a noun"),
                "PROPN": ("Proper Noun", "Names a specific person, place, or organization"),
                "PUNCT": ("Punctuation", "Punctuation marks such as commas and full stops"),
                "SCONJ": ("Subordinating Conjunction", "Introduces a dependent clause"),
                "SYM": ("Symbol", "A symbol such as %, $, or +"),
                "VERB": ("Verb", "Expresses an action, event, or state"),
                "X": ("Other", "A word that does not fit the standard POS categories")
            }

            st.markdown("#### 🏷️ POS Tag Full Forms")
            pos_legend = pd.DataFrame(
                [
                    {
                        "POS Tag": tag,
                        "Full Form": pos_full_forms.get(
                            tag, ("Other / Unmapped", "No description available")
                        )[0],
                        "Meaning": pos_full_forms.get(
                            tag, ("Other / Unmapped", "No description available")
                        )[1]
                    }
                    for tag in sorted(pos_counts.keys())
                ]
            )
            st.dataframe(
                pos_legend,
                use_container_width=True,
                hide_index=True
            )
            st.caption(
                "The pie chart uses Universal POS tags. This legend explains "
                "each abbreviation shown in the chart."
            )

        st.subheader("Morphological Analysis")

        morphology = [
            {
                "Word": token.text,
                "Lemma": token.lemma_,
                "POS": token.pos_,
                "Tag": token.tag_
            }
            for token in doc
            if token.is_alpha
        ]

        st.dataframe(
            pd.DataFrame(morphology).head(100),
            use_container_width=True
        )

        st.subheader("Top Keywords")

        keyword_df = pd.DataFrame(
            keywords,
            columns=["Keyword", "Frequency"]
        )

        st.dataframe(
            keyword_df,
            use_container_width=True
        )


    # -----------------------------------
    # TAB 4: NAMED ENTITIES
    # -----------------------------------

    with tabs[3]:

        st.header("Named Entity Recognition")

        st.write(
            "Identifying people, organizations, "
            "locations, dates and other entities."
        )

        if entities:

            entity_df = pd.DataFrame(
                entities,
                columns=["Entity", "Category"]
            )

            st.dataframe(
                entity_df,
                use_container_width=True
            )

            entity_counts = Counter(
                label for name, label in entities
            )

            fig = px.bar(
                x=list(entity_counts.keys()),
                y=list(entity_counts.values()),
                labels={
                    "x": "Entity Type",
                    "y": "Count"
                },
                title="Entity Distribution"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

        else:

            st.info("No named entities detected.")


    # -----------------------------------
    # TAB 5: ADVANCED SYNTAX ANALYSIS
    # -----------------------------------

    with tabs[4]:

        st.header("🌳 Advanced Syntactic Analysis")

        st.markdown(
            "Explore grammatical structures, "
            "dependency relationships and sentence parsing."
        )

        st.divider()

        # Select sentence
        sentence_options = [
            s.text.strip()
            for s in sentences
            if s.text.strip()
        ]

        if sentence_options:

            selected_sentence = st.selectbox(
                "Select a sentence for analysis",
                sentence_options
            )

            # Process selected sentence
            sentence_doc = nlp(selected_sentence)

            st.subheader("Selected Sentence")

            st.info(selected_sentence)

            # Dependency Tree
            st.subheader("🌳 Dependency Parse Tree")

            st.write(
                "This visualization displays grammatical "
                "relationships between words."
            )

            dependency_html = displacy.render(
                sentence_doc,
                style="dep",
                page=False,
                options={
                    "compact": False,
                    "distance": 150,
                    "word_spacing": 35,
                    "arrow_spacing": 12,
                    "font": "Arial"
                }
            )

            # Keep every token and dependency label readable.
            # Long sentences can be scrolled horizontally instead of being
            # squeezed into a tiny graph.
            token_count = max(1, len(sentence_doc))
            graph_min_width = max(900, token_count * 105)
            graph_height = min(680, max(300, 190 + token_count * 8))

            dependency_view = f"""
            <!DOCTYPE html>
            <html>
            <head>
            <meta name="viewport" content="width=device-width, initial-scale=1">
            <style>
                html, body {{
                    margin: 0;
                    padding: 0;
                    background: #FFFFFF;
                    font-family: Arial, sans-serif;
                }}
                .graph-scroll {{
                    width: 100%;
                    overflow-x: auto;
                    overflow-y: hidden;
                    background: #FFFFFF;
                    border: 1px solid #D7DEEA;
                    border-radius: 10px;
                    padding: 12px;
                    box-sizing: border-box;
                }}
                .graph-scroll svg {{
                    display: block;
                    width: {graph_min_width}px;
                    max-width: none;
                    height: auto;
                }}
                .graph-scroll text {{
                    font-size: 18px !important;
                }}
                .graph-scroll .displacy-label {{
                    font-size: 15px !important;
                }}
            </style>
            </head>
            <body>
                <div class="graph-scroll">{dependency_html}</div>
            </body>
            </html>
            """

            components.html(
                dependency_view,
                height=graph_height,
                scrolling=True
            )

            st.divider()

            # Grammatical Analysis
            st.subheader("📊 Grammatical Relationship Analysis")

            syntax_data = []

            for token in sentence_doc:

                syntax_data.append({

                    "Word": token.text,

                    "Part of Speech": token.pos_,

                    "Dependency": token.dep_,

                    "Head Word": token.head.text,

                    "Is Root": token.dep_ == "ROOT"

                })

            syntax_df = pd.DataFrame(syntax_data)

            st.dataframe(
                syntax_df,
                use_container_width=True,
                hide_index=True
            )

            st.divider()

            # Sentence statistics
            col1, col2, col3 = st.columns(3)

            col1.metric(
                "Total Tokens",
                len(sentence_doc)
            )

            col2.metric(
                "Nouns",
                sum(t.pos_ == "NOUN" for t in sentence_doc)
            )

            col3.metric(
                "Verbs",
                sum(t.pos_ == "VERB" for t in sentence_doc)
            )

            # Root word
            root_words = [
                token.text
                for token in sentence_doc
                if token.dep_ == "ROOT"
            ]

            st.subheader("Main Sentence Root")

        if root_words:
                st.success(
                    f"Main Root: {', '.join(root_words)}"
                )

        else:
            st.warning(
                "No sentences available for syntactic analysis."
            )


        # -----------------------------------
    # TAB 6: LANGUAGE MODELS
    # -----------------------------------

    with tabs[5]:

        st.header("N-gram Language Analysis")
        st.write(
            "Explore unigram, bigram and trigram patterns in the uploaded document."
        )

        st.subheader("Unigram Analysis")
        unigram_data = word_frequency.most_common(25)
        unigram_df = pd.DataFrame(
            unigram_data,
            columns=["Word (Unigram)", "Frequency"]
        )
        st.dataframe(
            unigram_df,
            use_container_width=True,
            hide_index=True
        )

        st.divider()
        col1, col2 = st.columns(2)

        with col1:

            st.subheader("Bigram Analysis")

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Phrase": " ".join(gram),
                            "Frequency": count
                        }
                        for gram, count in bigrams
                    ]
                ),
                use_container_width=True
            )

        with col2:

            st.subheader("Trigram Analysis")

            st.dataframe(
                pd.DataFrame(
                    [
                        {
                            "Phrase": " ".join(gram),
                            "Frequency": count
                        }
                        for gram, count in trigrams
                    ]
                ),
                use_container_width=True
            )

        st.subheader("Most Frequent Words — Visual Summary")

        common_words = word_frequency.most_common(15)

        freq_df = pd.DataFrame(
            common_words,
            columns=["Word", "Frequency"]
        )

        fig = px.bar(
            freq_df,
            x="Word",
            y="Frequency",
            title="Most Frequent Words"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )


    # -----------------------------------
    # TAB 7: SMART SUMMARY
    # -----------------------------------

    with tabs[6]:

        st.header("Intelligent Document Summary")

        st.write(
            "Extracting important sentences based "
            "on word frequency."
        )

        for i, sentence in enumerate(summary, 1):

            st.markdown(
                f"**{i}.** {sentence}"
            )

        st.divider()

        st.subheader("Readability Analysis")

        if readability is not None:

            st.metric(
                "Estimated Flesch Reading Ease",
                readability
            )

            if readability >= 60:
                st.success("Generally easier to read.")

            elif readability >= 30:
                st.warning("Moderately difficult text.")

            else:
                st.info("Complex text. May require advanced reading skills.")

        else:

            st.info("Not enough text to calculate readability.")


    # -----------------------------------
    # DOWNLOAD REPORT
    # -----------------------------------

    st.divider()

    st.header("📥 Export Analysis")

    report = f"""
INTELLIGENT DOCUMENT ANALYZER
=============================

Document: {uploaded_file.name}

Total Words: {len(words)}
Sentences: {len(sentences)}
Paragraphs: {len(paragraphs)}
Named Entities: {len(entities)}
Unique Words: {len(set(word_list))}
Headings: {len(headings)}
Characters: {len(text)}

TOP KEYWORDS
============

{chr(10).join(
    f"{word}: {count}"
    for word, count in keywords
)}

SMART SUMMARY
=============

{chr(10).join(summary)}

READABILITY SCORE
=================

{readability}
"""

    st.download_button(
        "📥 Download Analysis Report",
        report,
        file_name="document_analysis_report.txt",
        mime="text/plain"
    )

    # -----------------------------------
    # TAB 8: SEMANTIC ANALYSIS
    # -----------------------------------

    with tabs[7]:

        st.header("🧠 Semantic Intelligence Engine")

        st.write(
            "Extract meaningful relationships "
            "between subjects, actions and objects."
        )

        st.divider()

        semantic_results = []

        for sentence in sentences:

            root = next(
                (
                    token for token in sentence
                    if token.dep_ == "ROOT"
                    and token.pos_ in ["VERB", "AUX"]
                ),
                None
            )

            if root is None:
                continue

            subjects = [
                token for token in root.children
                if token.dep_ in [
                    "nsubj", "nsubjpass",
                    "csubj", "csubjpass"
                ]
            ]

            objects = [
                token for token in root.children
                if token.dep_ in [
                    "dobj", "obj", "attr", "oprd"
                ]
            ]

            # Extract prepositional objects
            for child in root.children:

                if child.dep_ == "prep":

                    objects.extend([
                        token for token in child.children
                        if token.dep_ == "pobj"
                    ])

            def get_phrase(token):

                subtree = list(token.subtree)

                start = min(t.i for t in subtree)
                end = max(t.i for t in subtree) + 1

                return doc[start:end].text

            subject_text = ", ".join(
                get_phrase(token)
                for token in subjects
            )

            object_text = ", ".join(
                get_phrase(token)
                for token in objects
            )

            semantic_results.append({

                "Subject": subject_text or "Not identified",

                "Action": root.lemma_,

                "Object": object_text or "Not identified",

                "Original Sentence": sentence.text.strip()

            })

        if semantic_results:

            semantic_df = pd.DataFrame(
                semantic_results
            )

            st.subheader("Subject–Action–Object Extraction")

            st.dataframe(
                semantic_df,
                use_container_width=True,
                hide_index=True
            )

            st.metric(
                "Relationships Extracted",
                len(semantic_results)
            )

            st.download_button(
                "Download Semantic Analysis",
                semantic_df.to_csv(index=False),
                file_name="semantic_analysis.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                "No suitable subject-action relationships detected."
            )

        st.caption(
            "Method: Dependency-based semantic relationship extraction. "
            "This is a heuristic foundation, not a full semantic role labeler."
        )


    # -----------------------------------
    # TAB 9: WORD SENSE DISAMBIGUATION
    # -----------------------------------

    with tabs[8]:

        st.header("🧩 Word Sense Disambiguation")

        st.write(
            "Identify the intended meaning of a word "
            "using its surrounding context."
        )

        st.divider()

        # Find meaningful words
        candidates = sorted(set(
            token.text.lower()
            for token in doc
            if token.is_alpha
            and not token.is_stop
            and token.pos_ in ["NOUN", "VERB", "ADJ", "ADV"]
        ))

        if candidates:

            selected_word = st.selectbox(
                "Select a word to understand its meaning",
                candidates
            )

            # Find sentences containing the selected word
            matching_sentences = [
                sentence
                for sentence in sentences
                if any(
                    token.text.lower() == selected_word
                    for token in sentence
                )
            ]

            if matching_sentences:

                selected_sentence = st.selectbox(
                    "Choose its context",
                    [s.text.strip() for s in matching_sentences]
                )

                context_doc = nlp(selected_sentence)

                target_token = next(
                    token for token in context_doc
                    if token.text.lower() == selected_word
                )

                pos_mapping = {
                    "NOUN": wn.NOUN,
                    "VERB": wn.VERB,
                    "ADJ": wn.ADJ,
                    "ADV": wn.ADV
                }

                word_pos = pos_mapping.get(
                    target_token.pos_
                )

                context_words = [
                    token.text.lower()
                    for token in context_doc
                    if token.is_alpha
                ]

                with st.spinner("Identifying word meaning..."):

                    sense = lesk(
                        context_words,
                        selected_word,
                        pos=word_pos
                    )

                st.subheader("Context Analysis")

                st.info(selected_sentence)

                if sense:

                    st.success(
                        f"Identified Meaning: {sense.definition()}"
                    )

                    st.markdown(
                        f"**Word:** {selected_word}"
                    )

                    st.markdown(
                        f"**Word Category:** {target_token.pos_}"
                    )

                    st.markdown(
                        f"**WordNet Sense:** {sense.name()}"
                    )

                    st.subheader("Meaning Examples")

                    examples = sense.examples()

                    if examples:

                        for example in examples:
                            st.markdown(f"- {example}")

                    else:

                        st.info("No example sentences available.")

                    st.subheader("Related Synonyms")

                    synonyms = sorted(set(
                        lemma.name().replace("_", " ")
                        for lemma in sense.lemmas()
                    ))

                    st.write(", ".join(synonyms))

                else:

                    st.warning(
                        "No suitable word sense was identified. "
                        "Try another word or sentence."
                    )

                st.caption(
                    "Algorithm: Simplified Lesk algorithm using NLTK WordNet. "
                    "This is a dictionary-overlap baseline, so results "
                    "may not always identify the correct contextual meaning."
                )

        else:

            st.warning(
                "No suitable words found for analysis."
            )

else:

    st.info(
        "Upload a document to unlock the analysis dashboard."
    )

    st.markdown("### Explore our NLP capabilities")

    c1, c2, c3 = st.columns(3)

    c1.info("🏗️ Document Structure Analysis")

    c2.info("🔤 Morphological & Linguistic Analysis")

    c3.info("🌳 Syntactic Analysis")

    c4, c5, c6 = st.columns(3)

    c4.info("🔍 Named Entity Recognition")

    c5.info("📈 N-gram Language Models")