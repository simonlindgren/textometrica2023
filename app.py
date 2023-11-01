import streamlit as st
import zipfile
import os
import shutil
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
import re
import string
from itertools import combinations
import networkx as nx
from pyvis.network import Network

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords

# Start page
def start_page():
    st.image('logo.png')
    st.markdown('***')
    st.markdown("[*Textometrica*](https://web.archive.org/web/20120201063603/http://textometrica.humlab.umu.se/) was an application for combining quantitative content analysis, qualitative thematization, and network analysis, originally conceived by me, Simon Lindgren, and coded in PHP by Fredrik Palm at Humlab, Umeå University, in 2011.")
    st.markdown("")
    st.markdown("This app, coded in Python by [Simon Lindgren](https://github.com/simonlindgren), makes the Textometrica workflow available anew. If you use this approach, conceived as CCA (Connected Concept Analysis), please cite:")
    st.markdown("> Lindgren, S. (2016). \"Introducing Connected Concept Analysis\". *Text & Talk*, 36(3), 341–362 [[doi](https://doi.org/10.1515/text-2016-0016)]")

    with open("cm.zip", 'rb') as f:
        st.download_button(
            label="Download example file for analysis",
            data=f,
            file_name="cm.zip",
            mime="application/zip"
        )

# Define all other pages
def page_1(): 
    st.markdown("### Upload data")
    uploaded_file = st.file_uploader("", type='zip')
    st.markdown("***")
    st.markdown("The data to be uploaded must be a zip archive, containing a set of individual *.txt files. In the analysis to be performed, these text files will be considered as the *documents*. We will analyse *co-occurrences* of *words* within documents.")
    if uploaded_file:
        with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
            temp_dir = "temp_txt_files"
            zip_ref.extractall(temp_dir)
            corpus = []
            for root, dirs, files in os.walk(temp_dir):
                for file in files:
                    if file.endswith(".txt"):
                        if not file.startswith("._"): # exclude annoying macos files
                            with open(os.path.join(root, file), 'r', encoding='utf-8', errors='replace') as f:
                                contents = f.read().lower()
                                corpus.append(contents)
            shutil.rmtree(temp_dir)
        st.markdown("✅ All text has now been imported and converted to lowercase.")
        st.markdown("  The next step is *Preprocess*.")
       
        
        # Save corpus to session state
        st.session_state.corpus = corpus

def page_2(): 
# Access corpus from session state
    if 'corpus' in st.session_state:
        st.markdown("### Preprocess")
        st.write("Number of documents in the corpus:", len(st.session_state.corpus))
        
        available_languages = sorted(stopwords.fileids())
        language = st.selectbox("Select NLTK stopword language:", available_languages, index=available_languages.index('english'))
        additional_stopwords = st.text_input("Enter additional stopwords (space-separated):")
        additional_stopwords_list = additional_stopwords.split()
        stop_words = list(set(stopwords.words(language)).union(additional_stopwords_list))
        st.session_state.stopwords = stop_words
        progress = st.progress(0)
        st.markdown("Clicking the button below will:")
        st.markdown("""
        - remove your stopwords
        - remove words with >2 characters
        - remove numerical and special characters
        """)
        if st.button("Run preprocessing"):
            progress.text("Filtering and Pre-processing...")
            progress.progress(0.3)
            progress.progress(0.6)
            cv = CountVectorizer(ngram_range=(1, 1),
                                strip_accents='unicode',
                                stop_words=stop_words,
                                token_pattern="[a-zA-Z][a-zA-Z]+")  # at least two letters, and no numerical or special characters
            dtm = cv.fit_transform(st.session_state.corpus)
            progress.progress(1.0)
            wordlist = cv.get_feature_names()
            docfreqs = list(np.squeeze(np.asarray((dtm != 0).sum(0))))  # count number of non-zero document occurrences for each row (i.e. each word)
            countsDF = pd.DataFrame(zip(wordlist, docfreqs)).reset_index()
            countsDF.columns = ["id", "word", "DF"]
            # Save df to session state
            st.session_state.countsDF = countsDF
            st.markdown("✅ Preprocessing done.")
            st.markdown("← You can now *Set threshold*.")

    else:
        st.write("Corpus is not available. Please upload a zip file.")

def page_3():  
    if 'countsDF' in st.session_state:
        st.markdown("### Set threshold")
        countsDF = st.session_state.countsDF
        countsDF = countsDF.drop('id', axis = 1).sort_values(by='DF', ascending = False).reset_index().drop('index', axis=1)
        st.markdown("This is a list of words, sorted by their *document frequencies* (i.e. how many documents they occur in, no matter how many times).")
        st.markdown("Manually inspect the list, by scrolling down, and decide how many rows you want to keep. Note that the first row is number 0. The number entered below is the number of the first row to be *excluded*.")
        st.markdown("If there is too much garbage here, you can preprocess again to remove more stopwords.")
        st.dataframe(countsDF, height=300, width=400)
        default_threshold = round(len(st.session_state.countsDF) / 3)
        threshold = st.text_input("Cut at row number: (default value = keep the top third, 33%):", value=str(default_threshold))
        if st.button("Confirm threshold"):
            try:
                threshold = int(threshold)
                if 1 <= threshold <= len(st.session_state.countsDF):
                    st.write(f"✅ Keeping all words above row {threshold}.")
                    st.write("← You can now move on to *Select words*.")
                    shortDF = st.session_state.countsDF.sort_values(by="DF", ascending=False).head(threshold)
                    # Save shortDF to session state
                    st.session_state.shortDF = shortDF
                else:
                    st.write(f"Please enter a number between 1 and {len(st.session_state.countsDF)}")
            except ValueError:
                st.write("Please enter a valid integer")
    else:
        st.write("Preprocessed data not available. Please run preprocessing.")

def page_4():
    st.markdown('<a name="top-of-page"></a>', unsafe_allow_html=True)
    if 'shortDF' in st.session_state:
        st.markdown("### Select words")
        st.markdown("Now, refine your selection of words by manually going through this list. Select the words you want to keep in the analysis, and deselect those that you are not interested in.")
        st.markdown("Use the 🔎 button to inspect a word in its context.")
        st.markdown("❗️Click the 'Confirm selection' button when done.")
        st.markdown("***")
        
        tokenlist = list(st.session_state.shortDF.word)

        # Dictionary to hold the state of each checkbox and snippet display
        if "checkbox_states" not in st.session_state:
            st.session_state.checkbox_states = {word: False for word in tokenlist}
        if "show_snippet" not in st.session_state:
            st.session_state.show_snippet = {word: False for word in tokenlist}

        keeplist = []
        snippet_length = 300  # Adjust for longer or shorter snippets

        # Create columns for layout
        col1, col2, space1, space2, col3 = st.columns([1,1,1,1,2])

        if col1.button("Select All"):
            for w in tokenlist:
                st.session_state.checkbox_states[w] = True
        
        if col2.button("Deselect All"):
            for w in tokenlist:
                st.session_state.checkbox_states[w] = False
        
        if col3.button("Confirm selection"):
            finalDF = st.session_state.shortDF[st.session_state.shortDF['word'].isin(keeplist)]
            st.write(f"✅ Keeping {len(st.session_state.keeplist)} words.")
            st.write("← You can now move on to *Make concepts*.")
            st.write("***")

        for word in tokenlist:
            col1, col2 = st.columns(2)
            
            is_checked = col1.checkbox(word, value=st.session_state.checkbox_states[word])
            
            st.session_state.checkbox_states[word] = is_checked
            if is_checked:
                keeplist.append(word)
            
            if col2.button(f"🔍 {word}"):
                st.session_state.show_snippet[word] = not st.session_state.show_snippet[word]

            if st.session_state.show_snippet[word]:
                word_pattern = re.compile(r'\b' + word + r'\b')  # Ensuring word boundaries
                
                for doc in st.session_state.corpus:
                    matches = [m.start() for m in word_pattern.finditer(doc)]
                    
                    for start_index in matches:
                        start = max(0, start_index - snippet_length)
                        end = min(len(doc), start_index + len(word) + snippet_length)
                        snippet = doc[start:end]

                        # highlight the word in purple
                        snippet = re.sub(word_pattern, f"<span style='color: orange;'>{word}</span>", snippet)

                        # snippet box styling
                        with st.container():
                            st.markdown(f"<div style='padding: 10px; border: 1px solid gray; border-radius: 5px; margin: 5px 0;'>...{snippet}...</div>", unsafe_allow_html=True)
            st.session_state.keeplist = keeplist

        st.markdown('<a name="bottom-of-page"></a>', unsafe_allow_html=True)
        st.markdown("[🔼 Back to the top. Remember to click 'Confirm selection'](#top-of-page)")
        
    else:
        st.write("Post-threshold data not available. Please set threshold.")  



def page_5():
    st.markdown('<a name="top-of-page"></a>', unsafe_allow_html=True)
    
    snippet_length = 300

    # initialize word_categories in session state if it doesn't exist
    if 'word_categories' not in st.session_state:
        st.session_state.word_categories = {}

    if 'shortDF' in st.session_state and 'keeplist' in st.session_state:
        st.markdown("### Make concepts")
        st.markdown("This step offers the opportunity for thematic coding:")
        st.markdown("- Each word is initially set as a 'one-word' category. You can keep some — or all words — in that state.")
        st.markdown("- If you want to assign a word to a concept, e.g., <span style='color: hotpink; font-size: 18px'>sushi</span> to FOOD, enter FOOD in 'Add concept' and press Enter.", unsafe_allow_html = True)
        st.markdown("- Go through all words in this way. (a) Keep it as a one-word concept; (b) Add a new concept to connect it to; or, (c) Connect it to an already created concept, using the 'Conneced concept' dropdown menu.")

        st.markdown("❗️ Click the 'Submit concepts' button when done.")

        if st.button("Submit concepts"):
            st.write("✅ Concepts submitted!")
            st.write("← You can now *View co-occurrences*.")
        
        st.markdown("***")

        if 'categories' not in st.session_state:
            st.session_state.categories = ["one-word"]

        # set default category for all words
        for word in st.session_state.keeplist:
            if word not in st.session_state.word_categories:
                st.session_state.word_categories[word] = "one-word"

        if "checkbox_states" not in st.session_state:
            st.session_state.checkbox_states = {word: False for word in st.session_state.keeplist}
        if "show_snippet" not in st.session_state:
            st.session_state.show_snippet = {word: False for word in st.session_state.keeplist}


        for word in st.session_state.keeplist:
            col1, col3, col4, col2 = st.columns([2, 2, 3, 4])
            
            with col1:
                st.markdown(f"<div style='border: 0px solid; padding: 0px;'><span style='color: hotpink; font-size: 18px''>{word}</span></div>", unsafe_allow_html=True)

            # context button
            with col2:
                st.markdown("<p style='font-size: 14px; margin-bottom: 30px;'>View context</p>", unsafe_allow_html=True)
                if col2.button(f"🔍 {word}"):
                    st.session_state.show_snippet[word] = not st.session_state.show_snippet[word]
                if st.session_state.show_snippet[word]:
                    word_pattern = re.compile(r'\b' + word + r'\b')  # ensuring word boundaries
                    for doc in st.session_state.corpus:
                        matches = [m.start() for m in word_pattern.finditer(doc)]
                        for start_index in matches:
                            start = max(0, start_index - snippet_length)
                            end = min(len(doc), start_index + len(word) + snippet_length)
                            snippet = doc[start:end]
                            snippet = re.sub(word_pattern, f"<span style='color: orange;'>{word}</span>", snippet)
                            with st.container():
                                st.markdown(f"<div style='padding: 10px; border: 1px solid red; border-radius: 5px; margin: 5px 0;'>...{snippet}...</div>", unsafe_allow_html=True)

            with col3:
                st.markdown("<p style='font-size: 14px; margin-bottom: 1px;'>Add concept</p>", unsafe_allow_html=True)
                new_category = st.text_input("", key=f"{word}_new").strip()
                if new_category and new_category not in st.session_state.categories:
                    st.session_state.categories.append(new_category)
                    st.session_state.word_categories[word] = new_category

            with col4:
                st.markdown("<p style='font-size: 14px; margin-bottom: 0px;'>Connected concept</p>", unsafe_allow_html=True)
                
                # Check if the word's category exists in the categories list
                if st.session_state.word_categories[word] in st.session_state.categories:
                    default_index = st.session_state.categories.index(st.session_state.word_categories[word])
                else:
                    default_index = 0  # Default to the first item in the dropdown
                
                selected_category = st.selectbox("", st.session_state.categories, index=default_index, key=f"{word}_category")
                st.session_state.word_categories[word] = selected_category


            st.write("----")

        st.markdown("[🔼 Back to the top. Remember to click 'Submit concepts'](#top-of-page)")
        

    else:
        st.write("No selected words available. Please select words first.")

def page_6():
    st.markdown('<a name="top-of-page"></a>', unsafe_allow_html=True)
    if 'word_categories' in st.session_state:
        st.markdown("### View co-occurrences")
        st.write("These are your co-occurring pairs. Select which ones you want to keep.")
        st.markdown("❗️ Click the 'Confirm selection' button when done.")
    
        # Your existing code for processing co-occurrences...
        all_coocs = []
        words = list(st.session_state.word_categories.keys())
        words_set = set(words)
        bows = []
        
        for doc in st.session_state.corpus:
            doc = doc.split()
            doc = [s.translate(str.maketrans('', '', string.punctuation)) for s in doc]
            doc = [t for t in doc if not t in st.session_state.stopwords]
            doc = [t for t in doc if len(t) > 0]
            doc = [t for t in doc if t.isalpha()]
            doc = list(set(doc))
            bows.append(doc)
        
        for bow in bows:
            bow = [t for t in bow if t in words]
            pairs = combinations(bow, 2) # or any other number than 
            for p in pairs:
                all_coocs.append(p)
        
        norm_coocs = [tuple(sorted(pair)) for pair in all_coocs]
        df = pd.DataFrame(norm_coocs, columns=['source', 'target'])
        
        def map_to_category(word):
            category = st.session_state.word_categories.get(word, word)
            return word if category == "one-word" else category
        
        df['source'] = df['source'].apply(map_to_category)
        df['target'] = df['target'].apply(map_to_category)
        df = df.groupby(['source', 'target']).size().reset_index(name='weight')
        df = df[df['source'] != df['target']]
       
        # Ensure 'df' is defined before using it in any buttons or functions
        df = pd.DataFrame(norm_coocs, columns=['source', 'target'])
        df['source'] = df['source'].apply(map_to_category)
        df['target'] = df['target'].apply(map_to_category)
        df = df.groupby(['source', 'target']).size().reset_index(name='weight')
        df = df[df['source'] != df['target']]

        # Define the confirm_selection function to use 'df'
        def confirm_selection(df, selected_rows):
            final_df = df.loc[selected_rows]
            st.markdown("✅ Co-occurrences saved.")
            st.markdown("← You can now *visualize the network*.")
            st.session_state.edgelist = final_df

        # Add a 'Confirm selection' button at the top of the page
        if st.button("Confirm selection"):
            selected_rows = [index for index, _ in df.iterrows() if st.session_state.get(f"checkbox_{index}", False)]
            confirm_selection(df, selected_rows)


        def display_dataframe(df):
            selected_rows = []
            
            col1, col2, _, col_threshold, col_button = st.columns([1, 1,1, 1, 2])
            
            with col1:
                if st.button("Select All"):
                    for index, row in df.iterrows():
                        st.session_state[f"checkbox_{index}"] = True
            
            with col2:
                if st.button("Deselect All"):
                    for index, row in df.iterrows():
                        st.session_state[f"checkbox_{index}"] = False
            
            with col_threshold:
                st.empty()  # This acts as a spacer if needed
                threshold = st.number_input('Threshold weight', min_value=0, value = 5)
            
            with col_button:
                st.write('<div style="margin-top: 29px;"></div>', unsafe_allow_html=True)
                if st.button("Select all >= "):
                    
                    for index, row in df.iterrows():
                        if row['weight'] >= threshold:
                            st.session_state[f"checkbox_{index}"] = True
            
            st.markdown("&nbsp; &nbsp; &nbsp; &nbsp; source -- target (weight)")

            for index, row in df.iterrows():
                checkbox_key = f"checkbox_{index}"
                checkbox_state = st.session_state.get(checkbox_key, False)
                is_selected = st.checkbox(f"{row['source']} -- {row['target']} ({row['weight']})", value=checkbox_state, key=checkbox_key)
                if is_selected:
                    selected_rows.append(index)
            

        
        display_dataframe(df)
        st.markdown("[🔼 Back to the top. Remember to click 'Confirm selection'](#top-of-page)")
    else:
        st.markdown("No concepts defined. Please make concepts.")

     

def page_7():
    if 'edgelist' in st.session_state:


        df = st.session_state.edgelist       
        
        # This version includes edge weight, and is the version of G that gets exported to GEXF
        G = nx.from_pandas_edgelist(df, 'source', 'target', ['weight'])

        # This version excludes edge weight and is used for pyvis visualisations
        G1 = nx.from_pandas_edgelist(df, 'source', 'target')

        # Create a Network object for visualization
        nt = Network(notebook=True, height="750px", bgcolor="black", font_color="white")
        

        # Pass the networkx graph object to it for the visualization
        nt.from_nx(G1)


        
        # Compute betweenness centrality for each node in the NetworkX graph
        betweenness = nx.betweenness_centrality(G)

        # Normalize the betweenness values to determine node sizes for visualization
        min_size =5
        max_size = 25
        min_betweenness = min(betweenness.values())
        max_betweenness = max(betweenness.values())

        if max_betweenness == min_betweenness:  # All nodes have the same betweenness centrality
            scaled_betweenness = {node: 2 for node in betweenness}
        else:
            scaled_betweenness = {
                node: min_size + (value - min_betweenness) * (max_size - min_size) / (max_betweenness - min_betweenness) 
                for node, value in betweenness.items()
            }

        # Configure node appearance
        for node in nt.nodes:
            node["color"] = "hotpink"  # Set node color to pink
            node["borderWidth"] = 0
            node["labelHighlightBold"] = True
            node_id = node['id']
            node['size'] = scaled_betweenness[node_id]

        # Set the physics options for the network using set_options


        physics_options = """
        var options = {
        "configure": {
                "enabled": true,
                "filter": ["physics"]
        },
        "edges": {
            "color": {
            "inherit": true
            },
            "smooth": false
        },
        "physics": {
            "repulsion": {
            "springLength": 175,
            "avoidOverlap": 1.0
            },
            "minVelocity": 10,
            "maxVelocity": 32,
            "solver": "repulsion",
            "timestep": 3.0
        }
        }
        """

        nt.set_options(physics_options)

        # Save the visualization as an HTML file
        nt.save_graph("network.html")

        # Display the network visualization in Streamlit
        st.markdown("#### Graph preview")
        st.components.v1.html(nt.html, width=770, height=770)
        
        st.markdown("🔴 Node size reflects betweenness centrality.")
        st.markdown("👆 Click and drag background to move graph.")
        st.markdown("👀 Two-finger up/down swipe (trackpad), or mousewheel, for zoom.")
        st.markdown("⚡️ Click and drag individual node for rubberband effect.")
        st.markdown("***")

        file_path = 'network.html'
        with open(file_path, 'r') as file:
            file_content = file.read()
        
        # html download
        btn = st.download_button(
            label="Download HTML (advanced preview)",
            data=file_content,
            file_name="textometrica_viz.html",
            mime="text/html"
        )

        # save GEXF
        nx.write_gexf(G, "textometrica_graph.gexf")

        # gexf download
        with open("textometrica_graph.gexf", "rb") as f:
            btn = st.download_button(
                label="Download GEXF for Gephi",
                data=f,
                file_name="graph.gexf",
                mime="application/xml"
            )

    else:
        st.write("No co-occurrences available. Please go to 'View co-occurrences'.")

#mMain function
def main():

    # Sidebar for quick links to steps
    st.sidebar.image('logo.png')
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    
    if st.sidebar.button("About"):
        st.session_state.page = 0
    if st.sidebar.button("Upload data"):
        st.session_state.page = 1
    if st.sidebar.button("Preprocess"):
        st.session_state.page = 2
    if st.sidebar.button("Set threshold"):
        st.session_state.page = 3
    if st.sidebar.button("Select words"):
        st.session_state.page = 4
    if st.sidebar.button("Make concepts"):
        st.session_state.page = 5
    if st.sidebar.button("View co-occurrences"):
        st.session_state.page = 6
    if st.sidebar.button("Visualize network"):
        st.session_state.page = 7
    
    if st.session_state.page == 0:
        start_page()
    elif st.session_state.page == 1:
        page_1()
    elif st.session_state.page == 2:
        page_2()
    elif st.session_state.page == 3:
        page_3()
    elif st.session_state.page == 4:
        page_4()
    elif st.session_state.page == 5:
        page_5()
    elif st.session_state.page == 6:
        page_6()
    elif st.session_state.page == 7:
        page_7()
    
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("")
    st.sidebar.markdown("<div style='margin-bottom: -30px;font-weight: bold;'>Citation</div>", unsafe_allow_html=True)
    st.sidebar.markdown(">Lindgren, S. (2016). \"Introducing Connected Concept Analysis\". *Text & Talk*, 36(3), 341–362 [[doi](https://doi.org/10.1515/text-2016-0016)]")
    
if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state.page = 0  # Default to the start page
    main()