from openai import OpenAI
import pandas as pd
import streamlit as st
import os



# Function to query OpenAI API
def query_openai(prompt, model="gpt-3.5-turbo"):
    try:
        # Ensure API key is set
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("API key is not set. Please set the 'OPENAI_API_KEY' environment variable or input it in the sidebar.")
        
        client = OpenAI(
        api_key=os.environ.get("OPENAI_API_KEY"),  # This is the default and can be omitted
        )

        
        # Make API call
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

#translate German to English
def translate_text(german_text):
    prompt = f"Translate the following German text to English:\n\n{german_text}"
    return query_openai(prompt)

# Helper function to process tabular outputs from OpenAI
def process_table_output(raw_output):
    try:
        lines = raw_output.strip().split("\n")
        columns = lines[0].split(" | ")
        rows = [line.split(" | ") for line in lines[2:]]  # Skip header and separator
        return pd.DataFrame(rows, columns=columns)
    except Exception:
        return pd.DataFrame()  # Return empty DataFrame if parsing fails



def extract_verben_mit_praepositionen(text):
    prompt = f"""
    Your task is to identify all verbs with prepositions (Verben mit Präpositionen). You don't have to identify Trennbare Verben. And beware that 'sein' is not a verb here.
    You will give the verb in its infinitive form, translate the verb and preposition into English, specify whether the preposition takes the dative or accusative case, and provide the example context where the verb with the preposition appears in the text.    คําบุพบทที่คุณระบุจะต้องไม่ใช่คําบุพบทหรือคําวิเศษณ์ชั่วคราวหรือตําแหน่ง.
    The preposition you identify must not be a temporal or positional preposition. It must not be an adverb also. Only focus on the preposition that is relevant with the verb.
    Format the output as a table with the following columns:
    - Verb 
    - Preposition
    - English Translation
    - Case
    - Example Sentence
    - English translation of Example Sentence
    And if there is no verb with a preposition in the text, don't return any output.
    
    
    German Text:{text}
    """
    return query_openai(prompt)





def extract_adjective_mit_praepositionen(text):
    prompt = f"""
    Your task is to extract all adjectives with prepositions (it is in the form of sein + adjective + prepositions). Examples of adjective with preposition are interessiert an, wütend auf. You don't have to identify adverbs.
    Only identify adjectives and prepositions if sentence is in active voice structure.
    Prepositions you identify are relevant to verbs only, not prepositional prepositions or temporal prepositions.
    You will translate the adjective and preposition to English, specify whether the preposition takes the dative or accusative case, and provide the example context where the adjective with the preposition appears in the text. 
    Format the output as a table with the following columns:
    - Adjective 
    - Preposition
    - English Translation
    - Case
    - Example Sentence
    - English translation of Example Sentence
    And if there is no adjective with a preposition in the text, don't return any output.
    German Text: {text}
    """
    return query_openai(prompt)





def extract_collocations(text):
    prompt = f"""
    Your task is to analyze the following German text and identify common collocations. Focus on the following types of collocations:
    - Verb-Noun Collocations: Combinations of verbs and nouns that frequently occur together (e.g., "eine Entscheidung treffen," "einen Fehler machen").
    - Adjective-Noun Collocations: Combinations of adjectives and nouns that frequently occur together (e.g., "ein schwieriges Problem," "eine wichtige Aufgabe").
    - Prepositional Phrases: Prepositions combined with nouns or verbs to form common phrases (e.g., "mit dem Problem umgehen," "auf die Zukunft hoffen").

    For each identified collocation, please provide the following information:
    1. Collocations:The exact German phrase
    2. English Translation:The closest English equivalent or translation
    3. Example Sentence: A sentence from the text where the collocation is used
    
    Format the response as a column
    - Collocations
    - English Translation 
    - Example Sentence
    - English translation of Example Sentence
    And if there is no collocation in the text, don't return output

    German Text:
    {text}
    """
    return query_openai(prompt)
# Streamlit App
def main():
    #web layout
    st.set_page_config(
    page_title="GER",  # Title of the app
    page_icon=":beer:",  # Emoji as the page icon
    layout="wide",  # Use "wide" layout for better usability
    )   

    st.title("📚German Text Analyzer🪄")
    st.write("""
    This application translates German text into English, and extracts verbs with prepositions, adjectives with prepositions, and collocations.
    """)


    # Input API key
    api_key_input = st.sidebar.text_input("Enter OpenAI API Key", type="password")
    if api_key_input:
        os.environ["OPENAI_API_KEY"] = api_key_input
        st.sidebar.success("API Key Saved!")
    
    german_text = st.text_area("Enter German text", height=200)

    if st.button("Analyze"):
        if not os.getenv("OPENAI_API_KEY"):
            st.error("API key is not set. Please enter it in the sidebar.")

    if german_text.strip():
        # Translation
        st.subheader("English Translation:")
        translation = translate_text(german_text)
        st.write(translation)

        # Verbs with prepositions
        st.subheader("Verbs with Prepositions:")
        verb_analysis = extract_verben_mit_praepositionen(german_text)
        verb_df = process_table_output(verb_analysis)
        st.dataframe(verb_df)
        csv = verb_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Verbs with Prepositions as CSV",
            data=csv,
            file_name='verbs_with_prepositions.csv',
            mime='text/csv',
        )

        # Adjectives with prepositions
        st.subheader("Adjectives with Prepositions:")
        adj_analysis = extract_adjective_mit_praepositionen(german_text)
        adj_df = process_table_output(adj_analysis)
        st.dataframe(adj_df)
        csv = adj_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Adjectives with Prepositions as CSV",
            data=csv,
            file_name='adjectives_with_prepositions.csv',
            mime='text/csv',
        )

        # Collocations
        st.subheader("Collocations:")
        collocation_analysis = extract_collocations(german_text)
        collocation_df = process_table_output(collocation_analysis)
        st.dataframe(collocation_df)
        csv = collocation_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download Collocations as CSV",
            data=csv,
            file_name='collocations.csv',
            mime='text/csv',
        )
    else:
        st.error("Please enter some text to analyze.")

# Run the app
if __name__ == "__main__":
    main()
