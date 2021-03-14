import streamlit as st
from apps import data_explore, comparison, feature_extraction, intro_and_dataexploration, sensitivity_scpecificity,text_processing 
st.set_page_config(layout="wide")

class MultiApp:

    def __init__(self):
        self.apps = []

    def add_app(self, title, func):
        """Adds a new application.
        Parameters
        ----------
        func:
            the python function to render this app.
        title:
            title of the app. Appears in the dropdown in the sidebar.
        """
        self.apps.append({
            "title": title,
            "function": func
        })

    def run(self):
        #app = st.sidebar.selectbox(
        app = st.selectbox(
            'Navigation',
            self.apps,
            format_func=lambda app: app['title'])

        app['function'](5)

app = MultiApp()

# Add all your application here
app.add_app("Introduction", intro_and_dataexploration.app)
app.add_app("Data Exploration", data_explore.app)
app.add_app("Text Processing", text_processing.app)
app.add_app("Feature Extraction", feature_extraction.app)
app.add_app("Classifier model comparison", comparison.app)
app.add_app("Sensitivity & Specificity", sensitivity_scpecificity.app)
# The main ap
app.run()