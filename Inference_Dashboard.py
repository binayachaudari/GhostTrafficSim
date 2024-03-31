#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pandas as pd

# Load the generated data (replace "traffic_simulation_data.csv" with your file)
data = pd.read_csv("labeled_data.csv")

# Train a Random Forest classifier on the data
X = data.drop(columns=["label"])
y = data["label"]
clf = RandomForestClassifier(random_state=42)
clf.fit(X, y)

# Define the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Traffic Simulation Dashboard"),
    html.Label("Select Feature:"),
    dcc.Dropdown(
        id="feature-dropdown",
        options=[{"label": col, "value": col} for col in X.columns],
        value=X.columns[0]
    ),
    dcc.Graph(id="feature-visualization"),
    html.Div(id="classification-report")
])

# Define callback to update visualization based on feature selection
@app.callback(
    Output("feature-visualization", "figure"),
    [Input("feature-dropdown", "value")]
)
def update_visualization(selected_feature):
    fig = px.histogram(data, x=selected_feature, color="label", barmode="overlay")
    return fig

# Define callback to display classification report
@app.callback(
    Output("classification-report", "children"),
    [Input("feature-dropdown", "value")]
)
def display_classification_report(selected_feature):
    # Assuming the model was already trained
    y_pred = clf.predict(X)
    report = classification_report(y, y_pred)
    return html.Div([
        html.H3("Classification Report"),
        html.Pre(report)
    ])

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)


# In[ ]:




