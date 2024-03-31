#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd

# Load the traffic simulation data
data = pd.read_csv("labeled_data.csv")

# Get unique car IDs and time steps
car_ids = data['car_id'].unique()
time_steps = sorted(data['time'].unique())

# Create a Dash application
app = dash.Dash(__name__)

# Define the layout of the dashboard
app.layout = html.Div([
    html.H1("Traffic Simulation Dashboard"),
    dcc.Graph(id='traffic-simulation-graph'),
    dcc.Interval(
        id='interval-component',
        interval=1*1000,  # in milliseconds
        n_intervals=0
    )
])

# Define callback to update the traffic simulation graph
@app.callback(
    Output('traffic-simulation-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_traffic_simulation(n):
    # Get the current time step
    current_time_step = time_steps[n % len(time_steps)]
    
    # Filter data for the current time step
    current_data = data[data['time'] == current_time_step]
    
    # Create traces for each car ID
    traces = []
    for car_id in car_ids:
        car_data = current_data[current_data['car_id'] == car_id]
        trace = go.Scatter(x=car_data['current_velocity'], y=car_data['following_distance'],
                           mode='markers', marker=dict(size=10), name=f"Car ID {car_id}")
        traces.append(trace)
    
    # Create layout for the graph
    layout = go.Layout(title=f"Traffic Simulation at Time Step {current_time_step}",
                       xaxis=dict(title='current_velocity'),
                       yaxis=dict(title='Following Distance'))
    
    # Return the figure
    return {'data': traces, 'layout': layout}

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=9090)


# In[ ]:




