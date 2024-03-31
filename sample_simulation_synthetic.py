#!/usr/bin/env python
# coding: utf-8

# In[1]:


import dash
from dash import dcc, html
from dash.dependencies import Input, Output
import plotly.graph_objs as go
import pandas as pd
import random

# Initialize the Dash app
app = dash.Dash(__name__)

# Define the layout of the app
app.layout = html.Div([
    html.H1("Traffic Simulation Dashboard"),
    dcc.Graph(id="traffic-simulation"),
    dcc.Interval(
        id='interval-component',
        interval=1000,  # Update every 1 second
        n_intervals=0
    )
])

# Define the callback to update the traffic simulation
@app.callback(
    Output('traffic-simulation', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_simulation(n):
    # Generate synthetic traffic simulation data
    num_cars = 50
    lanes = ['Lane 1', 'Lane 2', 'Lane 3']
    velocities = [random.randint(10, 30) for _ in range(num_cars)]
    distances = [random.uniform(0, 10) for _ in range(num_cars)]

    # Create a DataFrame for the simulation data
    simulation_data = pd.DataFrame({
        'Car ID': range(1, num_cars + 1),
        'Lane': [random.choice(lanes) for _ in range(num_cars)],
        'Velocity': velocities,
        'Distance': distances
    })

    # Create traces for each lane
    traces = []
    for lane in lanes:
        lane_data = simulation_data[simulation_data['Lane'] == lane]
        trace = go.Scatter(
            x=lane_data['Distance'],
            y=[lane] * len(lane_data),
            mode='markers',
            marker=dict(
                size=10,
                color='blue',
                line=dict(width=2, color='DarkSlateGrey')
            ),
            name=lane
        )
        traces.append(trace)

    # Define layout for the plot
    layout = dict(
        title='Traffic Simulation',
        xaxis=dict(title='Distance'),
        yaxis=dict(title='Lane', showticklabels=False),
        showlegend=False,
        height=600
    )

    # Create the figure
    fig = go.Figure(data=traces, layout=layout)
    return fig

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True, port=9091)

# In[ ]:




