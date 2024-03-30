import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.animation import FuncAnimation

# Parameters
num_lanes = 3
lane_width = 3
car_length = 5
car_width = 2
frame_interval = 200  # in milliseconds
time_step = 0.1  # in seconds
congestion_threshold = 20  # Adjust as needed

# Initialize DataFrame to store car data
car_data = pd.DataFrame(columns=['Position', 'Velocity', 'Lane', 'Time_in_Congestion', 'Total_Time_in_Congestion',
                                  'Initial_Speed', 'Final_Speed', 'Color', 'Changed_Lane', 'Label'])

# Function to generate new cars
def generate_cars():
    while True:
        if car_data['Time_in_Congestion'].sum() < congestion_threshold:
            position = np.random.uniform(0, 100)
            velocity = np.random.uniform(5, 10)  # Adjusted velocity range for slower movement
            lane = np.random.randint(0, num_lanes)
            color = 'blue'  # Default color
            new_car = {'Position': position, 'Velocity': velocity, 'Lane': lane, 'Time_in_Congestion': 0,
                       'Total_Time_in_Congestion': 0, 'Initial_Speed': velocity, 'Final_Speed': velocity, 'Color': color, 'Changed_Lane': False, 'Label': ''}
            yield new_car

# Create generator for new cars
car_generator = generate_cars()

# Function to update positions of cars
def update(frame):
    global car_data
    # Generate new cars if there is no congestion
    if car_data['Time_in_Congestion'].sum() < congestion_threshold:
        new_car = next(car_generator)
        car_data = pd.concat([car_data, pd.DataFrame([new_car])], ignore_index=True)

    # Simulate acceleration and deceleration
    car_data['Velocity'] -= 0.05
    car_data['Velocity'] = car_data['Velocity'].clip(lower=0)  # Ensure velocity doesn't go negative

    # Update positions
    car_data['Position'] -= car_data['Velocity'] * time_step  # Adjusted for time step (from bottom to top)

    # Simulate lane changes
    lane_change_chance = 0.02
    for i, car in car_data.iterrows():
        if np.random.rand() < lane_change_chance and not car['Changed_Lane']:
            new_lane = (car['Lane'] + np.random.choice([-1, 1])) % num_lanes
            car_data.at[i, 'Lane'] = new_lane
            car_data.at[i, 'Changed_Lane'] = True
            car_data.at[i, 'Label'] = 'Lane Change'

    # Simulate sudden slowdown
    slowdown_chance = 0.05
    for i, car in car_data.iterrows():
        if np.random.rand() < slowdown_chance:
            car_data.at[i, 'Velocity'] *= 0.8  # Reduce velocity abruptly
            car_data.at[i, 'Time_in_Congestion'] += time_step  # Adjusted for time step
            if car_data.at[i, 'Time_in_Congestion'] == time_step:  # Adjusted for time step
                car_data.at[i, 'Initial_Speed'] = car['Velocity']

    # Update total time spent in congestion
    car_data['Total_Time_in_Congestion'] += car_data['Time_in_Congestion']

    # Sort cars based on their positions (to create a natural flow)
    car_data = car_data.sort_values(by='Position', ascending=False).reset_index(drop=True)

    # Assign colors based on conditions
    for i, car in car_data.iterrows():
        if car['Changed_Lane']:
            car_data.at[i, 'Color'] = 'green'  # Color for cars that changed lanes
        elif car['Time_in_Congestion'] > 0:
            car_data.at[i, 'Color'] = 'red'  # Color for cars in congestion
        else:
            car_data.at[i, 'Color'] = 'blue'  # Default color

    # Plot lanes separators and cars
    plt.clf()
    ax = plt.gca()
    for i in range(num_lanes - 1):
        y = (i + 1) * lane_width
        ax.add_patch(Rectangle((0, y - 0.5), 100, 1, color='black'))
    for _, car in car_data.iterrows():
        ax.add_patch(Rectangle((car['Position'] - car_length / 2, car['Lane'] * lane_width - car_width / 2), car_length, car_width, color=car['Color']))
        if car['Changed_Lane']:
            ax.text(car['Position'], car['Lane'] * lane_width, car['Label'], ha='center', va='center', color='white')
    plt.xlim(0, 100)
    plt.ylim(0, num_lanes * lane_width)
    plt.xlabel('Position')
    plt.ylabel('Lane')
    plt.title('Traffic Simulation')

# Create animation
fig, ax = plt.subplots()
ani = FuncAnimation(fig, update, interval=frame_interval)

# Show animation
plt.show()

# Calculate time taken for cars in congestion
car_data['Time_in_Congestion'] = car_data['Total_Time_in_Congestion'] - car_data['Time_in_Congestion']

# Generate dataset for ghost

