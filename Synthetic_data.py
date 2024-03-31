#!/usr/bin/env python
# coding: utf-8

# ## Synthetic Data Preparation

# Based on the inital simulation we created a synthetic data on considering the 
#  * velocity of the moving car
#  * distance between cars 
#  * time and car referecne 

# In[2]:


import random
import pandas as pd
# Simulation parameters
total_time = 3600  # 1 hour
time_step = 5      
num_lanes = 3      
max_velocity = 30  
min_velocity = 10  
common_distance = 5 

# Initialize data structure to store simulation data
simulation_data = []


for t in range(0, total_time, time_step):
    for car_id in range(1, random.randint(10, 30)):  
        lane_number = random.randint(1, num_lanes)
        current_velocity = random.uniform(min_velocity, max_velocity)
        following_distance = random.uniform(0, common_distance)
        
        ahead_distance = random.uniform(0, common_distance)
        
        simulation_data.append({
            "time": t,
            "car_id": car_id,
            "lane_number": lane_number,
            "current_velocity": current_velocity,
            "following_distance": following_distance,
            "ahead_distance": ahead_distance
        })

pd.DataFrame(simulation_data).to_csv("trail_1.csv", index=False)


# In[3]:


data=pd.DataFrame(simulation_data)
data.head()


# In[4]:


# Sort the data by 'car_id' and 'time' to ensure proper order
data.sort_values(by=['time', 'car_id'], inplace=True)

# Initialize a new column for labels
data['label'] = 0

# Iterate through each row
for index, row in data.iterrows():
    # Get the previous row for the same car_id
    previous_rows = data[(data['car_id'] == row['car_id']) & (data['time'] < row['time'])]
    
    if not previous_rows.empty:
        previous_row = previous_rows.iloc[-1]
        
        # Compare 'ahead_distance' of current and previous rows
        if row['ahead_distance'] > previous_row['ahead_distance']:
            data.at[index, 'label'] = 1
        elif row['ahead_distance'] < previous_row['ahead_distance']:
            data.at[index, 'label'] = -1
        else:
            data.at[index, 'label'] = 0

# Save the labeled data
data.to_csv('label_trail_1.csv', index=False)


# In[5]:


# Initialize counter for correct labels
labeled_data=data
correct_labels = 0

# Iterate through each row skipping the first row
for index, row in labeled_data.iterrows():
    if index == 0:
        continue  # Skip the first row
    
    # Get the actual change in 'ahead_distance'
    actual_change = labeled_data.at[index, 'ahead_distance'] - labeled_data.at[index - 1, 'ahead_distance']
    
    # Check if the label matches the actual change
    if (row['label'] == 1 and actual_change > 0) or \
       (row['label'] == -1 and actual_change < 0) or \
       (row['label'] == 0 and actual_change == 0):
        correct_labels += 1

# Calculate percentage of correct labels
total_labels = len(labeled_data) - 1  # Exclude the first row which doesn't have a previous row
accuracy = (correct_labels / total_labels) * 100

print("Validation results:")
print(f"Total labeled instances: {total_labels}")
print(f"Correctly labeled instances: {correct_labels}")
print(f"Accuracy: {accuracy:.2f}%")


# In[ ]:




