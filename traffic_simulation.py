import pygame
import random
import pandas as pd


# Initialize Pygame
pygame.init()

# Set up the screen
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 900
LANE_WIDTH = 200
LANE_COUNT = 3
CAR_WIDTH = 30
CAR_HEIGHT = 50
MIN_DISTANCE = 20  # Minimum distance between cars
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
ACCELERATION = 0.1
BREAKING_DISTANCE = 30  # Distance at which braking starts
PROBABILITY_OF_ADDING_CARS = 0.3
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Highway Traffic Simulation")


# Data collection
simulation_data = []

# Define Bezier Curve
def cubic_bezier(t, p0, p1, p2, p3):
    return ((1 - t)**3) * p0 + 3 * ((1 - t)**2) * t * p1 + 3 * (1 - t) * (t**2) * p2 + (t**3) * p3


# Define car class
class Car(pygame.sprite.Sprite):
    car_count = 0
    def __init__(self, lane, idx):
        super().__init__()
        self.image = pygame.Surface((CAR_WIDTH, CAR_HEIGHT))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.centerx = lane * LANE_WIDTH + LANE_WIDTH // 2
        self.rect.bottom = SCREEN_HEIGHT + BREAKING_DISTANCE
        self.speed = random.randint(5, 5)  # Random initial speed
        self.max_speed = 5
        self.lane = lane
        self.min_speed = 0.1
        self.idx = idx
        self.previous_speed = self.speed
        self.start_time = pygame.time.get_ticks() / 1000

        # Generate unique car ID based on total cars generated
        self.car_id = Car.car_count
        Car.car_count += 1

    def update(self):
        self.rect.y -= self.speed

        if random.random() < 0.007:
            self.speed = max(self.min_speed, self.speed - random.randint(2,3))

        # Check if the car is at the 0th index or if there are no cars in front
        if self.idx == 0:
            # Accelerate if the car is at the top of the window
            if self.rect.top <= 0:
                self.speed = self.max_speed
            else:
                self.speed = min(self.speed + ACCELERATION, self.max_speed)

        # Check for collision with the car in front
        if self.idx > 0:
            prev_car = lane_positions[self.lane][self.idx - 1]
            distance_to_front = self.rect.top - prev_car.rect.bottom

            # Accelerate if there is enough distance to the car in front
            if distance_to_front > BREAKING_DISTANCE:
                self.speed = min(self.speed + ACCELERATION, self.max_speed)

            # Slow down if the car is too close to the car in front
            elif distance_to_front < MIN_DISTANCE:
                t = 1 - (distance_to_front / MIN_DISTANCE)  # Normalize distance
                speed_reduction = cubic_bezier(t, 0, 0.4, 0.6, 1) * ACCELERATION * (MIN_DISTANCE - distance_to_front)
                self.speed = max(self.min_speed, self.speed - speed_reduction)

            # Add data to simulation_data
            simulation_data.append({
                "car_id": self.car_id,
                "lane": self.lane,
                "change_in_velocity": self.previous_speed - self.speed,
                "initial_velocity": self.previous_speed,
                "final_velocity": self.speed,
                "ahead_distance": distance_to_front,
                "label": "Slow down" if self.previous_speed > self.speed else "Accelerate" if self.previous_speed < self.speed else "Maintain",
                "time_to_destination": (pygame.time.get_ticks() / 1000) - self.start_time  # Time to reach the destination
            })

        # Check for collision with the car behind
        if len(lane_positions[self.lane]) > 1:
            for i in range(self.idx + 1, len(lane_positions[self.lane])):
                next_car = lane_positions[self.lane][i]
                distance_to_behind = next_car.rect.bottom - self.rect.bottom

                if abs(distance_to_behind) < MIN_DISTANCE:
                    t = 1 - (distance_to_behind / MIN_DISTANCE)  # Normalize distance
                    speed_reduction = cubic_bezier(t, 0, 0.4, 0.6, 1) * ACCELERATION * (MIN_DISTANCE - distance_to_behind)
                    next_car_speed = max(self.min_speed, next_car.speed - speed_reduction)

                    # Ensure the next_car doesn't go faster than us
                    if next_car_speed < next_car.speed:
                        next_car.speed = next_car_speed
                    
        self.previous_speed = self.speed

    # def update(self):
    #     self.rect.y -= self.speed

    #     if random.random() < 0.01: 
    #         self.speed = max(0, self.speed - 1)

    #     # Check for collision with the car behind
    #     if len(lane_positions[self.lane]) > 1:
    #         for i in range(self.idx + 1, len(lane_positions[self.lane])):
    #             next_car = lane_positions[self.lane][i]
    #             distance_to_behind = next_car.rect.bottom - self.rect.bottom
    #             if abs(distance_to_behind) < MIN_DISTANCE:
    #                 # Calculate the desired speed reduction based on distance
    #                 t = 1 - (distance_to_behind / MIN_DISTANCE)  # Normalize distance
                    
    #                 # Evaluate the Bezier curve for desired speed reduction
    #                 speed_reduction = cubic_bezier(t, 0, 0.4, 0.6, 1) * ACCELERATION * (MIN_DISTANCE - distance_to_behind)

    #                 # Ensure the next_car doesn't go faster than us
    #                 next_car_speed = max(0, self.speed - speed_reduction)
                    
    #                 # Ensure the next_car doesn't go faster than us
    #                 if next_car_speed < next_car.speed:
    #                     next_car.speed = next_car_speed

                # elif distance_to_behind > BREAKING_DISTANCE and self.speed < self.max_speed:
                #     # Calculate the desired speed reduction based on distance
                #     t = 1 + (BREAKING_DISTANCE / MIN_DISTANCE)  # Normalize distance
                    
                #     # Evaluate the Bezier curve for desired speed reduction
                #     speed_increase = cubic_bezier(t, 0, 0.4, 0.6, 1) * ACCELERATION * (MIN_DISTANCE + distance_to_behind)

                #     self.speed = min(self.speed + speed_increase, self.max_speed)
 
            # Accelerate if a gap builds up after the car ahead moves forward  

        if self.rect.bottom < 0:
            lane_positions[self.lane].remove(self)
            self.kill()  # Remove the sprite from all groups
            for i, car in enumerate(lane_positions[self.lane]):
                car.idx = i


# Create sprite groups
cars = pygame.sprite.Group()

# List to keep track of car positions in each lane
lane_positions = [[] for _ in range(LANE_COUNT)]

# Main loop
clock = pygame.time.Clock()
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Update
    cars.update()

    # Check congestion and add new cars randomly to a lane
    lane = random.randint(0, LANE_COUNT - 1)
    new_idx = len(lane_positions[lane])
    if len(lane_positions[lane]) == 0 or lane_positions[lane][-1].rect.bottom < SCREEN_HEIGHT - MIN_DISTANCE:
        if random.random() < PROBABILITY_OF_ADDING_CARS:  # Adjust the probability of adding a new car
            enough_space = True
            if len(lane_positions[lane]) > 0:
                last_car_bottom = lane_positions[lane][-1].rect.bottom
                for car in lane_positions[lane][:-1]:
                    if abs(car.rect.bottom - last_car_bottom) < MIN_DISTANCE:
                        enough_space = False
                        break

            if enough_space:
                new_idx = len(lane_positions[lane])
                new_car = Car(lane, new_idx)
                cars.add(new_car)
                lane_positions[lane].append(new_car)

    # elif random.random() < PROBABILITY_OF_ADDING_CARS: 
    #     new_car = Car(lane, new_idx)
    #     cars.add(new_car)
    #     lane_positions[lane].append(new_car)


    # Draw
    screen.fill(BLACK)
    cars.draw(screen)

    pygame.display.flip()
    clock.tick(60)

# Quit Pygame
pygame.quit()

# Create a DataFrame from simulation_data
df = pd.DataFrame(simulation_data)

# Write simulation data to CSV
csv_filename = "simulation_data.csv"
df.to_csv(csv_filename, index=False)
