import numpy as np
import mesa
from mesa import Agent

###################
### AGENT CLASS ###
###################

# Define the NavigationAgent class
class NavigationAgent(Agent):
    def __init__(self, model, vision=5):  # Default vision range of 5 cells
        super().__init__(model)  # MESA `Agent` class initialization, auto-assigns unique_id in Mesa 3.0
        # Attributes of each agent
        self.found_exit = False  # Track if agent has reached the exit
        self.previous_pos = None  # Previous position of the agent
        self.vision = vision  # Vision range of the agent

    # Function to move the agent towards the exit
    def move_towards_exit(self):
        self.previous_pos = self.pos  # Store the current position before moving
        # MESA `get_neighborhood` function retrieves nearby cells based on vision range
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        min_distance = float('inf')  # Start with a very large distance
        best_step = None  # Initialize best step as None

        # Check each possible step to find the one closest to the exit
        for step in possible_steps:
            # Only consider steps that don't have obstacles and have less than 8 agents
            if step not in self.model.obstacles and len(self.model.grid.get_cell_list_contents(step)) < 8:
                dist = euclidean_distance(step, self.model.exit_location)  # Distance to the exit
                if dist < min_distance:
                    min_distance = dist
                    best_step = step  # Update best step to be closer to the exit
        if best_step:
            # MESA `move_agent` function moves the agent to the new cell
            self.model.grid.move_agent(self, best_step)

    # Function to move the agent randomly if the exit is not in sight
    def move_randomly(self):
        self.previous_pos = self.pos
        possible_steps = self.model.grid.get_neighborhood(self.pos, moore=True, include_center=False)

        # Filter steps to only those without obstacles and with fewer than 8 agents
        valid_steps = [step for step in possible_steps if
                       step not in self.model.obstacles and len(self.model.grid.get_cell_list_contents(step)) < 8]

        if valid_steps:
            # Randomly choose a valid position and move there
            random_step = self.random.choice(valid_steps)
            # MESA `move_agent` function moves the agent to the chosen position
            self.model.grid.move_agent(self, random_step)

    # Define the actions the agent will take in each step
    def step(self):
        # If the agent is at the exit, mark as exited
        if self.pos == self.model.exit_location:
            self.found_exit = True  # Set the agent's exit status to True
            self.model.grid.remove_agent(self)  # MESA function to remove the agent from the grid
            self.remove()  # self.remove() to remove from AgentSet
            self.model.cumulative_exited += 1  # Count this agent in cumulative exited agents
        else:
            # MESA `get_neighborhood` checks the agent's vision area for the exit
            vision_area = self.model.grid.get_neighborhood(self.pos, moore=True, radius=self.vision,
                                                           include_center=False)
            exit_in_vision = self.model.exit_location in vision_area

            # Move towards the exit if it's in sight, otherwise move randomly
            if exit_in_vision:
                self.move_towards_exit()
            else:
                self.move_randomly()