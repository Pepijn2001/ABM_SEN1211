import mesa
import numpy as np
from agent import NavigationAgent
from mesa import Model
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector

def find_exits(floor_plan):
    exit_locations = []
    rows, cols = len(floor_plan), len(floor_plan[0])

    # Process first row and last row
    for col in range(cols):
        if floor_plan[0][col] == 0:  # First row
            exit_locations.append((0, col))
        if floor_plan[rows - 1][col] == 0:  # Last row
            exit_locations.append((rows - 1, col))

    # Process first column and last column, excluding corners
    for row in range(1, rows - 1):
        if floor_plan[row][0] == 0:  # First column
            exit_locations.append((row, 0))
        if floor_plan[row][cols - 1] == 0:  # Last column
            exit_locations.append((row, cols - 1))
    return exit_locations

###################
### MODEL CLASS ###
###################

# Define the model class to handle the overall environment
class FloorPlanModel(Model):
    def __init__(self, floor_plan, num_agents, agent_vision):
        super().__init__()  # `Model` class initialization

        # Basic model settings
        self.num_agents = num_agents
        self.agent_vision = agent_vision
        self.grid = MultiGrid(floor_plan.shape[1], floor_plan.shape[0],
                              False)  # MESA grid with dimensions; False means no wrapping
        self.exit_location = find_exits(floor_plan)

        # Define obstacles and signs in the grid
        rows, cols = np.where(floor_plan == 1)
        self.obstacles = list(zip(rows, cols))
        self.signs = []

        # Initialize cumulative exited count
        self.cumulative_exited = 0

        # Initialize DataCollector (MESA tool for tracking metrics across steps)
        self.datacollector = DataCollector(
            model_reporters={
                "Active Agents": lambda m: len(m.agents),  # Count of agents still active
                "Exited Agents": lambda m: sum(
                    1 for agent in m.agents if isinstance(agent, NavigationAgent) and agent.found_exit),
                "Cumulative Exited Agents": lambda m: m.cumulative_exited,  # Cumulative exited count
                "Agents per Cell": self.count_agents_per_cell  # Counts agents in each cell
            },
            agent_reporters={
                "Found Exit": lambda a: a.found_exit if isinstance(a, NavigationAgent) else None
                # Reports exit status per agent
            }
        )

        self.place_agents(agent_vision)  # Place agents on the grid
        self.datacollector.collect(self)  # Collect data at the start of the simulation

    # Function to randomly place agents in the grid
    def place_agents(self, agent_vision):
        for i in range(self.num_agents):
            agent = NavigationAgent(self, vision=agent_vision)  # Create agent with initial vision range of 5 cells
            placed = False  # Track if the agent is successfully placed
            while not placed:
                x = self.random.randrange(self.grid.width)
                y = self.random.randrange(self.grid.height)
                cell_contents = self.grid.get_cell_list_contents((x, y))

                # Only place agent if cell has no obstacles and fewer than 8 agents
                if (x, y) not in self.obstacles and len(cell_contents) < 8:
                    self.grid.place_agent(agent, (x, y))  # MESA function to place agent in grid
                    placed = True  # Mark as placed
                    # if you want to see the initial placement of the agents, uncomment the line below. This will print the initial position of the agents
                    # print(f"Agent {i} placed at: ({x}, {y})")

    # Function to count agents in each cell
    def count_agents_per_cell(self):
        agent_counts = {}  # Dictionary to store agent counts by cell position
        # MESA `coord_iter` function iterates over grid cells and their contents
        for cell in self.grid.coord_iter():
            cell_contents, (x, y) = cell  # Unpack cell contents and coordinates
            # Count NavigationAgents in each cell
            nav_agent_count = sum(1 for obj in cell_contents if isinstance(obj, NavigationAgent))
            if nav_agent_count > 0:
                agent_counts[(x, y)] = nav_agent_count
        return agent_counts

    # Function to get the grid data for visualization
    def get_grid(self):
        # 0: empty, 1: obstacle, 2: agent, 3: exit, 4: sign
        grid_data = np.zeros((self.grid.width, self.grid.height))

        # Mark obstacles on the grid
        for x, y in self.obstacles:
            grid_data[y, x] = 1

        # Mark agents on the grid
        for agent in self.agents:
            if isinstance(agent, NavigationAgent):
                x, y = agent.pos
                grid_data[y, x] = 2

        # Mark signs and exit
        for x, y in self.signs:
            grid_data[y, x] = 4
        exit_x, exit_y = self.exit_location
        grid_data[exit_y, exit_x] = 3
        return grid_data

    # Model step function to update the simulation
    def step(self):
        self.agents.do("step")  # MESA 3.0 function to execute the `step` function of each agent
        self.datacollector.collect(self)  # MESA DataCollector collects metrics at each step