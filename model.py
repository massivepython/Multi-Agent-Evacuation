"""
Model module for the evacuation simulation.
Defines the room environment with smoke, obstacles, and exits.
"""

import numpy as np
from mesa import Model
from mesa.time import RandomActivation
from mesa.space import MultiGrid
from mesa.datacollection import DataCollector
from agent import Person, Door, Obstacle, Smoke


class EvacuationModel(Model):
    """
    A model simulating evacuation from a smoke-filled room.
    
    The room is a rectangular grid with:
    - Exits/doors at specified locations
    - Obstacles that block movement
    - Constant smoke across entire room (appears all at once)
    - People with varying altruism and health levels trying to escape
    """
    
    def __init__(self, width=50, height=30, num_people=50, 
                 altruistic_population_percent=50.0,
                 num_obstacles=10, smoke_level=50.0,
                 health_min=100, health_max=100,
                 speed_min=0.7, speed_max=1.1,
                 damage_per_smoke_unit=0.05,
                 panic_increase_per_smoke=0.00033,
                 panic_randomness_factor=0.4,
                 altruism_threshold=0.3, # Kept for compatibility but effectively unused with binary logic
                 help_health_threshold=30.0,
                 help_speed_ratio=0.5,
                 helping_radius=2,
                 helping_speed_percentage=80,
                 max_people_per_cell=2,
                 run_mode="Interactive Simulation",
                 batch_size=30):
        """
        Initialize the evacuation model.
        
        Args:
            width: Width of the room (grid)
            height: Height of the room (grid)
            num_people: Number of people in the room
            altruistic_population_percent: Percentage of agents that are altruistic (0-100)
            num_obstacles: Number of obstacle blocks
            smoke_level: Constant smoke level across entire room (0-100)
            health_min: Minimum initial health for agents
            health_max: Maximum initial health for agents
            speed_min: Minimum base speed for agents
            speed_max: Maximum base speed for agents
            damage_per_smoke_unit: Health damage per unit of smoke per step
            panic_increase_per_smoke: Panic increase per unit of smoke per step
            panic_randomness_factor: Multiplier for panic effect on random moves
            altruism_threshold: Minimum altruism to consider helping
            help_health_threshold: Minimum health required to help others
            help_speed_ratio: Speed ratio threshold for helping (neighbor.speed < ratio * self.speed)
            helping_radius: Radius within which agents can help
            helping_speed_percentage: Percentage of helper's speed (0-100) used for the pair
            max_people_per_cell: Maximum number of people allowed per cell
            run_mode: "Interactive Simulation" or "Batch Analysis"
            batch_size: Number of iterations for the internal batch run.
        """
        super().__init__()
        
        self.batch_result_message = ""
        self.run_mode = run_mode
        self.batch_size = batch_size
        
        # Store initialization params for batch running
        self._init_params = {
            'width': width, 'height': height, 'num_people': num_people,
            'altruistic_population_percent': altruistic_population_percent,
            'num_obstacles': num_obstacles, 'smoke_level': smoke_level,
            'health_min': health_min, 'health_max': health_max,
            'speed_min': speed_min, 'speed_max': speed_max,
            'damage_per_smoke_unit': damage_per_smoke_unit,
            'panic_increase_per_smoke': panic_increase_per_smoke,
            'panic_randomness_factor': panic_randomness_factor,
            'altruism_threshold': altruism_threshold,
            'help_health_threshold': help_health_threshold,
            'help_speed_ratio': help_speed_ratio,
            'helping_radius': helping_radius,
            'helping_speed_percentage': helping_speed_percentage,
            'max_people_per_cell': max_people_per_cell
        }

        # Proceed with normal initialization so the visualization has something to render
        self.width = width
        self.height = height
        self.num_people = num_people
        self.altruistic_population_percent = altruistic_population_percent
        self.smoke_level = smoke_level
        self.health_min = health_min
        self.health_max = health_max
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.damage_per_smoke_unit = damage_per_smoke_unit
        self.panic_increase_per_smoke = panic_increase_per_smoke
        self.panic_randomness_factor = panic_randomness_factor
        self.helping_speed_percentage = helping_speed_percentage
        self.altruism_threshold = altruism_threshold
        self.help_health_threshold = help_health_threshold
        self.help_speed_ratio = help_speed_ratio
        self.helping_radius = helping_radius
        self.max_people_per_cell = max_people_per_cell
        
        # Load map layout from file
        self.obstacles = []
        self.red_obstacles = []  # Initialize BEFORE loading map
        self.exits = []
        self.spawn_points = []
        self.width, self.height = self._load_map_from_file("map.txt")
        
        # Create grid (multiple agents can occupy same cell)
        self.grid = MultiGrid(self.width, self.height, torus=False)
        self.height = height
        self.num_people = num_people
        self.altruistic_population_percent = altruistic_population_percent
        self.smoke_level = smoke_level
        self.health_min = health_min
        self.health_max = health_max
        self.speed_min = speed_min
        self.speed_max = speed_max
        self.damage_per_smoke_unit = damage_per_smoke_unit
        self.panic_increase_per_smoke = panic_increase_per_smoke
        self.panic_randomness_factor = panic_randomness_factor
        self.altruism_threshold = altruism_threshold
        self.help_health_threshold = help_health_threshold
        self.help_speed_ratio = help_speed_ratio
        self.helping_radius = helping_radius
        self.max_people_per_cell = max_people_per_cell
        
        # Load map layout from file
        self.obstacles = []
        self.red_obstacles = []
        self.exits = []
        self.spawn_points = []
        self.width, self.height = self._load_map_from_file("map.txt")
        
        # Create grid (multiple agents can occupy same cell)
        self.grid = MultiGrid(self.width, self.height, torus=False)
        
        # Scheduler for agent activation
        self.schedule = RandomActivation(self)
        
        # Per-exit escape counters
        self.exit_counts = {pos: 0 for pos in self.exits}

        # Place Door agents on exits for visualization
        self._place_doors()
        
        # Place Obstacle agents on obstacles for visualization
        self._place_obstacle_agents()
        
        # Place Red Obstacle agents
        self._place_red_obstacle_agents()
        
        # Smoke grid - initially empty (0.0), spreads from source
        self.smoke_grid = np.zeros((self.width, self.height))
        
        # Define smoke source (middle of right wall)
        mid_y = self.height // 2
        self.smoke_source = [(self.width - 1, mid_y), (self.width - 1, mid_y + 1)]
        self.smoke_radius = 0
        
        # Optimization structures for smoke spreading
        self.smoke_agents_grid = [[None for _ in range(self.height)] for _ in range(self.width)]
        self.cells_sorted_by_distance = []
        self.smoke_spread_index = 0
        
        # Place Smoke agents on all cells for visualization (initially 0)
        self._place_smoke_agents()
        
        # Precompute distances for optimized spreading
        self._precompute_smoke_distances()
        
        # Tracking
        self.escaped_count = 0
        self.dead_count = 0
        self.steps = 0
        self.total_people = 0  # actual number of Person agents placed
        self.running = True  # Model needs this to run
        
        # Create agents
        self._create_agents()
        
        # Data collection
        self.datacollector = DataCollector(
            model_reporters={
                "Escaped": lambda m: m.escaped_count,
                "Dead": lambda m: m.dead_count,
                "Remaining": lambda m: m.get_remaining_count(),
                "Average Health": lambda m: m.get_average_health(),
                "Average Speed": lambda m: m.get_average_speed(),
                "Average Panic": lambda m: m.get_average_panic(),
            },
            agent_reporters={
                "Health": "health",
                "Speed": "speed",
                "Altruism": "altruism",
                "Escaped": "escaped",
                "Position": "pos"
            }
        )
        
    def _load_map_from_file(self, filename):
        """
        Load grid layout from a text file.
        
        Symbols:
        . = Empty
        # = Obstacle
        E = Exit
        S = Spawn Point (Empty but valid for spawning)
        
        Returns (width, height)
        """
        try:
            with open(filename, 'r') as f:
                # Do not strip lines to preserve indentation/spacing
                lines = [line.rstrip('\n') for line in f if line.strip()]
        except FileNotFoundError:
            print(f"Map file {filename} not found. Using default 50x30 empty grid.")
            return 50, 30

        height = len(lines)
        width = max(len(line) for line in lines)

        # Parse the map from top to bottom
        # Grid coordinate system: (0,0) is bottom-left
        # File coordinate system: line 0 is top
        # We need to invert Y axis
        
        for y, line in enumerate(reversed(lines)):
            # Pad line with spaces if shorter than width
            padded_line = line.ljust(width)
            for x, char in enumerate(padded_line):
                if x >= width: break
                
                pos = (x, y)
                
                if char == '#':
                    self.obstacles.append(pos)
                elif char == 'X':
                    # Red obstacles are obstacles AND need special visualization
                    self.obstacles.append(pos)
                    self.red_obstacles.append(pos)
                elif char == 'E':
                    self.exits.append(pos)
                elif char == 'S':
                    self.spawn_points.append(pos)
                # '.' is just empty, no action needed
        
        return width, height

    def _place_doors(self):
        """Place Door agents at each exit position (for rendering)."""
        for i, exit_pos in enumerate(self.exits):
            door = Door(f"door_{i}", self)
            # Doors are non-blocking; just place for visualization
            self.grid.place_agent(door, exit_pos)
            # Initialize counter for this exit
            if exit_pos not in self.exit_counts:
                self.exit_counts[exit_pos] = 0
    
    def _place_obstacle_agents(self):
        """Place Obstacle agents at each obstacle position (for visualization)."""
        # Only place standard Obstacle agents where there ISN'T a RedObstacle
        # to avoid stacking two agents on the same spot
        red_set = set(self.red_obstacles)
        for i, obs_pos in enumerate(self.obstacles):
            if obs_pos not in red_set:
                obstacle = Obstacle(f"obstacle_{i}", self)
                self.grid.place_agent(obstacle, obs_pos)

    def _place_red_obstacle_agents(self):
        """Place RedObstacle agents."""
        from agent import RedObstacle
        for i, obs_pos in enumerate(self.red_obstacles):
            obstacle = RedObstacle(f"red_obstacle_{i}", self)
            self.grid.place_agent(obstacle, obs_pos)
    
    def _place_smoke_agents(self):
        """Place Smoke agents at all positions (for visualization)."""
        smoke_id = 0
        for x in range(self.width):
            for y in range(self.height):
                # Initialize with 0.0 smoke to match the empty grid
                smoke = Smoke(f"smoke_{smoke_id}", self, 0.0)
                self.grid.place_agent(smoke, (x, y))
                self.smoke_agents_grid[x][y] = smoke # Store reference for fast access
                smoke_id += 1
    
    def _precompute_smoke_distances(self):
        """
        Pre-calculate the distance of every cell to the smoke source.
        Sorts cells by distance to allow O(1) smoke spreading per step.
        """
        cells = []
        for x in range(self.width):
            for y in range(self.height):
                # Calculate min distance to any source point
                dist = min(
                    np.sqrt((x - sx)**2 + (y - sy)**2) 
                    for sx, sy in self.smoke_source
                )
                cells.append((dist, x, y))
        
        # Sort by distance (ascending)
        self.cells_sorted_by_distance = sorted(cells, key=lambda item: item[0])

    def _create_agents(self):
        """Create person agents and place them on spawn points (1 agent per 'S')."""
        # If no spawn points defined, fallback to random free spots (preserving old logic just in case)
        if not self.spawn_points:
            possible_spawns = []
            for x in range(self.width):
                for y in range(self.height):
                    if (x, y) not in self.obstacles and (x, y) not in self.exits:
                        possible_spawns.append((x, y))
            self.spawn_points = possible_spawns
            # Limit fallback to a default number if we had to auto-generate
            # But if map has 'S', we use exactly that count
            
        # Override num_people to match the map's spawn points
        self.num_people = len(self.spawn_points)
        self.total_people = 0
        
        # Calculate how many agents should be altruistic
        num_altruistic = int(self.num_people * (self.altruistic_population_percent / 100.0))
        
        # Create a list of altruism flags (True/False) and shuffle it
        # This ensures exact percentage is met
        altruism_flags = [True] * num_altruistic + [False] * (self.num_people - num_altruistic)
        self.random.shuffle(altruism_flags)

        for i, pos in enumerate(self.spawn_points):
            # Get binary altruism trait
            is_altruistic = altruism_flags[i]
            # Convert to float 1.0/0.0 for compatibility with agent logic, or update agent logic
            # Using 1.0 for True and 0.0 for False is robust
            altruism_value = 1.0 if is_altruistic else 0.0
            
            # Generate health from specified range
            health = np.random.uniform(self.health_min, self.health_max)
            
            # Create agent
            agent = Person(i, self, altruism=altruism_value, health=health)
            
            # Place agent strictly on the spawn point
            self.grid.place_agent(agent, pos)
            self.schedule.add(agent)
            self.total_people += 1
                
    def spread_smoke(self):
        """
        Spread smoke using pre-computed distances (Optimized O(k)).
        Only iterates over new cells that the smoke has just reached.
        """
        # Increase radius
        self.smoke_radius += 2.0
        
        # Iterate through the sorted list, starting from where we left off
        while self.smoke_spread_index < len(self.cells_sorted_by_distance):
            dist, x, y = self.cells_sorted_by_distance[self.smoke_spread_index]
            
            if dist <= self.smoke_radius:
                # This cell is now covered by smoke
                self.smoke_grid[x][y] = self.smoke_level
                
                # Update the Smoke agent directly (no search needed)
                smoke_agent = self.smoke_agents_grid[x][y]
                if smoke_agent:
                    smoke_agent.smoke_level = self.smoke_level
                
                # Move to next cell
                self.smoke_spread_index += 1
            else:
                # The next cell is too far, so we stop here for this step
                break
            
    def run_batch_analysis(self):
        """Execute the batch analysis logic."""
        print(f"\n--- Starting Batch Analysis (N={self.batch_size}) ---")
        dead_counts = []
        
        # Progress bar loop
        for i in range(self.batch_size):
            # Visual progress bar for console
            progress = int((i / self.batch_size) * 20)
            bar = "[" + "=" * progress + " " * (20 - progress) + "]"
            print(f"\r{bar} Running simulation {i+1}/{self.batch_size}", end="", flush=True)
            
            # Create temp model
            # Use **self._init_params to unpack all stored args
            # Force run_mode="Interactive Simulation" to prevent recursion
            params = self._init_params.copy()
            params['run_mode'] = "Interactive Simulation"
            
            temp_model = EvacuationModel(**params)
            temp_model.run_model(max_steps=200)
            dead_counts.append(temp_model.dead_count)
            
        print(f"\nBatch Complete.")
        
        avg_dead = np.mean(dead_counts)
        self.batch_result_message = f"Batch Result (N={self.batch_size}): Avg Dead = {avg_dead:.2f}"
        print(self.batch_result_message)

    def step(self):
        """Advance the model by one step."""
        if self.run_mode == "Batch Analysis":
            self.run_batch_analysis()
            self.running = False
            return

        self.steps += 1
        
        # Spread smoke (no-op: smoke is constant)
        self.spread_smoke()
        
        # Activate all agents
        self.schedule.step()
        
        # Update dead count after agent actions (count Person agents only)
        self.dead_count = len([a for a in self.schedule.agents 
                              if isinstance(a, Person) and a.health <= 0 and not a.escaped])
        
        # Collect data
        self.datacollector.collect(self)
        
        # Check if simulation should end (no more active Person agents)
        active_agents = [a for a in self.schedule.agents 
                        if isinstance(a, Person) and not a.escaped and a.health > 0]
        if len(active_agents) == 0:
            self.running = False
    
    def get_remaining_count(self):
        """Return the number of people still alive and inside."""
        count = 0
        for agent in self.schedule.agents:
            if isinstance(agent, Person) and not agent.escaped and agent.health > 0:
                count += 1
        return count
    
    def get_average_health(self):
        """Return the average health of alive people inside."""
        health_values = []
        for agent in self.schedule.agents:
            if isinstance(agent, Person) and not agent.escaped and agent.health > 0:
                health_values.append(agent.health)
        
        if len(health_values) > 0:
            return np.mean(health_values)
        else:
            return 0
    
    def get_average_speed(self):
        """Return the average speed of alive people inside (scaled by 100)."""
        speed_values = []
        for agent in self.schedule.agents:
            if isinstance(agent, Person) and not agent.escaped and agent.health > 0:
                speed_values.append(agent.speed)
        
        if len(speed_values) > 0:
            return np.mean(speed_values) * 100.0
        else:
            return 0
    
    def get_average_panic(self):
        """Return the average panic level of alive people inside."""
        panic_values = []
        for agent in self.schedule.agents:
            if isinstance(agent, Person) and not agent.escaped and agent.health > 0:
                panic_values.append(agent.panic_level)
        
        if len(panic_values) > 0:
            return np.mean(panic_values)
        else:
            return 0
            
    def run_model(self, max_steps=500):
        """Run the model until completion or max steps."""
        for _ in range(max_steps):
            if not self.running:
                break
            self.step()

