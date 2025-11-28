"""
Agent module for the evacuation simulation.
Defines the Person agent with altruism-based behavior.
"""

import numpy as np
from mesa import Agent


class Person(Agent):
    """
    A person agent trying to escape from a smoke-filled room.
    
    Attributes:
        unique_id: Unique identifier for the agent
        model: Reference to the model
        pos: Current position (x, y)
        altruism: Level of altruism (0.0 = selfish, 1.0 = fully altruistic)
        health: Current health level (decreases with smoke exposure)
        initial_health: Starting health (used for speed calculation)
        base_speed: Base movement speed when at full health
        speed: Current speed (function of health: speed = base_speed * health/initial_health)
        escaped: Whether the agent has escaped
        partner: Reference to paired agent (mutual awareness)
        in_pair: Whether currently in a helping pair
        is_helper: Whether this agent is the helper (True) or helped (False) in the pair
        panic_level: Panic level (0.0 to 1.0), affects movement randomness
    """
    
    def __init__(self, unique_id, model, altruism=0.5, health=None, base_speed=None):
        super().__init__(unique_id, model)
        self.altruism = altruism
        
        # Health is variable per agent (assigned from a range)
        if health is None:
            self.health = np.random.uniform(
                getattr(model, 'health_min', 50), 
                getattr(model, 'health_max', 100)
            )
        else:
            self.health = health
        
        self.initial_health = self.health  # Store initial health for speed calculation
        
        # Base speed: varies between agents (faster/slower individuals)
        if base_speed is None:
            self.base_speed = np.random.uniform(
                getattr(model, 'speed_min', 0.8),
                getattr(model, 'speed_max', 1.2)
            )
        else:
            self.base_speed = base_speed
        
        # Current speed starts at base_speed
        self.speed = self.base_speed
        
        self.escaped = False
        self.panic_level = 0.0
        
        # Pair/helping state (mutual awareness)
        self.partner = None      # Reference to paired agent
        self.in_pair = False     # Whether currently paired
        self.is_helper = False   # True if this agent initiated help
        
    def step(self):
        """Execute one step of agent behavior."""
        # Dead or escaped agents do nothing
        if self.escaped or self.health <= 0:
            return
        
        # Take smoke damage based on smoke level at current position
        smoke_level = self.model.smoke_grid[self.pos[0]][self.pos[1]]
        damage_per_smoke_unit = getattr(self.model, 'damage_per_smoke_unit', 0.05)
        self.health -= smoke_level * damage_per_smoke_unit
        
        # Clamp health to zero (no negative health)
        self.health = max(0.0, self.health)

        # Check for death right after damage is applied
        if self.health <= 0:
            self._handle_death()
            return  # Stop executing the rest of the step
        
        # Update speed based on health (speed decreases as health decreases)
        # When health → 0, speed → 0
        if self.initial_health > 0:
            # Clip speed to be a valid probability [0, 1]
            self.speed = np.clip(
                self.base_speed * (self.health / self.initial_health),
                0.0,
                1.0
            )
        else:
            self.speed = 0.0
        
        # Increase panic based on smoke (slower growth than before)
        panic_increase_per_smoke = getattr(self.model, 'panic_increase_per_smoke', 0.002)
        self.panic_level = min(1.0, self.panic_level + smoke_level * panic_increase_per_smoke)
        
        # Check if partner died or escaped; if so, break the pair
        if self.in_pair and self.partner is not None:
            if self.partner.health <= 0 or self.partner.escaped:
                self._break_pair()
        
        # Altruistic behavior: Check if should help nearby agents
        # Only if not already in a pair and meets helping criteria
        altruism_threshold = getattr(self.model, 'altruism_threshold', 0.3)
        if self.altruism >= altruism_threshold and not self.in_pair:
            self.check_for_help()
        
        # Determine the probability of moving for this turn
        move_prob = self.speed
        if self.in_pair and self.is_helper:
            # Calculate pair speed based on helping_speed_percentage
            # 100 = 100% helper speed, 0 = 100% helped speed
            percentage = getattr(self.model, 'helping_speed_percentage', 80) / 100.0
            
            if self.partner:
                move_prob = (self.speed * percentage) + (self.partner.speed * (1.0 - percentage))
            else:
                move_prob = self.speed * 0.8

        # Move toward exit (only if not in pair OR if helper in pair)
        # Helped agents in pair don't move themselves; helper moves both
        # Movement is probabilistic based on the calculated move_prob
        if (not self.in_pair or self.is_helper) and self.random.random() < move_prob:
            self.move()
        
        # Check if reached exit
        if self.pos in self.model.exits:
            self._handle_escape()
            
    def check_for_help(self):
        """
        Check if there are nearby agents that need help based on speed ratio.
        
        Helping conditions:
        - Helper must be altruistic (binary check)
        - Helper must be healthy enough (health > help_health_threshold)
        - Neighbor must be slower (neighbor.speed < 0.75 * self.speed)
        - Neighbor must not already be in a pair
        """
        # Binary altruism check: either you are (1.0) or you aren't (0.0)
        if self.altruism <= 0:
            return  # Not an altruistic person
        
        # Check if helper is healthy enough to help
        help_health_threshold = getattr(self.model, 'help_health_threshold', 30.0)
        if self.health <= help_health_threshold:
            return  # Too weak to help
        
        # Get neighbors within helping radius
        helping_radius = getattr(self.model, 'helping_radius', 2)
        neighbors = self.model.grid.get_neighbors(
            self.pos,
            moore=True,
            include_center=False,
            radius=helping_radius
        )
        
        # Find agents that need help based on speed ratio
        # New rule: Help if neighbor is slower by factor of 0.75 (3/4)
        help_speed_ratio = 0.75
        
        for neighbor in neighbors:
            if isinstance(neighbor, Person) and not neighbor.escaped and neighbor.health > 0:
                # Check if neighbor is not already in a pair
                if not neighbor.in_pair:
                    # Check if neighbor is significantly slower (needs help)
                    if neighbor.speed < help_speed_ratio * self.speed:
                        # Form a pair with mutual awareness
                        self._form_pair(neighbor)
                        break
                    
    def move(self):
        """
        Move toward the nearest exit, avoiding obstacles.
        If in a pair and is_helper, also move the partner.
        """
        # Find nearest exit
        nearest_exit = self.find_nearest_exit()
        if nearest_exit is None:
            return
        
        # Get possible moves
        possible_moves = self.model.grid.get_neighborhood(
            self.pos,
            moore=True,
            include_center=False
        )
        
        # Filter out obstacles and overly crowded cells
        valid_moves = []
        max_people_per_cell = getattr(self.model, 'max_people_per_cell', 3)
        
        for move in possible_moves:
            # Check if within bounds
            if not self.model.grid.out_of_bounds(move):
                # Check if not an obstacle
                if move not in self.model.obstacles:
                    # Check if not too crowded (count only Person agents)
                    cell_contents = self.model.grid.get_cell_list_contents([move])
                    num_people = sum(1 for obj in cell_contents if isinstance(obj, Person))
                    if num_people < max_people_per_cell:
                        valid_moves.append(move)
        
        if not valid_moves:
            return  # Stuck
        
        # Choose move that gets closer to exit
        best_move = min(
            valid_moves,
            key=lambda pos: self.distance(pos, nearest_exit)
        )
        
        # Apply panic: sometimes make suboptimal (random) moves
        # "Calming" Effect: Agents in a pair DO NOT panic. They stay focused.
        if not self.in_pair:
            panic_randomness_factor = getattr(self.model, 'panic_randomness_factor', 0.5)
            if np.random.random() < self.panic_level * panic_randomness_factor:
                best_move = self.random.choice(valid_moves)
        
        # Move self
        self.model.grid.move_agent(self, best_move)
        
        # If in a pair and is helper, move partner too
        if self.in_pair and self.is_helper and self.partner is not None:
            # Only move partner if they are still alive and not escaped
            if self.partner.health > 0 and not self.partner.escaped:
                self._move_partner(best_move)
                    
    def find_nearest_exit(self):
        """Find the nearest exit position."""
        if not self.model.exits:
            return None
        return min(self.model.exits, key=lambda exit: self.distance(self.pos, exit))
    
    def distance(self, pos1, pos2):
        """Calculate Manhattan distance between two positions."""
        return abs(pos1[0] - pos2[0]) + abs(pos1[1] - pos2[1])
    
    def _form_pair(self, neighbor):
        """
        Form a helping pair with mutual awareness.
        Self becomes the helper, neighbor becomes the helped.
        """
        self.partner = neighbor
        self.in_pair = True
        self.is_helper = True
        
        neighbor.partner = self
        neighbor.in_pair = True
        neighbor.is_helper = False
    
    def _break_pair(self):
        """
        Break the current helping pair (partner died or escaped).
        Reset pair state for both agents.
        """
        if self.partner is not None:
            # Reset partner's state
            self.partner.partner = None
            self.partner.in_pair = False
            self.partner.is_helper = False
        
        # Reset self state
        self.partner = None
        self.in_pair = False
        self.is_helper = False
    
    def _move_partner(self, helper_pos):
        """
        Move partner to same cell or adjacent cell near helper.
        Called by helper after helper moves.
        """
        # Try to move partner to the same cell as helper
        cell_contents = self.model.grid.get_cell_list_contents([helper_pos])
        num_people = sum(1 for obj in cell_contents if isinstance(obj, Person))
        max_people_per_cell = getattr(self.model, 'max_people_per_cell', 3)
        
        if num_people < max_people_per_cell:
            # Room in helper's cell
            self.model.grid.move_agent(self.partner, helper_pos)
        else:
            # Try adjacent cells
            neighbors = self.model.grid.get_neighborhood(
                helper_pos,
                moore=True,
                include_center=False
            )
            
            valid_moves = []
            for move in neighbors:
                if not self.model.grid.out_of_bounds(move):
                    if move not in self.model.obstacles:
                        cell_contents = self.model.grid.get_cell_list_contents([move])
                        num_people = sum(1 for obj in cell_contents if isinstance(obj, Person))
                        if num_people < max_people_per_cell:
                            valid_moves.append(move)
            
            if valid_moves:
                # Move to closest valid cell to helper
                best_move = min(valid_moves, key=lambda p: self.distance(p, helper_pos))
                self.model.grid.move_agent(self.partner, best_move)
            # If no valid moves, partner stays where they are
    
    def _handle_death(self):
        """
        Handle agent death.
        Mark as dead, remove from grid, leave a marker.
        If in a pair, break it.
        """
        # Get position before removing
        dead_pos = self.pos
        
        # If in a pair, break it
        if self.in_pair:
            self._break_pair()

        # Remove self from grid so it no longer blocks
        self.model.grid.remove_agent(self)

        # Place a DeadDot for visualization
        dot = DeadDot(f"dead_{self.unique_id}", self.model)
        self.model.grid.place_agent(dot, dead_pos)

    def _handle_escape(self):
        """
        Handle agent reaching an exit.
        Mark as escaped, remove from grid, update counters, place escape marker.
        If in a pair, break the pair.
        """
        # Store position before removing agent
        exit_x, exit_y = self.pos
        self.escaped = True
        self.model.grid.remove_agent(self)
        self.model.escaped_count += 1
        
        # Increment per-exit counter
        try:
            self.model.exit_counts[(exit_x, exit_y)] += 1
        except Exception:
            pass
        
        # If in a pair, break it (partner continues alone)
        if self.in_pair:
            self._break_pair()
        
        # Drop a green dot just outside the exit for visualization
        candidates = [
            (exit_x, exit_y - 1),
            (exit_x, exit_y + 1),
            (exit_x - 1, exit_y),
            (exit_x + 1, exit_y)
        ]
        dot = EscapedDot(f"escaped_{self.unique_id}", self.model)
        placed = False
        for cx, cy in candidates:
            if 0 <= cx < self.model.width and 0 <= cy < self.model.height:
                # Use an on-grid neighbor cell next to exit
                if (cx, cy) not in self.model.exits and (cx, cy) not in self.model.obstacles:
                    self.model.grid.place_agent(dot, (cx, cy))
                    placed = True
                    break
        if not placed:
            # Fallback: place on the exit cell
            self.model.grid.place_agent(dot, (exit_x, exit_y))


class Obstacle(Agent):
    """Static agent representing an obstacle (furniture, debris). Blocks movement."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_obstacle = True
    
    def step(self):
        return


class Door(Agent):
    """Static agent representing an exit door. Does not act or block."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_exit = True

    def step(self):
        return


class EscapedDot(Agent):
    """Visual marker for a person who escaped through an exit."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_marker = True

    def step(self):
        return


class Smoke(Agent):
    """Visual representation of smoke in a cell."""
    def __init__(self, unique_id, model, smoke_level):
        super().__init__(unique_id, model)
        self.smoke_level = smoke_level
        self.is_smoke = True
    
    def step(self):
        return


class DeadDot(Agent):
    """Visual marker for a person who has died."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_marker = True

    def step(self):
        return


class RedObstacle(Agent):
    """Static agent representing a red obstacle. Blocks movement just like a regular obstacle."""
    def __init__(self, unique_id, model):
        super().__init__(unique_id, model)
        self.is_obstacle = True
    
    def step(self):
        return

