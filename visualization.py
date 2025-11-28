"""
Visualization module for the evacuation simulation.
Creates an interactive web-based visualization using Mesa's visualization tools.
"""

from mesa.visualization.modules import CanvasGrid, ChartModule, TextElement
from mesa.visualization.ModularVisualization import ModularServer
from mesa.visualization.UserParam import Slider, Checkbox, Choice # Import Choice
from model import EvacuationModel
from agent import Person, Door, EscapedDot, Obstacle, Smoke, DeadDot, RedObstacle


def get_person_portrayal(agent, model):
    """
    Define how Person agents are portrayed in the visualization.
    
    Shape logic:
    - Agents in helping pair: ONE merged green circle (centered)
    - Single agent: top-left triangle ◤
    - Two agents in cell: first = top-left ◤, second = bottom-right ◢
    - Yellow = altruistic, Blue = non-altruistic
    """
    if agent is None:
        return None
    
    # Get all Person agents in the same cell (alive, not escaped)
    cell_contents = model.grid.get_cell_list_contents([agent.pos])
    persons_in_cell = [a for a in cell_contents 
                       if isinstance(a, Person) and a.health > 0 and not a.escaped]
    
    # Check if there's a helping pair in this cell
    paired_agents = [a for a in persons_in_cell if a.in_pair]
    
    if len(paired_agents) > 0:
        # If this cell has agents in a pair, show ONE green circle
        # Only render for the first paired agent to avoid duplicates
        if agent == paired_agents[0]:
            return {
                "Shape": "images/green_circle.png",
                "scale": 1.0,
                "Layer": 3
            }
        else:
            # Don't render other paired agents in same cell
            return None
    
    # Filter to only non-paired agents
    non_paired = [a for a in persons_in_cell if not a.in_pair]
    
    if agent not in non_paired:
        return None
    
    # Find this agent's index among non-paired agents
    agent_index = non_paired.index(agent)
    
    # Determine if altruistic (yellow) or not (blue)
    is_altruistic = getattr(agent, 'altruism', 0) > 0
    
    if agent_index == 0:
        # First agent: top-left triangle ◤
        if is_altruistic:
            shape = "images/yellow_tl.png"
        else:
            shape = "images/blue_tl.png"
    else:
        # Second agent: bottom-right triangle ◢
        if is_altruistic:
            shape = "images/yellow_br.png"
        else:
            shape = "images/blue_br.png"
    
    return {
        "Shape": shape,
        "scale": 1.0,
        "Layer": 3
    }


def grid_portrayal(agent):
    """Portray individual agents and environment elements."""
    if agent is None:
        return None

    # Dead agents as black dots (Layer 1)
    if isinstance(agent, DeadDot):
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 1,
            "r": 0.4,
            "Color": "black"
        }
    
    # Smoke as semi-transparent gray background (Layer 0 - lowest)
    if isinstance(agent, Smoke):
        smoke_opacity = min(1.0, agent.smoke_level / 100.0)
        # Make it reddish-gray to show danger
        # Base gray decreases with opacity (darker)
        smoke_gray = int(200 - (smoke_opacity * 100))
        # Red component increases with opacity
        smoke_red = int(smoke_gray + (smoke_opacity * 55))
        
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 0,
            "w": 1,
            "h": 1,
            "Color": f"rgba({smoke_red}, {smoke_gray}, {smoke_gray}, {smoke_opacity * 0.6})"
        }
    
    # If it's a Person agent (Layer 3 - highest)
    if isinstance(agent, Person):
        return get_person_portrayal(agent, agent.model)
    
    # Obstacles as dark gray squares (Layer 1)
    if isinstance(agent, Obstacle):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
            "Color": "#555555"  # Darker gray to contrast with smoke
        }
    
    # Red Obstacles (Layer 1)
    if isinstance(agent, RedObstacle):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 1,
            "w": 1,
            "h": 1,
            "Color": "#e74c3c"  # Bright red
        }
    
    # Doors as light green squares (Layer 2)
    if isinstance(agent, Door):
        return {
            "Shape": "rect",
            "Filled": "true",
            "Layer": 2,
            "w": 1,
            "h": 1,
            "Color": "lightgreen"
        }
    
    # Escape markers (Layer 1)
    if isinstance(agent, EscapedDot):
        return {
            "Shape": "circle",
            "Filled": "true",
            "Layer": 1,
            "r": 0.2,
            "Color": "#2ecc71"
        }
    
    return None


# Create charts
class ChartTitle(TextElement):
    """Simple text title/description to place above charts."""
    def __init__(self, title, subtitle=""):
        self.title = title
        self.subtitle = subtitle

    def render(self, model):
        sub = f"<div style='color:#666;font-size:12px;margin-top:-6px'>{self.subtitle}</div>" if self.subtitle else ""
        return f"<div style='margin:6px 0 2px 0'><strong>{self.title}</strong></div>" + sub


def get_map_dimensions(filename):
    """Read the map file to determine grid dimensions for the visualization."""
    try:
        with open(filename, 'r') as f:
            lines = [line.rstrip('\n') for line in f if line.strip()]
        if not lines:
            return 50, 30
        height = len(lines)
        width = max(len(line) for line in lines)
        return width, height
    except FileNotFoundError:
        return 50, 30

# Get dimensions from map file
map_width, map_height = get_map_dimensions("map.txt")

escape_chart_title = ChartTitle(
    "Evacuation Status",
    "Number of people who escaped (green), died (red), or remain inside (blue)"
)

escape_chart = ChartModule(
    [
        {"Label": "Escaped", "Color": "#2ecc71"},    # green
        {"Label": "Dead", "Color": "#e74c3c"},       # red
        {"Label": "Remaining", "Color": "#3498db"}  # blue
    ],
    data_collector_name="datacollector"
)

health_chart_title = ChartTitle(
    "Average Health & Speed",
    "Mean health and speed of alive agents inside the room"
)

health_chart = ChartModule(
    [
        {"Label": "Average Health", "Color": "#8e44ad"},
        {"Label": "Average Speed", "Color": "#f39c12"}
    ],
    data_collector_name="datacollector"
)

panic_chart_title = ChartTitle(
    "Average Panic Level",
    "Mean panic level of alive agents (0–1)"
)

panic_chart = ChartModule(
    [
        {"Label": "Average Panic", "Color": "#e74c3c"}
    ],
    data_collector_name="datacollector"
)


class EvacCounts(TextElement):
    """Display a clean text summary of evacuation counts and timer."""
    def render(self, model):
        try:
            escaped = model.escaped_count
            dead = model.dead_count
            # Count actual Person agents still in the room
            remaining = len([a for a in model.schedule.agents if hasattr(a, 'health') and not a.escaped and a.health > 0])
            smoke_level = model.smoke_level
            # Timer in seconds (each step = 1 second)
            time_seconds = model.steps
            
            # Check for batch result message
            batch_msg = getattr(model, 'batch_result_message', "")
            batch_html = ""
            if batch_msg:
                batch_html = f"<div style='color:#c0392b;font-size:14px;font-weight:bold;margin-bottom:4px'>{batch_msg}</div>"
                
        except Exception:
            escaped = dead = remaining = 0
            smoke_level = 0
            time_seconds = 0
            batch_html = ""
            
        return (
            f"{batch_html}"
            "<div style='margin:6px 0'>"
            f"<strong>Time:</strong> <span style='color:#3498db;font-size:16px;font-weight:bold'>{time_seconds}s</span> | "
            f"<strong>Escaped:</strong> <span style='color:#2ecc71'>{escaped}</span> | "
            f"<strong>Dead:</strong> <span style='color:#e74c3c'>{dead}</span> | "
            f"<strong>Remaining:</strong> <span style='color:#3498db'>{remaining}</span> | "
            f"<strong>Smoke:</strong> <span style='color:#95a5a6'>{smoke_level:.0f}</span>"
            "</div>"
        )


# Create canvas grid using dynamic dimensions
canvas_element = CanvasGrid(grid_portrayal, map_width, map_height, 750, 450)

# Model parameters
# Fixed parameters (no sliders): num_people=150, num_obstacles (auto), health_min=50, health_max=100
model_params = {
    "width": map_width,
    "height": map_height,
    "num_people": 150,  # Fixed: 150 people
    "num_obstacles": 10,  # Fixed: auto-calculated based on room size
    "health_min": 100,  # Fixed: minimum health
    "health_max": 100,  # Fixed: maximum health
    "run_mode": Choice("Run Mode", value="Interactive Simulation", choices=["Interactive Simulation", "Batch Analysis"]),
    "batch_size": Slider("Batch Size (if Batch Analysis)", 30, 10, 100, 5),
    "altruistic_population_percent": Slider(
        "Altruistic Population %",
        50.0,
        0.0,
        100.0,
        5.0,
        description="Percentage of agents that are altruistic"
    ),
    "smoke_level": Slider(
        "Smoke Level",
        50.0,
        10.0,
        100.0,
        5.0,
        description="Constant smoke level across entire room"
    ),
    "helping_speed_percentage": Slider(
        "Helping Speed Impact",
        80,
        0,
        100,
        5,
        description="Pair Speed: 0%=Helped Speed, 100%=Helper Speed"
    )
}

# Create server
server = ModularServer(
    EvacuationModel,
    [
        canvas_element,
        EvacCounts(),
        escape_chart_title,
        escape_chart,
        health_chart_title,
        health_chart,
        panic_chart_title,
        panic_chart
    ],
    "Emergency Evacuation Simulation",
    model_params
)

server.port = 8521  # Default Mesa port


if __name__ == "__main__":
    server.launch()
