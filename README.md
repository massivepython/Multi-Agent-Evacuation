# Emergency Evacuation Multi-Agent Simulation

A multi-agent system (MAS) that simulates emergency evacuation from a smoke-filled room. This model explores how **altruistic behavior** affects survival rates during emergencies.

## What is This?

This simulation models people trying to escape from a room filling with smoke. Each person (agent) has different characteristics:

- **Altruistic agents** (yellow triangles) - Will stop to help slower agents escape
- **Non-altruistic agents** (blue triangles) - Focus only on their own escape
- **Helping pairs** (green circles) - When an altruistic agent helps someone, they move together as one unit

The key research question: **Does altruism help or hurt overall survival rates during evacuations?**

## Visual Guide

When you run the simulation, you'll see:

| Symbol | Meaning |
|--------|---------|
| 🟡 Yellow triangle | Altruistic agent (willing to help others) |
| 🔵 Blue triangle | Non-altruistic agent (self-focused) |
| 🟢 Green circle | Two agents in a helping pair (moving together) |
| ⬛ Dark gray | Obstacles/furniture |
| 🟥 Red | Blocked exits or hazards |
| 🟩 Light green | Exit doors |
| ⚫ Black dot | Dead agent |
| Grey overlay | Smoke |

**Cell occupancy:**
- One agent = single triangle (fills half the cell diagonally)
- Two agents = two triangles (fills the whole cell)

## Setup Instructions

### Requirements
- **Python 3.10 or 3.11** (required - other versions may not work)
- pip or conda for package management

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/Multi-Agent-Evacuation.git
   cd Multi-Agent-Evacuation
   ```

2. **Create a virtual environment (recommended):**
   
   Using conda:
   ```bash
   conda create -n MAS python=3.10
   conda activate MAS
   ```
   
   Or using venv:
   ```bash
   python -m venv venv
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## How to Run

### Main Menu
```bash
python mas.py
```

This gives you three options:

1. **Interactive Visualization** - Watch the simulation in real-time in your browser
2. **Batch Experiments** - Run multiple simulations to collect statistical data
3. **Single Simulation** - Run one simulation with custom parameters

### Interactive Visualization

Choose option 1, then open your browser to: **http://localhost:8521**

**Controls:**
- **Start/Step** - Run the simulation
- **Reset** - Restart with new random positions
- **Adjust sliders** - Change parameters in real-time:
  - *Altruistic Population %* - How many agents are altruistic
  - *Smoke Level* - How much smoke is in the room
  - *Helping Speed Impact* - How much the helper's speed affects the pair

### Advanced Analysis

**3D Parameter Analysis:**
```bash
python 3d_analysis.py
```
Runs a grid search over altruism % and helping speed % to find optimal configurations. Generates 3D surface plots and heatmaps saved to `vis_result/`.

**Statistical Analysis:**
```bash
python stats_analysis.py
```
Performs correlation analysis, ANOVA, and t-tests on results from 3d_analysis.py.

## Project Structure

```
├── mas.py              # Main entry point
├── model.py            # Core simulation model (room, smoke, exits)
├── agent.py            # Agent behaviors (Person, Obstacle, Door, etc.)
├── visualization.py    # Mesa web visualization
├── map.txt             # Room layout (walls, exits, spawn points)
├── batch_run.py        # Batch experiment runner
├── 3d_analysis.py      # Parameter grid search & visualization
├── stats_analysis.py   # Statistical analysis of results
├── requirements.txt    # Python dependencies
├── images/             # Triangle/circle sprites for visualization
└── vis_result/         # Generated analysis plots
```

## Map Format (map.txt)

The room layout is defined in `map.txt`:
- `#` = Wall/obstacle
- `E` = Exit door
- `S` = Spawn point (where agents start) — **Each 'S' creates exactly 1 agent**
- `X` = Red obstacle (blocked area)
- `.` = Empty walkable space


## Key Slider Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `altruistic_population_percent` | % of agents that are altruistic (binary: yes/no) | 50% |
| `smoke_level` | Smoke intensity (0-100), affects damage and panic growth | 50 |
| `helping_speed_percentage` | Pair speed: 0% = helped's speed, 100% = helper's speed | 80 |
| `max_people_per_cell` | Maximum agents allowed per grid cell | 2 |

**Important:** The number of agents is determined by counting 'S' characters in the map, NOT by any slider. The current map has **218 spawn points** = 218 agents.

## Agent Initialization

When the simulation starts, each agent is initialized with:

| Property | How it's set | Range |
|----------|--------------|---------------|
| **Position** | Placed on an 'S' spawn point from map.txt | Always 218 agents to model the Cube241 classroom full |
| **Altruism** |  Random based on slider % | Binary: 1.0 (altruistic) or 0.0 (not) |
| **Health** | Everyone starts with 100 health | 0.0-100.0 |
| **Speed** | Starts random between 0.7 and 1 | 0.0-1.0 |
| **Panic** | Starts at 0, grows with smoke exposure | 0.0-1.0 |
| **In Pair** | Starts as False | Binary: True or False |


## How Movement Works

Each step, an agent:
1. **Takes smoke damage:** `health -= smoke_level × 0.05`
2. **Speed updates:** `speed = base_speed × (health / initial_health)`, capped between 0 and 1
3. **Panic increases:** `panic += smoke_level × 0.002` (capped at 1.0)
4. **Speed = Movement probability:** Agent moves **1 cell** with probability = speed. Example: speed 0.7 = 70% chance to move, 30% chance to stay still

**Direction choice:**
- Agents can move in **8 directions** (including diagonals)
- They pick the valid move that gets closest to the nearest exit
- Valid moves = not blocked, not too crowded (max 2 agents per cell)

**Panic effect:**
- Chance of random move = `panic × 0.4` (max 40% at full panic)
- If triggered, agent picks randomly from ALL valid moves
- **Agents in helping pairs don't panic**

## How Altruism Works

1. **Altruistic agents** scan for nearby agents within a 2-cell radius
2. They help if: the neighbor's **speed < 75% of their own speed**
3. The helper must also have **health > 30** (can't help if too weak themselves)
4. The pair moves together at a combined speed (weighted by the "Helping Speed Impact" slider)
5. Both agents in a pair **don't panic** (calming effect)
6. If one agent dies, the pair breaks apart



## Results & Findings

We ran the analysis scripts for our findings.
Check `vis_result/` for generated plots after running analysis.

## Dependencies

- `mesa==2.1.4` - Agent-based modeling framework
- `numpy==1.24.3` - Numerical computing
- `pandas==2.0.3` - Data analysis
- `matplotlib==3.7.2` - Plotting and visualization
- `Pillow==10.0.0` - Generating images for agent visualization


