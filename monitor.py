import os
import sys
import time
import json
import argparse
import requests
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    """Monitor for decentralized training progress."""
    
    def __init__(self, coordinator_url: str, update_interval: int = 5):
        self.coordinator_url = coordinator_url
        self.update_interval = update_interval
        self.rounds = []
        self.active_nodes = []
        self.gpu_nodes = []
        self.cpu_nodes = []
        self.last_update = None
        
        # Initialize plot
        plt.style.use('ggplot')
        self.fig, (self.ax1, self.ax2) = plt.subplots(2, 1, figsize=(10, 8))
        self.fig.suptitle('Decentralized Training Monitor', fontsize=16)
        
        # Configure axes
        self.ax1.set_ylabel('Training Round')
        self.ax1.set_xlabel('Time')
        
        self.ax2.set_ylabel('Number of Nodes')
        self.ax2.set_xlabel('Time')
        
        # Create lines
        self.line_rounds, = self.ax1.plot([], [], label='Training Round', marker='o', linestyle='-', color='blue')
        self.line_active, = self.ax2.plot([], [], label='Active Nodes', marker='o', linestyle='-', color='green')
        self.line_gpu, = self.ax2.plot([], [], label='GPU Nodes', marker='s', linestyle='--', color='red')
        self.line_cpu, = self.ax2.plot([], [], label='CPU Nodes', marker='x', linestyle='--', color='orange')
        
        # Add legends
        self.ax1.legend()
        self.ax2.legend()
        
        # Add grid
        self.ax1.grid(True)
        self.ax2.grid(True)
        
        # Create timestamp list for x-axis
        self.timestamps = []
        
    def fetch_coordinator_status(self):
        """Fetch status from coordinator."""
        try:
            print(f"Attempting to connect to {self.coordinator_url}/api/status...")
            response = requests.get(f"{self.coordinator_url}/api/status", timeout=30)  # Increased timeout from 5 to 30 seconds
            if response.status_code == 200:
                data = response.json()
                self.last_update = datetime.now()
                print(f"Connected successfully, received data: {data}")
                return data
            else:
                logger.error(f"Failed to fetch status: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error fetching status: {e}")
            return None
    
    def update_plot(self, frame):
        """Update the plot with new data."""
        status = self.fetch_coordinator_status()
        
        if status is not None:
            # Get current time
            now = datetime.now()
            time_str = now.strftime("%H:%M:%S")
            
            # Add new data points
            self.timestamps.append(time_str)
            self.rounds.append(status.get("training_round", 0))
            self.active_nodes.append(status.get("active_nodes", 0))
            self.gpu_nodes.append(status.get("gpu_nodes", 0))
            self.cpu_nodes.append(status.get("cpu_nodes", 0))
            
            # Keep only the last 30 data points for better visibility
            if len(self.timestamps) > 30:
                self.timestamps = self.timestamps[-30:]
                self.rounds = self.rounds[-30:]
                self.active_nodes = self.active_nodes[-30:]
                self.gpu_nodes = self.gpu_nodes[-30:]
                self.cpu_nodes = self.cpu_nodes[-30:]
            
            # Update the lines
            self.line_rounds.set_data(range(len(self.timestamps)), self.rounds)
            self.line_active.set_data(range(len(self.timestamps)), self.active_nodes)
            self.line_gpu.set_data(range(len(self.timestamps)), self.gpu_nodes)
            self.line_cpu.set_data(range(len(self.timestamps)), self.cpu_nodes)
            
            # Update the x-axis ticks and labels
            self.ax1.set_xlim(0, len(self.timestamps) - 1)
            self.ax2.set_xlim(0, len(self.timestamps) - 1)
            
            if len(self.timestamps) > 0:
                max_y1 = max(self.rounds) if self.rounds else 1
                max_y2 = max(max(self.active_nodes), max(self.gpu_nodes), max(self.cpu_nodes)) if self.active_nodes else 1
                
                self.ax1.set_ylim(0, max_y1 * 1.1 + 1)
                self.ax2.set_ylim(0, max_y2 * 1.1 + 1)
                
                # Set x ticks at certain intervals
                tick_interval = max(1, len(self.timestamps) // 10)
                tick_positions = range(0, len(self.timestamps), tick_interval)
                tick_labels = [self.timestamps[i] for i in tick_positions if i < len(self.timestamps)]
                
                self.ax1.set_xticks(tick_positions)
                self.ax1.set_xticklabels(tick_labels, rotation=45)
                self.ax2.set_xticks(tick_positions)
                self.ax2.set_xticklabels(tick_labels, rotation=45)
            
            # Update title with current status
            training_status = "In Progress" if status.get("training_in_progress", False) else "Idle"
            self.fig.suptitle(f'Decentralized Training Monitor - {training_status}\nLast Update: {time_str}')
            
            # Print status to console
            print(f"[{time_str}] Round: {status.get('training_round', 0)}, " 
                  f"Active Nodes: {status.get('active_nodes', 0)} "
                  f"(GPU: {status.get('gpu_nodes', 0)}, CPU: {status.get('cpu_nodes', 0)})")
        
        # Return all artists that need to be redrawn
        return [self.line_rounds, self.line_active, self.line_gpu, self.line_cpu]
    
    def run(self):
        """Run the monitoring animation."""
        try:
            # Try to get initial status
            print(f"Attempting initial connection to coordinator at {self.coordinator_url}...")
            initial_status = self.fetch_coordinator_status()
            if initial_status is None:
                logger.error(f"Cannot connect to coordinator at {self.coordinator_url}")
                print(f"Cannot connect to coordinator at {self.coordinator_url}")
                print("Please check the coordinator URL and make sure the coordinator is running.")
                print("NOTE: Make sure you don't include 'http://' in the coordinator address parameter")
                return
                
            print(f"Connected to coordinator at {self.coordinator_url}")
            print(f"Training round: {initial_status.get('training_round', 0)}")
            print(f"Active nodes: {initial_status.get('active_nodes', 0)}")
            print("Starting monitoring...")
            
            # Set up the animation
            ani = FuncAnimation(
                self.fig, 
                self.update_plot, 
                interval=self.update_interval * 1000,  # Convert to milliseconds
                blit=True
            )
            
            # Show the plot
            plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust for the title
            plt.show()
            
        except KeyboardInterrupt:
            print("\nMonitoring stopped by user")
        except Exception as e:
            logger.error(f"Error in monitoring: {e}")
            print(f"Error: {e}")

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Decentralized ML Training Monitor")
    parser.add_argument("--coordinator-address", type=str, required=True,
                        help="Coordinator address in the format <host>:<port>")
    parser.add_argument("--interval", type=int, default=5,
                        help="Update interval in seconds (default: 5)")
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    # Construct coordinator URL
    coordinator_url = f"http://{args.coordinator_address}"
    
    # Remove any double http:// if present
    if coordinator_url.startswith("http://http://"):
        coordinator_url = coordinator_url.replace("http://http://", "http://")
    
    print(f"Using coordinator URL: {coordinator_url}")
    
    # Create and run monitor
    monitor = TrainingMonitor(
        coordinator_url=coordinator_url,
        update_interval=args.interval
    )
    
    monitor.run() 