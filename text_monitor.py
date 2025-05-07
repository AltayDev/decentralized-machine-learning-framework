import sys
import time
import json
import argparse
import requests
from datetime import datetime
import os
import logging

import config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(os.path.join(config.LOGS_DIR, 'monitor.log'))
    ]
)
logger = logging.getLogger(__name__)

def clear_screen():
    """Clear the terminal screen."""
    os.system('cls' if os.name == 'nt' else 'clear')

def fetch_coordinator_status(coordinator_url):
    """Fetch status from coordinator."""
    try:
        print(f"Connecting to {coordinator_url}/api/status...")
        response = requests.get(f"{coordinator_url}/api/status", timeout=10)
        if response.status_code == 200:
            data = response.json()
            # Print raw JSON for debugging
            debug_file = os.path.join(config.LOGS_DIR, 'status_debug.json')
            with open(debug_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"Debug info saved to {debug_file}")
            print("-" * 60)
            return data
        else:
            print(f"Failed to fetch status: {response.status_code} - {response.text}")
            return None
    except Exception as e:
        print(f"Error fetching status: {e}")
        return None

def print_status(status, previous_status=None, update_interval=5):
    """Print coordinator status in text format."""
    if status is None:
        print("No status information available.")
        return
    
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Training round information
    current_round = status.get("training_round", 0)
    training_in_progress = status.get("training_in_progress", False)
    
    # Node information
    active_nodes = status.get("active_nodes", 0)
    gpu_nodes = status.get("gpu_nodes", 0)
    cpu_nodes = status.get("cpu_nodes", 0)
    updates_received = status.get("updates_received", 0)
    
    # Calculate changes if previous status is available
    round_change = ""
    nodes_change = ""
    if previous_status:
        prev_round = previous_status.get("training_round", 0)
        prev_nodes = previous_status.get("active_nodes", 0)
        
        if current_round > prev_round:
            round_change = f" (+{current_round - prev_round})"
        
        if active_nodes != prev_nodes:
            if active_nodes > prev_nodes:
                nodes_change = f" (+{active_nodes - prev_nodes})"
            else:
                nodes_change = f" (-{prev_nodes - active_nodes})"
    
    # Format and print status
    clear_screen()
    print("=" * 60)
    print(f"DECENTRALIZED ML COORDINATOR STATUS - {now}")
    print("=" * 60)
    print(f"Training round: {current_round}{round_change}")
    print(f"Training in progress: {'Yes' if training_in_progress else 'No'}")
    print(f"Active nodes: {active_nodes}{nodes_change}")
    print(f"  - GPU nodes: {gpu_nodes}")
    print(f"  - CPU nodes: {cpu_nodes}")
    print(f"Updates received: {updates_received}")
    print("-" * 60)
    
    # Print node details if available
    if "nodes" in status:
        print("\nNODE DETAILS:")
        print("-" * 60)
        for node_id, node_info in status["nodes"].items():
            node_type = "GPU" if node_info.get("system_info", {}).get("gpu_available", False) else "CPU"
            last_heartbeat = datetime.fromtimestamp(node_info.get("last_heartbeat", 0)).strftime("%H:%M:%S")
            print(f"Node ID: {node_id[:8]}... ({node_type})")
            print(f"  Last heartbeat: {last_heartbeat}")
            print(f"  Status: {node_info.get('status', 'unknown')}")
            print(f"  IP: {node_info.get('ip', 'unknown')}")
            
            if "last_update" in node_info:
                last_update = datetime.fromtimestamp(node_info.get("last_update", 0)).strftime("%H:%M:%S")
                print(f"  Last model update: {last_update} (round {node_info.get('update_round', -1)})")
            
            print()
    
    print("=" * 60)
    print(f"Press Ctrl+C to quit. Next update in {update_interval} seconds...")
    
    # Save status history to log file
    log_file = os.path.join(config.LOGS_DIR, 'status_history.log')
    with open(log_file, 'a') as f:
        f.write(f"{now} - Round: {current_round}, Active: {active_nodes}, GPU: {gpu_nodes}, CPU: {cpu_nodes}, Updates: {updates_received}\n")

def monitor_coordinator(coordinator_url, update_interval=5):
    """Monitor coordinator status in text mode."""
    print(f"Starting text-based monitoring for {coordinator_url}")
    print(f"Logs will be saved to {config.LOGS_DIR}")
    print("Press Ctrl+C to exit.")
    
    previous_status = None
    
    try:
        while True:
            status = fetch_coordinator_status(coordinator_url)
            if status:
                print_status(status, previous_status, update_interval)
                previous_status = status
            else:
                print(f"Failed to connect to coordinator at {coordinator_url}")
                print(f"Will retry in {update_interval} seconds...")
            
            time.sleep(update_interval)
            
    except KeyboardInterrupt:
        print("\nMonitoring stopped by user.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Text-based Decentralized ML Coordinator Monitor")
    parser.add_argument("--coordinator-address", type=str, required=True,
                        help="Coordinator address in the format <host>:<port>")
    parser.add_argument("--interval", type=int, default=5,
                        help="Update interval in seconds (default: 5)")
    args = parser.parse_args()
    
    # Construct coordinator URL
    coordinator_url = f"http://{args.coordinator_address}"
    if coordinator_url.startswith("http://http://"):
        coordinator_url = coordinator_url.replace("http://http://", "http://")
    
    monitor_coordinator(coordinator_url, args.interval) 