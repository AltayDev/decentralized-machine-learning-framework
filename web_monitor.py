import os
from flask import Flask, render_template, jsonify
import plotly
import plotly.graph_objs as go
import json
import requests
from datetime import datetime
import threading
import time
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TrainingMonitor:
    def __init__(self, coordinator_url: str, update_interval: int = 5):
        self.coordinator_url = coordinator_url
        self.update_interval = update_interval
        self.data = {
            'timestamps': [],
            'rounds': [],
            'active_nodes': [],
            'gpu_nodes': [],
            'cpu_nodes': [],
            'training_status': []
        }
        self.lock = threading.Lock()
        
        # Start background update thread
        self.is_running = True
        self.update_thread = threading.Thread(target=self._update_loop)
        self.update_thread.daemon = True
        self.update_thread.start()
    
    def _update_loop(self):
        """Background thread to update data from coordinator."""
        while self.is_running:
            try:
                self._fetch_coordinator_status()
            except Exception as e:
                logger.error(f"Error updating status: {e}")
            time.sleep(self.update_interval)
    
    def _fetch_coordinator_status(self):
        """Fetch status from coordinator."""
        try:
            response = requests.get(f"{self.coordinator_url}/api/status", timeout=30)
            if response.status_code == 200:
                data = response.json()
                now = datetime.now()
                
                with self.lock:
                    # Keep only last 100 data points
                    if len(self.data['timestamps']) >= 100:
                        for key in self.data:
                            self.data[key] = self.data[key][-99:]
                    
                    self.data['timestamps'].append(now.strftime("%H:%M:%S"))
                    self.data['rounds'].append(data.get("training_round", 0))
                    self.data['active_nodes'].append(data.get("active_nodes", 0))
                    self.data['gpu_nodes'].append(data.get("gpu_nodes", 0))
                    self.data['cpu_nodes'].append(data.get("cpu_nodes", 0))
                    self.data['training_status'].append(
                        "In Progress" if data.get("training_in_progress", False) else "Idle"
                    )
                
        except Exception as e:
            logger.error(f"Error fetching status: {e}")
    
    def get_plot_data(self):
        """Get data for plotting."""
        with self.lock:
            return self.data.copy()
    
    def stop(self):
        """Stop the monitor."""
        self.is_running = False
        self.update_thread.join()

# Create monitor instance
monitor = None

@app.route('/')
def index():
    """Render the main dashboard page."""
    return render_template('index.html')

@app.route('/api/status')
def get_status():
    """Get current training status."""
    if monitor is None:
        return jsonify({"error": "Monitor not initialized"}), 500
    
    data = monitor.get_plot_data()
    
    # Create plot traces
    traces = [
        go.Scatter(
            x=data['timestamps'],
            y=data['rounds'],
            name='Training Round',
            line=dict(color='blue')
        ),
        go.Scatter(
            x=data['timestamps'],
            y=data['active_nodes'],
            name='Active Nodes',
            line=dict(color='green')
        ),
        go.Scatter(
            x=data['timestamps'],
            y=data['gpu_nodes'],
            name='GPU Nodes',
            line=dict(color='red')
        ),
        go.Scatter(
            x=data['timestamps'],
            y=data['cpu_nodes'],
            name='CPU Nodes',
            line=dict(color='orange')
        )
    ]
    
    # Create layout
    layout = go.Layout(
        title='Training Progress',
        xaxis=dict(title='Time'),
        yaxis=dict(title='Value'),
        showlegend=True
    )
    
    # Create figure
    figure = go.Figure(data=traces, layout=layout)
    
    # Get current status
    current_status = {
        'training_round': data['rounds'][-1] if data['rounds'] else 0,
        'active_nodes': data['active_nodes'][-1] if data['active_nodes'] else 0,
        'gpu_nodes': data['gpu_nodes'][-1] if data['gpu_nodes'] else 0,
        'cpu_nodes': data['cpu_nodes'][-1] if data['cpu_nodes'] else 0,
        'training_status': data['training_status'][-1] if data['training_status'] else 'Unknown'
    }
    
    return jsonify({
        'plot': json.dumps(figure, cls=plotly.utils.PlotlyJSONEncoder),
        'status': current_status
    })

def create_templates():
    """Create necessary template files."""
    os.makedirs('templates', exist_ok=True)
    
    # Create index.html
    with open('templates/index.html', 'w') as f:
        f.write('''
<!DOCTYPE html>
<html>
<head>
    <title>Training Monitor</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: Arial, sans-serif;
            margin: 20px;
            background-color: #f5f5f5;
        }
        .container {
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 20px;
            border-radius: 5px;
            box-shadow: 0 0 10px rgba(0,0,0,0.1);
        }
        .status-panel {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .status-card {
            background-color: #fff;
            padding: 15px;
            border-radius: 5px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        .status-card h3 {
            margin: 0;
            color: #666;
        }
        .status-card p {
            margin: 10px 0 0;
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }
        #plot {
            width: 100%;
            height: 600px;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Training Monitor</h1>
        
        <div class="status-panel">
            <div class="status-card">
                <h3>Training Round</h3>
                <p id="training-round">-</p>
            </div>
            <div class="status-card">
                <h3>Active Nodes</h3>
                <p id="active-nodes">-</p>
            </div>
            <div class="status-card">
                <h3>GPU Nodes</h3>
                <p id="gpu-nodes">-</p>
            </div>
            <div class="status-card">
                <h3>CPU Nodes</h3>
                <p id="cpu-nodes">-</p>
            </div>
            <div class="status-card">
                <h3>Status</h3>
                <p id="training-status">-</p>
            </div>
        </div>
        
        <div id="plot"></div>
    </div>

    <script>
        function updateStatus() {
            $.get('/api/status', function(data) {
                // Update plot
                Plotly.newPlot('plot', JSON.parse(data.plot));
                
                // Update status cards
                $('#training-round').text(data.status.training_round);
                $('#active-nodes').text(data.status.active_nodes);
                $('#gpu-nodes').text(data.status.gpu_nodes);
                $('#cpu-nodes').text(data.status.cpu_nodes);
                $('#training-status').text(data.status.training_status);
            });
        }

        // Update every 5 seconds
        setInterval(updateStatus, 5000);
        updateStatus();  // Initial update
    </script>
</body>
</html>
        ''')

def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Web-based Training Monitor")
    parser.add_argument("--coordinator-address", type=str, required=True,
                        help="Coordinator address in the format <host>:<port>")
    parser.add_argument("--port", type=int, default=5000,
                        help="Port to run the web server on (default: 5000)")
    parser.add_argument("--interval", type=int, default=5,
                        help="Update interval in seconds (default: 5)")
    
    args = parser.parse_args()
    
    # Create templates
    create_templates()
    
    # Initialize monitor
    global monitor
    coordinator_url = f"http://{args.coordinator_address}"
    monitor = TrainingMonitor(coordinator_url, args.interval)
    
    try:
        # Run Flask app
        app.run(host='0.0.0.0', port=args.port, debug=True)
    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        if monitor:
            monitor.stop()

if __name__ == "__main__":
    main() 