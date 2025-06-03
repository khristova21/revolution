# Author: Safa Obuz
# Date: 6/2/2025
# Purpose: Manual Logging

import os
import time
import argparse
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def get_latest_log_dir(experiment_name):
    """Find the latest log directory for the experiment"""
    log_root = os.path.join("logs", "skrl", experiment_name)
    if not os.path.exists(log_root):
        return None
    
    subdirs = [d for d in os.listdir(log_root) if os.path.isdir(os.path.join(log_root, d))]
    if not subdirs:
        return None
    
    latest = sorted(subdirs)[-1]
    return os.path.join(log_root, latest)

def plot_training_curves(log_dir):
    """Plot training curves from TensorBoard logs"""
    event_path = None
    for root, dirs, files in os.walk(log_dir):
        for file in files:
            if file.startswith("events.out.tfevents"):
                event_path = os.path.join(root, file)
                break
    
    if not event_path:
        print("No TensorBoard event files found")
        return None, None
    
    # Load TensorBoard data
    ea = EventAccumulator(event_path)
    ea.Reload()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f'Training Progress: {os.path.basename(log_dir)}')
    
    metrics = [
        ('Reward/cumulative_reward', axes[0, 0], 'Cumulative Reward'),
        ('Reward/episodic_reward', axes[0, 1], 'Episodic Reward'),
        ('Loss/policy_loss', axes[1, 0], 'Policy Loss'),
        ('Loss/value_loss', axes[1, 1], 'Value Loss'),
    ]
    
    for metric_name, ax, title in metrics:
        try:
            scalar_events = ea.Scalars(metric_name)
            if scalar_events:
                steps = [e.step for e in scalar_events]
                values = [e.value for e in scalar_events]
                ax.plot(steps, values)
                ax.set_xlabel('Steps')
                ax.set_ylabel(title)
                ax.set_title(title)
                ax.grid(True, alpha=0.3)
        except KeyError:
            ax.text(0.5, 0.5, f'{metric_name} not found', 
                   ha='center', va='center', transform=ax.transAxes)
    
    plt.tight_layout()
    return fig, ea

def live_monitor(experiment_name, update_interval=5):
    """Live monitoring of training progress"""
    print(f"Monitoring experiment: {experiment_name}")
    print("Press Ctrl+C to stop")
    
    plt.ion()
    fig = None
    
    while True:
        try:
            log_dir = get_latest_log_dir(experiment_name)
            if not log_dir:
                print(f"No logs found for experiment: {experiment_name}")
                time.sleep(update_interval)
                continue
            
            # Clear and update plot
            if fig:
                plt.close(fig)
            
            fig, ea = plot_training_curves(log_dir)
            if fig:
                plt.draw()
                plt.pause(0.1)
            
            # Print latest statistics
            print(f"\n[{time.strftime('%H:%M:%S')}] Latest stats:")
            try:
                reward_events = ea.Scalars('Reward/episodic_reward')
                if reward_events:
                    latest_reward = reward_events[-1]
                    print(f"  Episode Reward: {latest_reward.value:.2f} (step {latest_reward.step})")
            except:
                pass
            
            time.sleep(update_interval)
            
        except KeyboardInterrupt:
            print("\nStopping monitor...")
            break
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(update_interval)
    
    plt.ioff()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monitor training progress")
    parser.add_argument("--experiment", type=str, default="custom_agent", 
                       help="Experiment name to monitor")
    parser.add_argument("--interval", type=int, default=5,
                       help="Update interval in seconds")
    parser.add_argument("--static", action="store_true",
                       help="Generate static plots instead of live monitoring")
    
    args = parser.parse_args()
    
    if args.static:
        log_dir = get_latest_log_dir(args.experiment)
        if log_dir:
            fig, _ = plot_training_curves(log_dir)
            if fig:
                plt.show()
        else:
            print(f"No logs found for experiment: {args.experiment}")
    else:
        live_monitor(args.experiment, args.interval)