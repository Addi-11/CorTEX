#!/usr/bin/env python3
"""
Script to plot training progress from HuggingFace trainer output.
Can parse logs from tmux capture, checkpoint trainer_state.json, or TensorBoard.
"""

import os
import re
import json
import argparse
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def parse_trainer_state(checkpoint_dir: str):
    """Parse trainer_state.json from checkpoint directory."""
    trainer_state_path = Path(checkpoint_dir) / "trainer_state.json"
    
    # Try to find the latest checkpoint
    if not trainer_state_path.exists():
        checkpoint_path = Path(checkpoint_dir)
        checkpoints = sorted(checkpoint_path.glob("checkpoint-*"), 
                           key=lambda x: int(x.name.split("-")[1]))
        if checkpoints:
            trainer_state_path = checkpoints[-1] / "trainer_state.json"
    
    if not trainer_state_path.exists():
        print(f"No trainer_state.json found in {checkpoint_dir}")
        return None
    
    with open(trainer_state_path) as f:
        state = json.load(f)
    
    log_history = state.get("log_history", [])
    
    steps = []
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    eval_losses = []
    eval_steps = []
    
    for entry in log_history:
        if "loss" in entry and "eval_loss" not in entry:
            steps.append(entry.get("step", 0))
            losses.append(entry["loss"])
            epochs.append(entry.get("epoch", 0))
            learning_rates.append(entry.get("learning_rate", 0))
            if "grad_norm" in entry:
                grad_norms.append(entry["grad_norm"])
        elif "eval_loss" in entry:
            eval_steps.append(entry.get("step", 0))
            eval_losses.append(entry["eval_loss"])
    
    return {
        "steps": steps,
        "losses": losses,
        "epochs": epochs,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
        "eval_steps": eval_steps,
        "eval_losses": eval_losses,
        "source": "trainer_state.json"
    }

def parse_log_file(log_file: str):
    """Parse training logs from a text file (e.g., tmux capture output)."""
    with open(log_file) as f:
        content = f.read()
    return parse_log_text(content)

def parse_log_text(content: str):
    """Parse training logs from text content."""
    # Pattern for HuggingFace trainer log entries
    # {'loss': 0.6694, 'grad_norm': 3.20, 'learning_rate': 1.64e-05, 'epoch': 0.03}
    pattern = r"\{'loss': ([\d.]+), 'grad_norm': ([\d.]+), 'learning_rate': ([\d.e-]+), 'epoch': ([\d.]+)\}"
    
    steps = []
    losses = []
    epochs = []
    learning_rates = []
    grad_norms = []
    
    # Also try to extract step from progress bar
    # 3%|█████▏ | 55/1805 [18:47<10:31:33, 21.65s/it]
    progress_pattern = r"(\d+)%\|.*\| (\d+)/(\d+)"
    
    matches = re.findall(pattern, content)
    step_matches = re.findall(progress_pattern, content)
    
    for i, match in enumerate(matches):
        loss, grad_norm, lr, epoch = match
        losses.append(float(loss))
        grad_norms.append(float(grad_norm))
        learning_rates.append(float(lr))
        epochs.append(float(epoch))
        
        # Try to estimate step from matches
        # HF trainer logs every 10 steps by default
        steps.append((i + 1) * 10)
    
    # Try to get more accurate steps from progress bars
    if step_matches:
        last_progress = step_matches[-1]
        current_step = int(last_progress[1])
        total_steps = int(last_progress[2])
        print(f"Progress bar indicates: {current_step}/{total_steps} steps")
    
    return {
        "steps": steps,
        "losses": losses,
        "epochs": epochs,
        "learning_rates": learning_rates,
        "grad_norms": grad_norms,
        "eval_steps": [],
        "eval_losses": [],
        "source": "log_text"
    }

def plot_training_metrics(data: dict, output_dir: str = None, title_prefix: str = ""):
    """Plot training metrics."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{title_prefix}Training Progress - CorTEX Tool Selection V3', fontsize=14, fontweight='bold')
    
    # 1. Training Loss
    ax1 = axes[0, 0]
    if data["losses"]:
        ax1.plot(data["steps"], data["losses"], 'b-', linewidth=2, label='Train Loss', marker='o', markersize=3)
        
        # Add trend line
        if len(data["steps"]) > 5:
            z = np.polyfit(data["steps"], data["losses"], 1)
            p = np.poly1d(z)
            ax1.plot(data["steps"], p(data["steps"]), 'b--', alpha=0.5, label='Trend')
        
        # Add eval loss if available
        if data["eval_losses"]:
            ax1.plot(data["eval_steps"], data["eval_losses"], 'r-', linewidth=2, 
                    label='Eval Loss', marker='s', markersize=4)
        
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training & Evaluation Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(bottom=0)
    else:
        ax1.text(0.5, 0.5, 'No loss data', ha='center', va='center', transform=ax1.transAxes)
    
    # 2. Learning Rate Schedule
    ax2 = axes[0, 1]
    if data["learning_rates"]:
        ax2.plot(data["steps"], data["learning_rates"], 'g-', linewidth=2, marker='o', markersize=3)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.grid(True, alpha=0.3)
        ax2.ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    else:
        ax2.text(0.5, 0.5, 'No LR data', ha='center', va='center', transform=ax2.transAxes)
    
    # 3. Gradient Norm
    ax3 = axes[1, 0]
    if data["grad_norms"]:
        ax3.plot(data["steps"], data["grad_norms"], 'm-', linewidth=2, marker='o', markersize=3)
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Norm (Training Stability)')
        ax3.grid(True, alpha=0.3)
    else:
        ax3.text(0.5, 0.5, 'No grad norm data', ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Loss by Epoch
    ax4 = axes[1, 1]
    if data["losses"] and data["epochs"]:
        ax4.plot(data["epochs"], data["losses"], 'c-', linewidth=2, marker='o', markersize=3)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss')
        ax4.set_title('Loss by Epoch')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim(bottom=0)
        
        # Mark epoch boundaries
        for i in range(1, 6):
            ax4.axvline(x=i, color='gray', linestyle='--', alpha=0.3)
    else:
        ax4.text(0.5, 0.5, 'No epoch data', ha='center', va='center', transform=ax4.transAxes)
    
    plt.tight_layout()
    
    # Save plot
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'training_progress.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    
    plt.show()
    return fig

def print_training_summary(data: dict):
    """Print a summary of training metrics."""
    print("\n" + "="*60)
    print("TRAINING SUMMARY")
    print("="*60)
    
    if data["losses"]:
        print(f"Total logged steps: {len(data['losses'])} entries")
        print(f"Latest step: {data['steps'][-1] if data['steps'] else 'N/A'}")
        print(f"Latest epoch: {data['epochs'][-1]:.2f}" if data['epochs'] else "N/A")
        print(f"\nLoss Statistics:")
        print(f"  Initial loss: {data['losses'][0]:.4f}")
        print(f"  Final loss:   {data['losses'][-1]:.4f}")
        print(f"  Min loss:     {min(data['losses']):.4f}")
        print(f"  Max loss:     {max(data['losses']):.4f}")
        print(f"  Loss reduction: {((data['losses'][0] - data['losses'][-1]) / data['losses'][0] * 100):.1f}%")
        
        if data["learning_rates"]:
            print(f"\nLearning Rate:")
            print(f"  Current LR: {data['learning_rates'][-1]:.2e}")
            print(f"  Max LR:     {max(data['learning_rates']):.2e}")
        
        if data["grad_norms"]:
            print(f"\nGradient Norm:")
            print(f"  Current: {data['grad_norms'][-1]:.2f}")
            print(f"  Max:     {max(data['grad_norms']):.2f}")
            print(f"  Avg:     {np.mean(data['grad_norms']):.2f}")
        
        if data["eval_losses"]:
            print(f"\nEvaluation Loss:")
            print(f"  Latest:  {data['eval_losses'][-1]:.4f}")
            print(f"  Best:    {min(data['eval_losses']):.4f}")
    else:
        print("No training data available yet.")
    
    print("="*60 + "\n")

def main():
    parser = argparse.ArgumentParser(description='Plot training progress')
    parser.add_argument('--checkpoint-dir', '-c', type=str, 
                       default='checkpoints/llava-med-tool-selection-v3',
                       help='Path to checkpoint directory')
    parser.add_argument('--log-file', '-l', type=str,
                       help='Path to log file from tmux capture')
    parser.add_argument('--output-dir', '-o', type=str,
                       default='training_plots',
                       help='Directory to save plots')
    parser.add_argument('--tmux-session', '-t', type=str,
                       help='Tmux session name to capture logs from')
    
    args = parser.parse_args()
    
    data = None
    
    # Try different sources in order of preference
    # 1. Tmux session (live logs)
    if args.tmux_session:
        import subprocess
        result = subprocess.run(
            ['tmux', 'capture-pane', '-t', args.tmux_session, '-p', '-S', '-1000'],
            capture_output=True, text=True
        )
        if result.returncode == 0:
            data = parse_log_text(result.stdout)
            print(f"Parsed logs from tmux session: {args.tmux_session}")
    
    # 2. Log file
    if data is None and args.log_file and os.path.exists(args.log_file):
        data = parse_log_file(args.log_file)
        print(f"Parsed logs from file: {args.log_file}")
    
    # 3. Trainer state (checkpoint)
    if data is None and args.checkpoint_dir and os.path.exists(args.checkpoint_dir):
        data = parse_trainer_state(args.checkpoint_dir)
        if data:
            print(f"Parsed trainer_state.json from: {args.checkpoint_dir}")
    
    if data is None or not data["losses"]:
        print("No training data found! Try:")
        print("  1. --tmux-session train_v3")
        print("  2. --log-file path/to/logs.txt")
        print("  3. --checkpoint-dir checkpoints/llava-med-tool-selection-v3")
        return
    
    print_training_summary(data)
    plot_training_metrics(data, args.output_dir)

if __name__ == "__main__":
    main()
