"""Plotting utilities for visualization and analysis."""

from typing import Optional, List
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_trajectory_comparison(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    times: np.ndarray,
    state_names: List[str] = None,
    controls: Optional[np.ndarray] = None,
    control_names: List[str] = None,
    save_path: Optional[str] = None,
    pred_std: Optional[np.ndarray] = None,
):
    """Plot comparison between true and predicted trajectories.
    
    Args:
        true_states: True states (n_steps, state_dim)
        pred_states: Predicted states (n_steps, state_dim) or (n_samples, n_steps, state_dim)
        times: Time array (n_steps,)
        state_names: Names of state variables
        controls: Control inputs (n_steps, control_dim)
        control_names: Names of control variables
        save_path: Path to save figure
        pred_std: Standard deviation for ensemble predictions (n_steps, state_dim)
    """
    if state_names is None:
        state_names = ["Ca", "Cb", "T", "Tc"]
    if control_names is None:
        control_names = ["F_in", "Tc_in"]
    
    # Handle ensemble predictions
    if len(pred_states.shape) == 3:
        # (n_samples, n_steps, state_dim) -> compute mean and std
        pred_mean = np.mean(pred_states, axis=0)
        pred_std = np.std(pred_states, axis=0)
        pred_states = pred_mean
    
    state_dim = true_states.shape[1]
    
    # Create figure
    if controls is not None:
        fig, axes = plt.subplots(state_dim + 1, 2, figsize=(14, 3 * (state_dim + 1)))
    else:
        fig, axes = plt.subplots(state_dim, 1, figsize=(10, 3 * state_dim))
        axes = axes.reshape(-1, 1)
    
    # Plot states
    for i in range(state_dim):
        ax = axes[i, 0] if controls is not None else axes[i, 0]
        ax.plot(times, true_states[:, i], 'b-', linewidth=2, label='True', alpha=0.8)
        ax.plot(times, pred_states[:, i], 'r--', linewidth=2, label='Predicted', alpha=0.8)
        
        # Add uncertainty band if available
        if pred_std is not None:
            ax.fill_between(
                times,
                pred_states[:, i] - 2 * pred_std[:, i],
                pred_states[:, i] + 2 * pred_std[:, i],
                color='red',
                alpha=0.2,
                label='±2σ'
            )
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(state_names[i], fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    # Plot controls
    if controls is not None:
        control_dim = controls.shape[1]
        for i in range(control_dim):
            ax = axes[i, 1]
            ax.plot(times, controls[:, i], 'g-', linewidth=2)
            ax.set_xlabel('Time', fontsize=12)
            ax.set_ylabel(control_names[i], fontsize=12)
            ax.grid(True, alpha=0.3)
        
        # Fill remaining subplots if needed
        for i in range(control_dim, state_dim + 1):
            if i < len(axes):
                axes[i, 1].axis('off')
    
    plt.suptitle('Trajectory Comparison: True vs Predicted', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_training_history(
    history_dict: dict,
    save_path: Optional[str] = None,
):
    """Plot training history.
    
    Args:
        history_dict: Dictionary with training history
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()
    
    steps = history_dict.get("step", list(range(len(history_dict["train_loss"]))))
    
    # Plot total loss
    ax = axes[0]
    ax.semilogy(steps, history_dict["train_loss"], 'b-', label='Train', linewidth=2)
    if "val_loss" in history_dict and len(history_dict["val_loss"]) > 0:
        val_steps = np.linspace(steps[0], steps[-1], len(history_dict["val_loss"]))
        ax.semilogy(val_steps, history_dict["val_loss"], 'r--', label='Val', linewidth=2)
    ax.set_xlabel('Step', fontsize=12)
    ax.set_ylabel('Total Loss', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Total Loss', fontsize=12, fontweight='bold')
    
    # Individual loss components (if available)
    loss_names = ['reconstruction', 'kl', 'trajectory', 'mass_balance', 'energy_balance']
    for idx, loss_name in enumerate(loss_names, start=1):
        if loss_name in history_dict and idx < len(axes):
            ax = axes[idx]
            ax.plot(steps, history_dict[loss_name], linewidth=2)
            ax.set_xlabel('Step', fontsize=12)
            ax.set_ylabel(loss_name.replace('_', ' ').title(), fontsize=12)
            ax.grid(True, alpha=0.3)
            ax.set_title(loss_name.replace('_', ' ').title(), fontsize=12, fontweight='bold')
    
    plt.suptitle('Training History', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_latent_space(
    z_trajectories: np.ndarray,
    labels: Optional[np.ndarray] = None,
    method: str = "pca",
    save_path: Optional[str] = None,
):
    """Visualize latent trajectories using PCA or t-SNE.
    
    Args:
        z_trajectories: Latent trajectories (n_trajectories, n_steps, latent_dim)
        labels: Optional labels for coloring
        method: Dimensionality reduction method ("pca" or "tsne")
        save_path: Path to save figure
    """
    from sklearn.decomposition import PCA
    
    # Reshape for dimensionality reduction
    n_traj, n_steps, latent_dim = z_trajectories.shape
    z_flat = z_trajectories.reshape(-1, latent_dim)
    
    # Reduce to 2D
    if method.lower() == "pca":
        reducer = PCA(n_components=2)
        z_2d = reducer.fit_transform(z_flat)
        explained_var = reducer.explained_variance_ratio_
        title = f'Latent Space (PCA: {explained_var[0]:.2%} + {explained_var[1]:.2%} variance)'
    else:
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=2, random_state=42)
        z_2d = reducer.fit_transform(z_flat)
        title = 'Latent Space (t-SNE)'
    
    # Reshape back to trajectories
    z_2d = z_2d.reshape(n_traj, n_steps, 2)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for i in range(n_traj):
        color = plt.cm.viridis(i / n_traj) if labels is None else plt.cm.tab10(labels[i])
        ax.plot(z_2d[i, :, 0], z_2d[i, :, 1], alpha=0.6, linewidth=1.5)
        ax.scatter(z_2d[i, 0, 0], z_2d[i, 0, 1], marker='o', s=50, color=color, edgecolors='black')
        ax.scatter(z_2d[i, -1, 0], z_2d[i, -1, 1], marker='s', s=50, color=color, edgecolors='black')
    
    ax.set_xlabel('Component 1', fontsize=12)
    ax.set_ylabel('Component 2', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add legend for start/end markers
    ax.scatter([], [], marker='o', s=50, color='gray', edgecolors='black', label='Start')
    ax.scatter([], [], marker='s', s=50, color='gray', edgecolors='black', label='End')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_conservation_violation(
    mass_residuals: np.ndarray,
    energy_residuals: np.ndarray,
    times: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot conservation law violations over time.
    
    Args:
        mass_residuals: Mass balance residuals (n_steps-1,)
        energy_residuals: Energy balance residuals (n_steps-1,)
        times: Time array (n_steps-1,)
        save_path: Path to save figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # Mass balance
    ax = axes[0]
    ax.semilogy(times, mass_residuals, 'b-', linewidth=2, label='Model')
    ax.axhline(y=1e-3, color='r', linestyle='--', linewidth=2, label='Target (1e-3)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Mass Balance Residual', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Mass Conservation Violation', fontsize=12, fontweight='bold')
    
    # Energy balance
    ax = axes[1]
    ax.semilogy(times, energy_residuals, 'g-', linewidth=2, label='Model')
    ax.axhline(y=1e-2, color='r', linestyle='--', linewidth=2, label='Target (1e-2)')
    ax.set_xlabel('Time', fontsize=12)
    ax.set_ylabel('Energy Balance Residual', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_title('Energy Conservation Violation', fontsize=12, fontweight='bold')
    
    plt.suptitle('Conservation Law Violations', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_mpc_results(
    states: np.ndarray,
    controls: np.ndarray,
    setpoints: np.ndarray,
    times: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plot MPC control results.
    
    Args:
        states: State trajectory (n_steps, state_dim)
        controls: Control trajectory (n_steps, control_dim)
        setpoints: Setpoint trajectory (n_steps, state_dim)
        times: Time array (n_steps,)
        save_path: Path to save figure
    """
    state_names = ["Ca", "Cb", "T", "Tc"]
    control_names = ["F_in", "Tc_in"]
    
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    
    # Plot states vs setpoints
    for i in range(4):
        ax = axes[i // 2, i % 2]
        ax.plot(times, states[:, i], 'b-', linewidth=2, label='Actual')
        ax.plot(times, setpoints[:, i], 'r--', linewidth=2, label='Setpoint')
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(state_names[i], fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{state_names[i]} Tracking', fontsize=12, fontweight='bold')
    
    # Plot controls
    for i in range(2):
        ax = axes[2, i]
        ax.plot(times, controls[:, i], 'g-', linewidth=2)
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel(control_names[i], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{control_names[i]} Action', fontsize=12, fontweight='bold')
    
    plt.suptitle('MPC Control Results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig


def plot_prediction_error(
    true_states: np.ndarray,
    pred_states: np.ndarray,
    times: np.ndarray,
    state_names: List[str] = None,
    save_path: Optional[str] = None,
):
    """Plot prediction error over time.
    
    Args:
        true_states: True states (n_steps, state_dim)
        pred_states: Predicted states (n_steps, state_dim)
        times: Time array (n_steps,)
        state_names: Names of state variables
        save_path: Path to save figure
    """
    if state_names is None:
        state_names = ["Ca", "Cb", "T", "Tc"]
    
    errors = np.abs(true_states - pred_states)
    relative_errors = errors / (np.abs(true_states) + 1e-8) * 100
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i in range(4):
        ax = axes[i]
        ax.plot(times, errors[:, i], 'b-', linewidth=2, label='Absolute Error')
        ax_twin = ax.twinx()
        ax_twin.plot(times, relative_errors[:, i], 'r--', linewidth=2, label='Relative Error (%)')
        
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Absolute Error', fontsize=12, color='b')
        ax_twin.set_ylabel('Relative Error (%)', fontsize=12, color='r')
        ax.tick_params(axis='y', labelcolor='b')
        ax_twin.tick_params(axis='y', labelcolor='r')
        ax.grid(True, alpha=0.3)
        ax.set_title(f'{state_names[i]} Prediction Error', fontsize=12, fontweight='bold')
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax_twin.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    
    plt.suptitle('Prediction Error Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot to {save_path}")
    
    return fig
