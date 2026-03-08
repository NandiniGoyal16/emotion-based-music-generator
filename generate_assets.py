import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create assets directory if it doesn't exist
os.makedirs("assets", exist_ok=True)

def generate_performance_matrix(agent_name, music_type):
    """Generates a performance matrix heatmap like the one provided by the user."""
    metrics = ["Spectral Convergence", "Log Magnitude Distance", "Mel Cepstral Distortion", 
               "Pitch Accuracy", "Rhythm Consistency", "Actor Loss", "Critic Loss", "Reward"]
    categories = ["Pitch Accuracy", "Rhythm Consistency", "Emotion Match"]
    
    # Generate mock data consistent with the provided image
    # Seed with different values per agent AND music type for unique differentiation
    seed_val = hash(f"{agent_name}_{music_type}") % 10**6
    np.random.seed(seed_val)
    data = np.random.uniform(0.3, 0.8, (len(metrics), len(categories)))
    # Adjust losses to be negative like in the image
    data[5] = np.random.uniform(-0.1, 0.4, 3) # Actor Loss
    data[6] = np.random.uniform(-0.6, 0.1, 3) # Critic Loss
    
    plt.figure(figsize=(12, 10))
    # Using 'viridis' to match the green-yellow-purple aesthetic of the provided image
    sns.heatmap(data, annot=True, fmt=".3f", cmap="viridis", 
                xticklabels=categories, yticklabels=metrics,
                cbar_kws={'label': 'Metric Value'})
    
    plt.title(f"{agent_name} Performance Matrix ({music_type})", fontsize=16)
    plt.xticks(rotation=20, ha='right')
    plt.tight_layout()
    
    save_path = f"assets/{agent_name.lower()}_performance_matrix_{music_type.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Generated {save_path}")

def generate_learning_graph(agent_name, music_type):
    """Generates a learning performance line graph like the one provided by the user."""
    steps = np.linspace(1, 100, 100)
    
    # Generate smooth wavy lines like the ones in the provided image
    seed_val = hash(f"{agent_name}_{music_type}_graph") % 10**6
    np.random.seed(seed_val)
    actor_loss = -0.6 + 0.05 * np.sin(steps / (10 if music_type == "Indian" else 15)) + 0.03 * np.random.randn(100)
    critic_loss = 0.1 + 0.05 * np.cos(steps / (8 if music_type == "Indian" else 12)) + 0.02 * np.random.randn(100)
    
    # Softening the lines for the 'aesthetic' look in the image
    from scipy.interpolate import make_interp_spline
    X_smooth = np.linspace(steps.min(), steps.max(), 300)
    spl_actor = make_interp_spline(steps, actor_loss, k=3)
    spl_critic = make_interp_spline(steps, critic_loss, k=3)
    actor_smooth = spl_actor(X_smooth)
    critic_smooth = spl_critic(X_smooth)

    plt.figure(figsize=(12, 6))
    plt.plot(X_smooth, actor_smooth, color='mediumblue', linewidth=2.5, label='Actor Loss')
    plt.plot(X_smooth, critic_smooth, color='gold', linewidth=2.5, label='Critic Loss')
    
    plt.title(f"{agent_name} Learning Graph ({music_type})", fontsize=14)
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.5)
    plt.legend(loc='center right', fontsize=12)
    plt.tight_layout()
    
    save_path = f"assets/{agent_name.lower()}_learning_graph_{music_type.lower()}.png"
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Generated {save_path}")

if __name__ == "__main__":
    for mt in ["Indian", "Western"]:
        # Generate for SAC
        generate_performance_matrix("SAC", mt)
        generate_learning_graph("SAC", mt)
        
        # Generate for TRPO
        generate_performance_matrix("TRPO", mt)
        generate_learning_graph("TRPO", mt)
