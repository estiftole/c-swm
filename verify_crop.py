import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import ale_py
from PIL import Image

def verify_crop(env_id, crop_ratio, warmstart=0):
    print(f"Testing {env_id} with crop {crop_ratio}...")
    env = gym.make(env_id, render_mode="rgb_array")

    ob, info = env.reset(seed=42)

    # Run through the warmstart frames
    for _ in range(warmstart):
        ob, _, _, _, _ = env.step(env.action_space.sample())

    original_img = ob

    # Apply the exact crop and resize logic from your data script
    cropped_img = original_img[crop_ratio[0]:crop_ratio[1]]
    final_50x50 = np.array(Image.fromarray(cropped_img).resize((50, 50), Image.Resampling.LANCZOS))

    # Visualization
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Panel 1: Original with cut lines
    axes[0].imshow(original_img)
    axes[0].axhline(crop_ratio[0], color='red', linestyle='--', linewidth=2, label='Top Crop')
    axes[0].axhline(crop_ratio[1], color='red', linestyle='--', linewidth=2, label='Bottom Crop')
    axes[0].set_title(f"Original ({original_img.shape[0]}x{original_img.shape[1]})\nRed lines = Crop boundaries")
    axes[0].legend()

    # Panel 2: The Cropped Area
    axes[1].imshow(cropped_img)
    axes[1].set_title(f"Cropped Field ({cropped_img.shape[0]}x{cropped_img.shape[1]})")

    # Panel 3: What the CNN actually sees
    axes[2].imshow(final_50x50)
    axes[2].set_title(f"Network Input (50x50)")

    plt.tight_layout()
    plt.show()
    env.close()

if __name__ == '__main__':
    # Test Breakout
    verify_crop('ALE/Breakout-v5', crop_ratio=(34, 194), warmstart=90)

    # Test Centipede
    verify_crop('ALE/Centipede-v5', crop_ratio=(25, 182), warmstart=60)
