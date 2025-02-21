import os
import matplotlib.pyplot as plt
from datetime import datetime

def plot_emotions(emotion_counts, output_dir="output/plots"):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig, ax = plt.subplots()
    ax.bar(emotion_counts.keys(), emotion_counts.values())
    fig.autofmt_xdate()
    plt.title("Emotion Distribution")
    plt.xlabel("Emotions")
    plt.ylabel("Count")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = os.path.join(output_dir, f"emotion_distribution_{timestamp}.png")
    plt.savefig(output_path)
    plt.close()
    print(f"Emotion plot saved to {output_path}")
