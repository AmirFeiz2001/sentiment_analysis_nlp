import argparse
from src.preprocess import preprocess_text, load_stop_words
from src.analyze import load_emotions, analyze_emotions, sentiment_analyze
from src.visualize import plot_emotions

def main():
    parser = argparse.ArgumentParser(description="Sentiment and Emotion Analysis")
    parser.add_argument('--text_file', type=str, required=True, help='Path to input text file')
    parser.add_argument('--emotions_file', type=str, required=True, help='Path to emotions file')
    parser.add_argument('--output_dir', type=str, default='output', help='Directory to save outputs')
    args = parser.parse_args()

    try:
        with open(args.text_file, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return

    stop_words = load_stop_words()
    final_words = preprocess_text(text, stop_words)

    emotions_dict = load_emotions(args.emotions_file)
    emotion_counts = analyze_emotions(final_words, emotions_dict)
    print("Emotion Counts:", emotion_counts)

    plot_emotions(emotion_counts, args.output_dir + "/plots")

    user_text = input("Enter your text for sentiment analysis: ")
    print("=" * 120)
    cleaned_user_text = preprocess_text(user_text)[0]
    sentiment = sentiment_analyze(cleaned_user_text)
    print(f"Sentiment: {sentiment}")

if __name__ == "__main__":
    main()
