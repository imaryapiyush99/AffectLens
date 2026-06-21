import matplotlib.pyplot as plt

def moving_average(data: list[float], window_size: int) -> list[float]:
    """
    Computes the moving average of a list of numerical data.

    Args: data (list[float]): A list of numerical values, window_size (int): The size of the moving average window.

    Returns: list[float]: A list containing the moving average values. 
    """
    if not isinstance(data, list):
        raise ValueError("Data must be a list of numerical values.")
    if not all(isinstance(x, (int, float)) for x in data):
        raise ValueError("All elements in data must be numeric.")
    if not isinstance(window_size, int) or window_size <= 0:
        raise ValueError("Window size must be a positive integer.")
    if window_size > len(data):
        raise ValueError("Window size cannot be larger than the data length.")

    moving_averages = []
    for i in range(len(data)):
        start_index = max(0, i - window_size + 1)
        window = data[start_index:i + 1]
        moving_averages.append(sum(window) / len(window))
    return moving_averages

def plot_sentiment_trend(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]]) -> plt.Figure:
    """
    Plots ensemble sentiment scores across the sequence of posts and highlights posts flagged as highly volatile.

    Args: 
        enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data.

    Returns: plt.Figure: A matplotlib figure object representing the sentiment trend plot.    
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Ensemble_score" in post and "High_volatility_flag" in post for post in enriched_posts):
        raise ValueError("Each post must contain 'Ensemble_score' and 'High_volatility_flag' keys.")
    
    ensemble_scores, post_sequences, boolean_values = [], [], []
    for i, post in enumerate(enriched_posts):
            ensemble_scores.append(post["Ensemble_score"])
            boolean_values.append(post["High_volatility_flag"])
            post_sequences.append(i + 1) 
         
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_values = ensemble_scores

    if len(ensemble_scores) > 1000:
        plot_values = moving_average(ensemble_scores, 100)

    ax.plot(post_sequences, plot_values)
    
    if len(ensemble_scores) > 1000:
        ax.set_title("Sentiment Trend Over Time (100-Post Moving Average)")
    else:
        ax.set_title("Sentiment Trend Over Time")
    ax.set_xlabel('Post Sequence', fontsize=14)
    ax.set_ylabel('Ensemble Sentiment Score', fontsize=14)
    ax.grid(True)
    if len(ensemble_scores) <= 1000:
        first_label = True
        for x, y, flag in zip(post_sequences, ensemble_scores, boolean_values):
            if flag:
                ax.scatter(x, y, c="red", s=100, marker="o", label="High Volatility" if first_label else None)
                first_label = False
        if not first_label:
            ax.legend()   
    plt.tight_layout()

    return fig



def plot_volatility_trend(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]], threshold: float) -> plt.Figure:
    """
    Plots volatility scores across the sequence of posts and highlights posts flagged as highly volatile.

    Args: 
        enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data, threshold (float): The threshold for identifying highly volatile posts.

    Returns: plt.Figure: A matplotlib figure object representing the volatility trend plot.    
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Volatility_score" in post for post in enriched_posts):
        raise ValueError("Each post must contain a 'Volatility_score' key.")
    if not isinstance(threshold, (int, float)):
        raise ValueError("Threshold must be a numeric value.")
    if not (0 <= threshold <= 1):
        raise ValueError("Threshold should be between 0 and 1.")
    
    volatility_scores, post_sequences = [], []
    for i, post in enumerate(enriched_posts):
            volatility_scores.append(post["Volatility_score"])
            post_sequences.append(i + 1)
    valid_scores = [(score, seq) for score, seq in zip(volatility_scores, post_sequences) if score is not None]
    post_sequences = [seq for _, seq in valid_scores]
    volatility_scores = [score for score, _ in valid_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_values = volatility_scores
    if len(volatility_scores) > 1000:
        plot_values = moving_average(volatility_scores, 100)
    ax.plot(post_sequences, plot_values, marker="o", linestyle="-", color="orange", linewidth=0.5)
    ax.axhline(y=threshold, color="red", linestyle="--", label=f"Volatility Threshold ({threshold})")
    if len(volatility_scores) > 1000:
        ax.set_title("Volatility Trend Over Time (100-Post Moving Average)")
    else:
        ax.set_title('Volatility Trend Over Time', fontsize=16)
    ax.set_xlabel('Post Sequence', fontsize=14)
    ax.set_ylabel('Volatility Score', fontsize=14)
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_emotional_swing_trend(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]]) -> plt.Figure:
    """
    Plots emotional swing scores across the sequence of posts and highlights posts flagged as highly volatile.

    Args: enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data.

    Returns: plt.Figure: A matplotlib figure object representing the emotional swing trend plot.
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Emotional_swing" in post and "High_volatility_flag" in post for post in enriched_posts):
        raise ValueError("Each post must contain 'Emotional_swing' and 'High_volatility_flag' keys.")
    emotional_swing_scores, post_sequences, boolean_values = [], [], []
    for i, post in enumerate(enriched_posts):
            emotional_swing_scores.append(post["Emotional_swing"])
            boolean_values.append(post["High_volatility_flag"])
            post_sequences.append(i + 1)
    valid_scores = [(score, seq, flag) for score, seq, flag in zip(emotional_swing_scores, post_sequences, boolean_values) if score is not None]
    post_sequences = [seq for _, seq, _ in valid_scores]
    emotional_swing_scores = [score for score, _, _ in valid_scores]
    boolean_values = [flag for _, _, flag in valid_scores]

    fig, ax = plt.subplots(figsize=(10, 6))
    plot_values = emotional_swing_scores
    if len(emotional_swing_scores) > 1000:
        plot_values = moving_average(emotional_swing_scores, 100)
    ax.plot(post_sequences, plot_values, marker="o", linestyle="-", color="green", linewidth=0.5)
    if len(emotional_swing_scores) > 1000:
        ax.set_title("Emotional Swing Trend Over Time (100-Post Moving Average)")
    else:
        ax.set_title('Emotional Swing Trend Over Time', fontsize=16)
    ax.set_xlabel('Post Sequence', fontsize=14)
    ax.set_ylabel('Emotional Swing Score', fontsize=14)

    first_label = True 
    if len(emotional_swing_scores) <= 1000:
        for x, y, flag in zip(post_sequences, emotional_swing_scores, boolean_values):
            if flag:
                ax.scatter(x, y, c="red", s=100, marker="o", label="High Volatility" if first_label else None) 
                first_label = False   
        if not first_label:
            ax.legend()
    ax.grid(True)
    plt.tight_layout()
    return fig


def plot_emotion_distribution(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]]) -> plt.Figure:
    """
    Plots the distribution of emotions across the sequence of posts.

    Args: enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data.

    Returns: plt.Figure: A matplotlib figure object representing the emotion distribution plot.
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Emotion" in post for post in enriched_posts):
        raise ValueError("Each post must contain an 'Emotion' key.")
    
    emotion_counts = {}
    for post in enriched_posts:
        emotions = list(post["Emotion"])
        for emotion in emotions:
            emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(emotion_counts.keys(), emotion_counts.values(), color='purple')
    ax.set_title('Emotion Distribution Across Posts', fontsize=16)
    ax.set_xlabel('Emotions', fontsize=14)
    ax.set_ylabel('Count', fontsize=14)
    ax.grid(axis='y')

    plt.tight_layout()
    return fig

def plot_emotion_heatmap(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]]) -> plt.Figure:
    """
    Plots a heatmap showing the presence of emotions across the sequence of posts.

    Args: enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data.

    Returns: plt.Figure: A matplotlib figure object representing the emotion heatmap.
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Emotion" in post for post in enriched_posts):
        raise ValueError("Each post must contain an 'Emotion' key.")

    all_emotions = []
    for post in enriched_posts:
        all_emotions.extend(emotion for emotion in post["Emotion"] if emotion not in all_emotions)

    heatmap_data = []
    for emotion in all_emotions:
        row = []
        for post in enriched_posts:       
            row.append(1 if emotion in post["Emotion"] else 0)
        heatmap_data.append(row)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(heatmap_data, aspect="auto")

    ax.set_xticks([])

    ax.set_yticks(range(len(all_emotions)))
    ax.set_yticklabels(all_emotions)

    ax.set_title('Emotion Heatmap Across Posts', fontsize=16)
    ax.set_xlabel('Post Sequence', fontsize=14)
    ax.set_ylabel('Emotions', fontsize=14)
    plt.colorbar(im, ax=ax, label='Presence of Emotion (1=Present, 0=Absent)')
    plt.tight_layout()
    return fig


def plot_emotion_transition_matrix(enriched_posts: list[dict[str, str | int | float | None | list[str | float]]]) -> plt.Figure:
    """
    Plots a transition matrix showing the movement of emotions across the sequence of posts.

    Args: enriched_posts (list[dict[str, str | int | float | None | list[str | float]]]): A list of dictionaries containing enriched post data.

    Returns: plt.Figure: A matplotlib figure object representing the emotion transition matrix.
    """
    if not isinstance(enriched_posts, list):
        raise ValueError("enriched_posts must be a list of dictionaries.")
    if not enriched_posts:
        raise ValueError("The list of enriched posts is empty.")
    if not all(isinstance(post, dict) for post in enriched_posts):
        raise ValueError("All items in enriched_posts must be dictionaries.")
    if not all("Emotion_scores" in post for post in enriched_posts):
        raise ValueError("Each post must contain an 'Emotion_scores' key.")
    dominant_emotions = []
    for post in enriched_posts:
        dominant_emotions.append(max(post["Emotion_scores"], key=post["Emotion_scores"].get))

    transition_matrix = [
        [0] * len(set(dominant_emotions)) for _ in range(len(set(dominant_emotions)))
    ]
    all_emotions = sorted(set(dominant_emotions))
    emotion_to_index = {
        emotion: index for index, emotion in enumerate(all_emotions)
    }

    for i in range(len(dominant_emotions) - 1):
        from_index = emotion_to_index[dominant_emotions[i]]
        to_index = emotion_to_index[dominant_emotions[i + 1]]
        transition_matrix[from_index][to_index] += 1

    fig, ax = plt.subplots(figsize=(8, 8))  

    im = ax.imshow(transition_matrix, aspect="auto")
    ax.set_xticks([])

    ax.set_yticks(range(len(all_emotions)))
    ax.set_yticklabels(all_emotions)

    ax.set_xlabel("To Emotion")
    ax.set_ylabel("From Emotion")
    ax.set_title("Emotion Transition Matrix", fontsize=16)
    plt.colorbar(im, ax=ax, label="Transition Count")
    plt.tight_layout()
    return fig