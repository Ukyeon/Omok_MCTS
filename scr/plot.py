import matplotlib.pyplot as plt
import numpy as np

def plot_learning_curve_vs():
    # Open the log.txt file in read mode
    with open('logs1.txt', 'r', encoding='utf-8') as file:
        # Read all lines from the file
        log_data = file.readlines()

    results = []
    for line in log_data:
        # Splitting the line by '][' to extract the [win, lose, draw] part
        result_text = line.split('][')[-1]  # Extracting the last part of the line
        result_text = result_text.replace('[', '').replace(']', '')  # Removing square brackets
        win_lose_draw = list(map(int, result_text.split(', ')))  # Converting to integers and splitting

        results.append(win_lose_draw)

    # Separate win, lose, draw data
    wins = [result[0] for result in results]
    loses = [result[1] for result in results]
    draws = [result[2] for result in results]

    # Create a plot
    iterations = range(1, len(log_data) + 1)  # Assuming each log corresponds to an iteration
    plt.plot(iterations, wins, label='Win counts for RL')
    plt.plot(iterations, loses, label='Win counts for A/B')
    plt.plot(iterations, draws, label='Draws')

    # Add labels and legend
    plt.xlabel('Games (Iterations)')
    plt.ylabel('Results')
    plt.legend()

    # Show the plot
    plt.show()


def plot_curve():
    # Open the log.txt file in read mode
    with open('logs2.txt', 'r', encoding='utf-8') as file:
        # Read all lines from the file
        log_data = file.readlines()

    results = []
    for line in log_data:
        # Splitting the line by tabs to extract the [win, lose, draw] part
        win_lose_draw = list(map(int, line.split('\t')))  # Converting to integers and splitting
        results.append(win_lose_draw)

    # Separate win, lose, draw data
    wins = [result[0] for result in results]
    loses = [result[1] for result in results]
    draws = [result[2] for result in results]

    # Create a plot
    iterations = range(1, len(log_data) + 1)  # Assuming each log corresponds to an iteration
    plt.plot(iterations, wins, label='Win counts for MCTS')
    plt.plot(iterations, loses, label='Win counts for RL')
    plt.plot(iterations, draws, label='Draws')

    # Add labels and legend
    plt.xlabel('Games (Iterations)')
    plt.ylabel('Results')
    plt.legend()

    # Show the plot
    plt.show()


def plot_learning_curve(rewards):
    """
    Plot the learning curve with confidence bands and an upper bound line.

    Args:
        learning_curve (numpy.ndarray): Array containing the learning curve data.
        title (str): Title for the plot.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for epsilon, learning_curve in rewards.items():
        trials, episodes = learning_curve.shape
        avg_learning_curve = np.mean(learning_curve, axis=0)
        std_dev = np.std(learning_curve, axis=0)
        standard_error = std_dev / np.sqrt(trials)
        confidence_band = 1.96 * standard_error  # 1.96 times standard error for 95% confidence interval
        upper_bound_line = avg_learning_curve + confidence_band

        # Plot the average learning curve
        x = range(1, episodes + 1)
        ax.plot(x, avg_learning_curve, label=f'ε = {epsilon}')

        # Add confidence interval shading
        ax.fill_between(x, avg_learning_curve - confidence_band, avg_learning_curve + confidence_band, alpha=0.2,
                        label=f'95% CI (ε = {epsilon})')

    # Add labels and a title
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Average Returns')
    ax.set_title('Learning Curves for on-policy first-visit Monte-Carlo (RaceTracks)')

    # Show the legend
    ax.legend()
    plt.show()

def plot_episodes_steps(steps_list, title):
    plt.plot(steps_list)
    plt.title(f'WindyGridWorld_{title}', fontsize='large')
    plt.xlabel("Number of Steps taken")
    plt.ylabel("Number of Episodes")
    plt.show()

def plot_histogram(targets):
    plt.hist(targets, bins=20, edgecolor='black')
    plt.xlabel('Learning Targets')
    plt.ylabel('Frequency')
    plt.title('Histogram of Learning Targets')
    plt.show()


plot_learning_curve_vs()
plot_curve()
