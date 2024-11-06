import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def calculate_mean_first_n(data, n):
    """Calculate mean of first n data points"""
    return np.mean(data[:n])


def calculate_cusum(data, target_mean, sigma, k, h):
    """
    Calculate CUSUM values and identify out-of-control points.

    Parameters:
    - data: List of observed data points
    - target_mean: Target mean (μ₀)
    - sigma: Process standard deviation (σ)
    - k: Reference value (K = kσ, where k is typically 0.5)
    - h: Decision interval (H = hσ, where h is typically 4 or 5)
    """
    # Initialize arrays to store CUSUM values
    n = len(data)
    S_plus = np.zeros(n)
    S_minus = np.zeros(n)
    out_of_control_points = []

    # Calculate standardized values
    standardized_data = [
        (x - target_mean) for x in data
    ]  # Remove sigma division for now

    # Calculate CUSUM values
    for i in range(n):
        if i == 0:
            # First point
            S_plus[i] = max(0, standardized_data[i] - k)
            S_minus[i] = max(0, -standardized_data[i] - k)
        else:
            # Recursive CUSUM calculation
            S_plus[i] = max(0, S_plus[i - 1] + standardized_data[i] - k)
            S_minus[i] = max(0, S_minus[i - 1] - standardized_data[i] - k)

        # Check for out-of-control points
        if S_plus[i] > h or S_minus[i] > h:
            out_of_control_points.append(i)

    return S_plus, S_minus, out_of_control_points


def plot_cusum_chart(
    data, S_plus, S_minus, h, out_of_control_points, title="CUSUM Control Chart"
):
    """
    Plot the CUSUM control chart
    """
    plt.figure(figsize=(20, 20))

    # Plot CUSUM values
    plt.plot(range(1, len(data) + 1), S_plus, "b-", label="CUSUM⁺", linewidth=1.5)
    plt.plot(range(1, len(data) + 1), S_minus, "r-", label="CUSUM⁻", linewidth=1.5)

    # Plot control limits
    plt.axhline(y=h, color="k", linestyle="--", label=f"H = ±{h:.2f}")
    plt.axhline(y=-h, color="k", linestyle="--")
    plt.axhline(y=0, color="g", linestyle="-", alpha=0.3)

    # Mark out-of-control points
    if out_of_control_points:
        for point in out_of_control_points:
            if S_plus[point] > h:
                plt.plot(
                    point + 1,
                    S_plus[point],
                    "ro",
                    markersize=8,
                    label="Out of Control" if point == out_of_control_points[0] else "",
                )
            if S_minus[point] > h:
                plt.plot(
                    point + 1,
                    S_minus[point],
                    "ro",
                    markersize=8,
                    label="Out of Control" if point == out_of_control_points[0] else "",
                )

    # Customize plot
    plt.title(title, pad=20, fontsize=12)
    plt.xlabel("Sample Number", fontsize=10)
    plt.ylabel("CUSUM Statistic", fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.ylim(-50, 50)  # Set reasonable y-axis limits

    # Show plot
    plt.tight_layout()
    plt.show()


# Main execution
if __name__ == "__main__":
    # Read data
    df = pd.read_csv("filtered_washer_info.csv")
    data = df["Diagonal (px)"].values

    # Calculate mean of first 10 points and use it as target
    target_mean = calculate_mean_first_n(data, 10)

    # Print first 10 points and their mean
    print("First 10 data points:")
    for i, value in enumerate(data[:10]):
        print(f"Point {i+1}: {value:.3f}")
    print(f"\nMean of first 10 points (target mean): {target_mean:.3f}")

    # Calculate standard deviation using the first 10 points
    sigma = np.std(data[:10])

    # CUSUM parameters
    k = 0.5 * sigma  # Reference value
    h = 3 * sigma  # Decision interval

    print(f"\nProcess Parameters:")
    print(f"Target mean (μ₀): {target_mean:.3f}")
    print(f"Standard deviation (σ): {sigma:.3f}")
    print(f"Reference value (K): {k:.3f}")
    print(f"Decision interval (H): {h:.3f}")

    # Calculate CUSUM
    S_plus, S_minus, out_of_control_points = calculate_cusum(
        data, target_mean, sigma, k, h
    )

    # Plot results
    title = f"CUSUM Control Chart\nTarget μ₀={target_mean:.2f}, σ={sigma:.2f}, K={k:.2f}, H={h:.2f}"
    plot_cusum_chart(data, S_plus, S_minus, h, out_of_control_points, title)

    # Print results
    if out_of_control_points:
        print(
            "\nOut-of-control points detected at samples:",
            [x + 1 for x in out_of_control_points],
        )
    else:
        print("\nNo out-of-control points detected.")

    # Print CUSUM statistics
    print(f"\nCUSUM Statistics:")
    print(f"Maximum CUSUM⁺: {max(S_plus):.3f}")
    print(f"Maximum CUSUM⁻: {max(S_minus):.3f}")

    # Print differences from target mean
    print("\nDifferences from target mean:")
    for i, value in enumerate(data[:10]):
        diff = value - target_mean
        print(f"Point {i+1} difference: {diff:.3f}")
