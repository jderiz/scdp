import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.data import chemical_symbols

plt.rcParams["xtick.labelsize"] = 18
plt.rcParams["ytick.labelsize"] = 18


def plot_coefficient_stability(
    relative_std,
    title="Coefficient Stability",
    filename="coefficient_stability_analysis.png",
):
    """
    Create a heatmap visualization of coefficient stability.

    Args:
        relative_std: Tensor of shape [num_atoms, num_coeffs] with relative standard deviations
        title: Title for the plot
        filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 8))
    plt.imshow(relative_std.numpy(), aspect="auto", cmap="viridis")
    plt.colorbar(label="Relative Standard Deviation")
    plt.xlabel("Coefficient Index", fontsize=16)
    plt.ylabel("Atom Index", fontsize=16)
    plt.title(title, fontsize=18)
    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nStability visualization saved as '{filename}'")


def plot_isclose_analysis(
    close_mask,
    atom_types,
    coeff_to_l_mapping,
    l_value_stats,
    title_prefix="IsClose Analysis",
    filename="isclose_analysis.png",
):
    """
    Create visualizations for isClose analysis between two sets of coefficients.

    Args:
        close_mask: Boolean tensor of shape [num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        l_value_stats: Dictionary with statistics by L value
        title_prefix: Prefix for plot titles
        filename: Filename to save the plot
    """
    plt.figure(figsize=(20, 16))

    # 1. Heatmap of close coefficients (boolean mask)
    plt.subplot(2, 2, 1)
    plt.imshow(close_mask.numpy(), aspect="auto", cmap="Blues")
    cbar = plt.colorbar(ticks=[0, 1])
    cbar.ax.set_yticklabels(["Different", "Close"], fontsize=18)
    plt.xlabel("Coefficient Index", fontsize=20)
    plt.ylabel("Atom Index", fontsize=20)
    plt.title(f"{title_prefix}: Coefficients that Remain Close", fontsize=22)

    # 2. Bar chart showing percentage of close coefficients by atom
    plt.subplot(2, 2, 2)
    close_percentages = (
        close_mask.sum(dim=1).float() / close_mask.shape[1] * 100
    ).numpy()
    atom_types_str = [
        f"{i} (Z={atom_types[i].item()}->{chemical_symbols[atom_types[i].item()]})"
        for i in range(len(atom_types))
    ]

    # Sort by percentage
    sorted_indices = np.argsort(close_percentages)
    sorted_percentages = close_percentages[sorted_indices]
    sorted_labels = [atom_types_str[i] for i in sorted_indices]

    plt.barh(range(len(sorted_percentages)), sorted_percentages, color="royalblue")
    plt.yticks(range(len(sorted_percentages)), sorted_labels)
    plt.xlabel("Percentage of Coefficients that Remain Close (%)", fontsize=20)
    plt.ylabel("Atom Index (Z=atom type)", fontsize=20)
    plt.title("Percentage of Close Coefficients by Atom", fontsize=22)
    plt.axvline(
        x=np.mean(close_percentages),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(close_percentages):.1f}%",
    )
    plt.legend(fontsize=16)
    plt.grid(axis="x", linestyle="--", alpha=0.7)

    # 3. Bar chart showing percentage of close coefficients by L value
    plt.subplot(2, 2, 3)
    l_values = sorted(l_value_stats.keys())
    # Calculate percentage of coefficients that remain close for each L value
    l_percentages = [
        (
            100 * l_value_stats[l]["close"] / l_value_stats[l]["total"]
            if l_value_stats[l]["total"] > 0
            else 0
        )
        for l in l_values
    ]

    plt.bar(l_values, l_percentages, color="royalblue")
    plt.xlabel("Angular Momentum (L)", fontsize=20)
    plt.ylabel("Percentage of Close Coefficients (%)", fontsize=20)
    plt.title("Percentage of Close Coefficients by L Value", fontsize=22)
    plt.xticks(l_values)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # 4. Heatmap by L value
    plt.subplot(2, 2, 4)

    # Reorganize data by L value for each atom
    # Create a matrix where rows are atoms and columns are L values
    l_max = max(l_value_stats.keys())
    data = np.zeros((len(atom_types), l_max + 1))

    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        # Count close coefficients for each L value for this atom
        l_counts = {}
        l_totals = {}

        for j, l in enumerate(l_values):
            if j >= close_mask.shape[1]:
                continue

            if l not in l_counts:
                l_counts[l] = 0
                l_totals[l] = 0

            l_totals[l] += 1
            if close_mask[i, j]:
                l_counts[l] += 1

        # Calculate percentage for each L value
        for l in range(l_max + 1):
            if l in l_counts and l_totals[l] > 0:
                data[i, l] = 100 * l_counts[l] / l_totals[l]

    im = plt.imshow(data, aspect="auto", cmap="viridis", vmin=0, vmax=100)
    plt.colorbar(im, label="Percentage of Close Coefficients (%)")
    plt.xlabel("Angular Momentum (L)", fontsize=20)
    plt.ylabel("Atom Index", fontsize=20)
    plt.title("Close Coefficient Percentage by Atom and L Value", fontsize=22)
    plt.xticks(range(l_max + 1))

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n{title_prefix} visualization saved as '{filename}'")


def plot_transformation_comparison(
    agreement_percentages,
    mean_rel_diffs,
    max_rel_diffs,
    labels,
    title_prefix="Transformation Comparison",
    filename="transformation_comparison.png",
):
    """
    Plot comparison between theoretical predictions and actual coefficients.

    Args:
        agreement_percentages: List of agreement percentages for each transformation
        mean_rel_diffs: List of mean relative differences
        max_rel_diffs: List of max relative differences
        labels: List of labels for the x-axis
        title_prefix: Prefix for plot titles
        filename: Filename to save the plot
    """
    plt.figure(figsize=(15, 12))

    # 1. Agreement percentages
    plt.subplot(2, 1, 1)
    x = np.arange(len(labels))
    plt.bar(x, agreement_percentages, color="royalblue")
    plt.axhline(
        y=np.mean(agreement_percentages),
        color="red",
        linestyle="--",
        label=f"Mean: {np.mean(agreement_percentages):.1f}%",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.xlabel("Transformation", fontsize=16)
    plt.ylabel("Agreement (%)", fontsize=16)
    plt.title(
        f"{title_prefix}: Percentage of Coefficients where Theory Matches Actual",
        fontsize=18,
    )
    plt.legend()
    plt.tight_layout()

    # 2. Relative differences
    plt.subplot(2, 1, 2)
    width = 0.35
    plt.bar(
        x - width / 2,
        mean_rel_diffs,
        width,
        label="Mean Relative Difference",
        color="royalblue",
    )
    plt.bar(
        x + width / 2,
        max_rel_diffs,
        width,
        label="Max Relative Difference",
        color="red",
    )
    plt.xticks(x, labels, rotation=45, ha="right")
    plt.yscale("log")
    plt.xlabel("Transformation", fontsize=16)
    plt.ylabel("Relative Difference (log scale)", fontsize=16)
    plt.title(
        f"{title_prefix}: Differences Between Theoretical and Actual Coefficients",
        fontsize=18,
    )
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\n{title_prefix} visualization saved as '{filename}'")


def plot_l_value_analysis(
    l_values, accuracy_by_l, title="L Value Analysis", filename="l_value_analysis.png"
):
    """
    Create a bar chart showing analysis by L value.

    Args:
        l_values: List of L values
        accuracy_by_l: List of accuracy percentages for each L value
        title: Title for the plot
        filename: Filename to save the plot
    """
    plt.figure(figsize=(10, 6))
    plt.bar(l_values, accuracy_by_l, color="royalblue")
    plt.xlabel("Angular Momentum (L)", fontsize=16)
    plt.ylabel("Agreement (%)", fontsize=16)
    plt.title(title, fontsize=18)
    plt.xticks(l_values)
    plt.ylim(0, 100)
    plt.grid(axis="y", linestyle="--", alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nL value analysis saved as '{filename}'")


def analyze_coefficient_stability(
    all_coeffs, atom_types, coeff_to_l_mapping, invariance_threshold=0.01
):
    """
    Analyze how coefficients change under transformation and identify invariant coefficients.

    Args:
        all_coeffs: Tensor of shape [num_transformations, num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        invariance_threshold: Threshold for considering a coefficient invariant

    Returns:
        Tuple of (relative_std, invariant_mask)
    """
    num_transformations, num_atoms, num_coeffs = all_coeffs.shape
    print(f"num_transformations: {num_transformations}")
    print(f"num_atoms: {num_atoms}")
    print(f"num_coeffs: {num_coeffs}")

    # Calculate standard deviation for each coefficient across transformations
    coeff_std = torch.std(all_coeffs, dim=0)  # [num_atoms, num_coeffs]

    # Calculate mean absolute value of coefficients
    coeff_abs_mean = torch.mean(torch.abs(all_coeffs), dim=0)  # [num_atoms, num_coeffs]

    # Calculate relative standard deviation (coefficient of variation)
    epsilon = 1e-10  # Small constant to prevent division by zero
    relative_std = coeff_std / (coeff_abs_mean + epsilon)

    # Consider a coefficient "invariant" if its relative standard deviation is below threshold
    invariant_mask = relative_std < invariance_threshold

    # For each atom, count how many coefficients are invariant
    invariant_counts = invariant_mask.sum(dim=1)

    print("\nCoefficient Stability Analysis:")
    print("================================")

    # Per-atom analysis
    for i in range(num_atoms):
        atom_type = chemical_symbols[atom_types[i].item()]
        count = invariant_counts[i].item()
        percentage = 100 * count / num_coeffs
        print(
            f"Atom {i} (Z={atom_types[i].item()}->{atom_type}): {count}/{num_coeffs} ({percentage:.2f}%) coefficients are invariant"
        )

    # Overall statistics
    total_invariant = invariant_mask.sum().item()
    total_coeffs = num_atoms * num_coeffs
    print(
        f"\nOverall: {total_invariant}/{total_coeffs} ({100*total_invariant/total_coeffs:.2f}%) coefficients are invariant"
    )

    # Group coefficients by atom type
    atom_types_unique = torch.unique(atom_types)
    for atom_type in atom_types_unique:
        mask = atom_types == atom_type
        type_invariant = invariant_mask[mask].sum().item()
        type_total = mask.sum().item() * num_coeffs
        print(
            f"Atom type Z={atom_type.item()}: {type_invariant}/{type_total} ({100*type_invariant/type_total:.2f}%) coefficients are invariant"
        )

    # Analysis by angular momentum
    print("\nAnalysis by Angular Momentum (L):")
    print("================================")

    # Collect statistics per L value
    l_value_stats = {}

    for i in range(num_atoms):
        atom_type = atom_types[i].item()

        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        # Create or append to the statistics for each L value
        for j, l in enumerate(l_values):
            if j >= invariant_mask.shape[1]:
                continue

            if l not in l_value_stats:
                l_value_stats[l] = {"total": 0, "invariant": 0}

            l_value_stats[l]["total"] += 1
            if invariant_mask[i, j]:
                l_value_stats[l]["invariant"] += 1

    # Print statistics for each L value
    for l, stats in sorted(l_value_stats.items()):
        percentage = (
            100 * stats["invariant"] / stats["total"] if stats["total"] > 0 else 0
        )
        print(
            f"L={l}: {stats['invariant']}/{stats['total']} ({percentage:.2f}%) coefficients are invariant"
        )

    return relative_std, invariant_mask, l_value_stats


def compare_coefficient_pairs(
    coeffs_a, coeffs_b, title_prefix="Coefficient Comparison"
):
    """
    Compare two sets of coefficients using isclose and relative differences.

    Args:
        coeffs_a: First set of coefficients [num_atoms, num_coeffs]
        coeffs_b: Second set of coefficients [num_atoms, num_coeffs]
        title_prefix: Prefix for output messages

    Returns:
        Tuple of (close_mask, rel_diff)
    """
    # Use torch.isclose() approach
    close_mask = torch.isclose(coeffs_a, coeffs_b, atol=1e-2, rtol=1e-2)

    # Count how many coefficients are close for the same atom
    close_counts = close_mask.sum(dim=1)
    total_counts = close_mask.shape[1]

    print(f"\n{title_prefix} (using isclose):")
    print("=" * (len(title_prefix) + 16))

    # Per-atom analysis
    for i in range(len(close_counts)):
        count = close_counts[i].item()
        percentage = 100 * count / total_counts
        print(
            f"Atom {i}: {count}/{total_counts} ({percentage:.2f}%) coefficients are close"
        )

    # Overall statistics
    total_close = close_mask.sum().item()
    total_coeffs = close_mask.numel()
    print(
        f"\nOverall: {total_close}/{total_coeffs} ({100*total_close/total_coeffs:.2f}%) coefficients are close"
    )

    # Calculate absolute difference
    abs_diff = torch.abs(coeffs_a - coeffs_b)

    # Calculate relative difference
    # Add small epsilon to prevent division by zero
    epsilon = 1e-10
    rel_diff = abs_diff / (torch.abs(coeffs_b) + epsilon)

    # Print summary statistics
    print(f"\n{title_prefix} (difference statistics):")
    print(f"Mean absolute difference: {abs_diff.mean().item():.4e}")
    print(f"Max absolute difference: {abs_diff.max().item():.4e}")
    print(f"Mean relative difference: {rel_diff.mean().item():.4f}")
    print(f"Max relative difference: {rel_diff.max().item():.4f}")

    return close_mask, rel_diff


def analyze_by_l_value(close_mask, atom_types, coeff_to_l_mapping):
    """
    Analyze comparison results by L value.

    Args:
        close_mask: Boolean tensor of shape [num_atoms, num_coefficients]
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values

    Returns:
        Dictionary with statistics by L value
    """
    # Analysis by L value
    l_value_stats = {}

    for i in range(len(atom_types)):
        atom_type = atom_types[i].item()
        if atom_type not in coeff_to_l_mapping:
            continue

        l_values = coeff_to_l_mapping[atom_type]

        for j, l in enumerate(l_values):
            if j >= close_mask.shape[1]:
                continue

            if l not in l_value_stats:
                l_value_stats[l] = {"total": 0, "close": 0}

            l_value_stats[l]["total"] += 1
            if close_mask[i, j]:
                l_value_stats[l]["close"] += 1

    # Print statistics for each L value
    print("\nAnalysis by Angular Momentum (L):")
    print("================================")
    for l, stats in sorted(l_value_stats.items()):
        percentage = 100 * stats["close"] / stats["total"] if stats["total"] > 0 else 0
        print(
            f"L={l}: {stats['close']}/{stats['total']} ({percentage:.2f}%) coefficients are close"
        )

    return l_value_stats
