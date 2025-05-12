import matplotlib.pyplot as plt
import numpy as np
import torch
from ase.data import chemical_symbols
from e3nn import o3
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scdp.scripts.predict_embedding_transform_rotation import \
    predict_transformed_embedding_rotation

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


def plot_pairwise_coefficient_heatmap(
    ref_coeffs,
    target_coeffs,
    standard_error=None,
    title="Coefficient Comparison Heatmap",
    filename="coefficient_heatmap.png",
    vmin=None,
    vmax=None,
    cmap="viridis",
):
    """
    Create a heatmap visualization comparing reference coefficients with target coefficients.

    Args:
        ref_coeffs: Reference coefficients tensor of shape [num_atoms, num_coeffs]
        target_coeffs: Target coefficients tensor of shape [num_atoms, num_coeffs]
        standard_error: Optional tensor of standard errors with same shape as coefficients
        title: Title for the plot
        filename: Filename to save the plot
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        cmap: Colormap to use
    """
    # Calculate absolute difference between reference and target
    abs_diff = torch.abs(ref_coeffs - target_coeffs)

    # Calculate relative difference (with small epsilon to prevent division by zero)
    epsilon = 1e-12
    rel_diff = abs_diff / (torch.abs(ref_coeffs) + epsilon)

    plt.figure(figsize=(14, 10))

    # Plot the main heatmap of relative differences
    plt.subplot(1, 2, 1)
    im = plt.imshow(rel_diff.numpy(), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    plt.colorbar(im, label="Relative Difference")
    plt.xlabel("Coefficient Index", fontsize=16)
    plt.ylabel("Atom Index", fontsize=16)
    plt.title(f"{title} - Relative Difference", fontsize=18)

    # Add summary statistics as a text box
    stats_text = (
        f"Mean Abs Diff: {abs_diff.mean().item():.2e}\n"
        f"Max Abs Diff: {abs_diff.max().item():.2e}\n"
        f"Mean Rel Diff: {rel_diff.mean().item():.2e}\n"
        f"Max Rel Diff: {rel_diff.max().item():.2e}\n"
    )
    plt.annotate(
        stats_text,
        xy=(0.02, 0.02),
        xycoords="axes fraction",
        fontsize=12,
        backgroundcolor="white",
        bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
    )

    # If standard error is provided, plot it as a second heatmap
    if standard_error is not None:
        plt.subplot(1, 2, 2)
        im2 = plt.imshow(standard_error.numpy(), aspect="auto", cmap="plasma")
        plt.colorbar(im2, label="Standard Error")
        plt.xlabel("Coefficient Index", fontsize=16)
        plt.ylabel("Atom Index", fontsize=16)
        plt.title(f"{title} - Standard Error", fontsize=18)

        # Add summary statistics for standard error
        se_stats = (
            f"Mean SE: {standard_error.mean().item():.2e}\n"
            f"Max SE: {standard_error.max().item():.2e}\n"
        )
        plt.annotate(
            se_stats,
            xy=(0.02, 0.02),
            xycoords="axes fraction",
            fontsize=12,
            backgroundcolor="white",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8),
        )

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nCoefficient comparison heatmap saved as '{filename}'")


def plot_transformation_coefficient_comparison(
    reference_coeffs,
    all_transformed_coeffs,
    transformation_names,
    title_prefix="Transformation vs Original",
    filename="transformation_coefficient_comparison.png",
):
    """
    Create a grid of heatmaps comparing original coefficients with multiple transformations.

    Args:
        reference_coeffs: Original/reference coefficients tensor [num_atoms, num_coeffs]
        all_transformed_coeffs: Tensor of transformed coefficients [num_transforms, num_atoms, num_coeffs]
        transformation_names: List of names for each transformation
        title_prefix: Prefix for the plot title
        filename: Filename to save the plot
    """
    num_transforms = all_transformed_coeffs.shape[0]

    # Calculate number of rows and columns for subplot grid
    grid_size = int(np.ceil(np.sqrt(num_transforms)))
    rows = grid_size
    cols = grid_size

    # Adjust if we have fewer transformations than grid cells
    if num_transforms <= grid_size * (grid_size - 1):
        rows = grid_size - 1

    plt.figure(figsize=(cols * 4, rows * 3.5))

    # Find global min/max for consistent colormap scaling
    all_rel_diffs = []
    epsilon = 1e-12

    for i in range(num_transforms):
        trans_coeffs = all_transformed_coeffs[i]
        abs_diff = torch.abs(reference_coeffs - trans_coeffs)
        rel_diff = abs_diff / (torch.abs(reference_coeffs) + epsilon)
        all_rel_diffs.append(rel_diff)

    all_rel_diffs_tensor = torch.stack(all_rel_diffs)
    vmin = 0
    vmax = torch.quantile(
        all_rel_diffs_tensor, 0.95
    ).item()  # Use 95th percentile as max to avoid outliers

    # Create a subplot for each transformation
    for i in range(num_transforms):
        plt.subplot(rows, cols, i + 1)

        trans_coeffs = all_transformed_coeffs[i]
        abs_diff = torch.abs(reference_coeffs - trans_coeffs)
        rel_diff = abs_diff / (torch.abs(reference_coeffs) + epsilon)

        # Calculate match percentage (relative difference < threshold)
        threshold = 0.01  # 1% relative difference threshold
        match_percentage = 100 * (rel_diff < threshold).sum().item() / rel_diff.numel()

        im = plt.imshow(
            rel_diff.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )

        # Add transformation name and match percentage to title
        trans_name = (
            transformation_names[i]
            if i < len(transformation_names)
            else f"Transform {i}"
        )
        plt.title(f"{trans_name}\nMatch: {match_percentage:.1f}%", fontsize=12)

        # Only add axes labels on left and bottom edges
        if i % cols == 0:
            plt.ylabel("Atom Index", fontsize=12)
        if i >= (rows - 1) * cols:
            plt.xlabel("Coefficient Index", fontsize=12)

    # Add a color bar for the whole figure
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Relative Difference", fontsize=14)

    plt.suptitle(title_prefix, fontsize=20)
    plt.subplots_adjust(top=0.9)

    plt.savefig(filename)
    print(f"\nTransformation coefficient comparison saved as '{filename}'")


def plot_theoretical_vs_actual_comparison(
    theoretical_coeffs,
    actual_coeffs,
    transformation_names,
    title_prefix="Theoretical vs Actual",
    filename="theoretical_vs_actual_comparison.png",
):
    """
    Create a grid of heatmaps comparing theoretical coefficient predictions with actual model outputs.

    Args:
        theoretical_coeffs: Tensor of theoretical coefficients [num_transforms, num_atoms, num_coeffs]
        actual_coeffs: Tensor of actual coefficients [num_transforms, num_atoms, num_coeffs]
        transformation_names: List of names for each transformation
        title_prefix: Prefix for the plot title
        filename: Filename to save the plot
    """
    num_transforms = theoretical_coeffs.shape[0]

    # Calculate number of rows and columns for subplot grid
    grid_size = int(np.ceil(np.sqrt(num_transforms)))
    rows = grid_size
    cols = grid_size

    # Adjust if we have fewer transformations than grid cells
    if num_transforms <= grid_size * (grid_size - 1):
        rows = grid_size - 1

    plt.figure(figsize=(cols * 4, rows * 3.5))

    # Find global min/max for consistent colormap scaling
    all_rel_diffs = []
    epsilon = 1e-12

    for i in range(num_transforms):
        theo_coeffs = theoretical_coeffs[i]
        act_coeffs = actual_coeffs[i]
        abs_diff = torch.abs(theo_coeffs - act_coeffs)
        rel_diff = abs_diff / (torch.abs(theo_coeffs) + epsilon)
        all_rel_diffs.append(rel_diff)

    all_rel_diffs_tensor = torch.stack(all_rel_diffs)
    vmin = 0
    vmax = torch.quantile(
        all_rel_diffs_tensor, 0.95
    ).item()  # Use 95th percentile as max to avoid outliers

    # Create a subplot for each transformation
    for i in range(num_transforms):
        plt.subplot(rows, cols, i + 1)

        theo_coeffs = theoretical_coeffs[i]
        act_coeffs = actual_coeffs[i]
        abs_diff = torch.abs(theo_coeffs - act_coeffs)
        rel_diff = abs_diff / (torch.abs(theo_coeffs) + epsilon)

        # Calculate standard error
        stderr = abs_diff.std() / np.sqrt(abs_diff.numel())

        # Calculate match percentage (relative difference < threshold)
        threshold = 0.01  # 1% relative difference threshold
        match_percentage = 100 * (rel_diff < threshold).sum().item() / rel_diff.numel()

        im = plt.imshow(
            rel_diff.numpy(), aspect="auto", cmap="viridis", vmin=vmin, vmax=vmax
        )

        # Add transformation name and match percentage to title
        trans_name = (
            transformation_names[i]
            if i < len(transformation_names)
            else f"Transform {i}"
        )
        plt.title(
            f"{trans_name}\nMatch: {match_percentage:.1f}%, SE: {stderr:.2e}",
            fontsize=12,
        )

        # Only add axes labels on left and bottom edges
        if i % cols == 0:
            plt.ylabel("Atom Index", fontsize=12)
        if i >= (rows - 1) * cols:
            plt.xlabel("Coefficient Index", fontsize=12)

    # Add a color bar for the whole figure
    plt.tight_layout()
    plt.subplots_adjust(right=0.9)
    cbar_ax = plt.axes([0.92, 0.15, 0.02, 0.7])
    cb = plt.colorbar(im, cax=cbar_ax)
    cb.set_label("Relative Difference", fontsize=14)

    plt.suptitle(title_prefix, fontsize=20)
    plt.subplots_adjust(top=0.9)

    plt.savefig(filename)
    print(f"\nTheoretical vs actual comparison saved as '{filename}'")


def plot_coeffs_vs_coeffs_by_atom(
    coeffs_a,
    coeffs_b,
    atom_types=None,
    title="Coefficient Comparison",
    filename="coeffs_vs_coeffs_by_atom.png",
    max_atoms_to_plot=9,
    marker_size=3,
):
    """
    Create a grid of scatter plots comparing coefficients between two sets (e.g., original vs transformed),
    with one plot per atom.

    Args:
        coeffs_a: First set of coefficients [num_atoms, num_coeffs]
        coeffs_b: Second set of coefficients [num_atoms, num_coeffs]
        atom_types: Optional tensor of atom types for labeling
        title: Title for the plot
        filename: Filename to save the plot
        max_atoms_to_plot: Maximum number of atoms to include in the grid
        marker_size: Size of the scatter plot markers
    """
    num_atoms = coeffs_a.shape[0]

    # Limit the number of atoms to plot
    num_atoms_to_plot = min(num_atoms, max_atoms_to_plot)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_atoms_to_plot)))

    # Create figure
    plt.figure(figsize=(grid_size * 4, grid_size * 4))

    # Find global min/max for consistent axis limits
    global_min = min(coeffs_a.min().item(), coeffs_b.min().item())
    global_max = max(coeffs_a.max().item(), coeffs_b.max().item())

    # Add some padding to the limits
    range_padding = 0.05 * (global_max - global_min)
    ax_min = global_min - range_padding
    ax_max = global_max + range_padding

    # Create a plot for each atom
    for i in range(num_atoms_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)

        # Extract coefficients for this atom
        atom_coeffs_a = coeffs_a[i].numpy()
        atom_coeffs_b = coeffs_b[i].numpy()

        # Create scatter plot
        plt.scatter(atom_coeffs_a, atom_coeffs_b, s=marker_size, alpha=0.7)

        # Add identity line (y=x)
        plt.plot([ax_min, ax_max], [ax_min, ax_max], "r--", alpha=0.5)

        # Set axis limits
        plt.xlim(ax_min, ax_max)
        plt.ylim(ax_min, ax_max)

        # Add atom label
        if atom_types is not None:
            atom_type = atom_types[i].item()
            atom_symbol = (
                chemical_symbols[atom_type]
                if atom_type < len(chemical_symbols)
                else f"Z{atom_type}"
            )
            plt.title(f"Atom {i} ({atom_symbol})")
        else:
            plt.title(f"Atom {i}")

        # Calculate correlation coefficient
        correlation = np.corrcoef(atom_coeffs_a, atom_coeffs_b)[0, 1]

        # Calculate RMSE
        rmse = np.sqrt(np.mean((atom_coeffs_a - atom_coeffs_b) ** 2))

        # Add statistics text
        plt.text(
            0.05,
            0.95,
            f"Corr: {correlation:.3f}\nRMSE: {rmse:.3e}",
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Only show axis labels for bottom and left plots
        if i % grid_size == 0:
            plt.ylabel("Coefficients B")
        if i >= grid_size * (grid_size - 1):
            plt.xlabel("Coefficients A")

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename)
    print(f"\nCoefficient comparison by atom saved as '{filename}'")


def plot_coeffs_by_angular_momentum(
    coeffs,
    atom_types,
    coeff_to_l_mapping,
    title="Coefficients by Angular Momentum",
    filename="coeffs_by_angular_momentum.png",
    cmap="viridis",
    vmin=None,
    vmax=None,
    max_atoms_to_plot=6,
):
    """
    Create a heatmap visualization of coefficients organized by angular momentum (L value).

    Args:
        coeffs: Coefficient tensor of shape [num_atoms, num_coeffs] or list of such tensors
        atom_types: Tensor of atom types
        coeff_to_l_mapping: Dictionary mapping atom types to L values for each coefficient
        title: Title for the plot
        filename: Filename to save the plot
        cmap: Colormap to use
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
        max_atoms_to_plot: Maximum number of atoms to include in the visualization
    """
    # Handle multiple coefficient sets (e.g., original and transformed)
    if isinstance(coeffs, list) or (
        isinstance(coeffs, torch.Tensor) and len(coeffs.shape) == 3
    ):
        multi_coeffs = True
        if isinstance(coeffs, torch.Tensor):
            num_sets = coeffs.shape[0]
            # For tensor with shape [num_sets, num_atoms, num_coeffs]
            coeffs_list = [
                coeffs[i] for i in range(min(2, num_sets))
            ]  # Limit to 2 sets for comparison
        else:
            coeffs_list = coeffs[
                : min(2, len(coeffs))
            ]  # Limit to 2 sets for comparison
        num_atoms = coeffs_list[0].shape[0]
    else:
        multi_coeffs = False
        coeffs_list = [coeffs]
        num_atoms = coeffs.shape[0]

    # Limit the number of atoms to plot
    num_atoms_to_plot = min(num_atoms, max_atoms_to_plot)

    # Determine max L value across all atom types
    max_l = 0
    for atom_type in atom_types[:num_atoms_to_plot]:
        atom_type = atom_type.item()
        if atom_type in coeff_to_l_mapping:
            l_values = coeff_to_l_mapping[atom_type]
            max_l = max(max_l, max(l_values) if l_values else 0)

    # Calculate figure size based on number of atoms
    if multi_coeffs:
        # For comparison plots, put them side by side
        fig, axes = plt.subplots(
            num_atoms_to_plot, 2, figsize=(12, 3 * num_atoms_to_plot), squeeze=False
        )
        set_names = ["Set A", "Set B"]
    else:
        # For single set, just show those plots
        fig, axes = plt.subplots(
            num_atoms_to_plot, 1, figsize=(8, 3 * num_atoms_to_plot), squeeze=False
        )

    # Create a heatmap for each atom
    for atom_idx in range(num_atoms_to_plot):
        atom_type = atom_types[atom_idx].item()
        atom_symbol = (
            chemical_symbols[atom_type]
            if atom_type < len(chemical_symbols)
            else f"Z{atom_type}"
        )

        for set_idx, coeffs_set in enumerate(coeffs_list):
            if multi_coeffs:
                ax = axes[atom_idx, set_idx]
            else:
                ax = axes[atom_idx, 0]

            # Get L values for this atom type
            if atom_type in coeff_to_l_mapping:
                l_values = coeff_to_l_mapping[atom_type]

                # Create a matrix where rows are L values (0 to max_l)
                # and columns are coefficients within each L value
                l_value_counts = {}
                for l in range(max_l + 1):
                    l_value_counts[l] = l_values.count(l)

                # Find the maximum number of coefficients for any L value
                max_coeffs_per_l = max(l_value_counts.values()) if l_value_counts else 0

                # Create an array to hold coefficient values by L
                # Initialize with NaN so missing values don't affect the colormap
                data = np.full((max_l + 1, max_coeffs_per_l), np.nan)

                # Fill in the coefficient values
                for coeff_idx, l in enumerate(l_values):
                    if (
                        coeff_idx < coeffs_set.shape[1]
                    ):  # Check if coeff_idx is within range
                        # Find position for this coefficient within its L row
                        l_position = 0
                        for i in range(coeff_idx):
                            if l_values[i] == l:
                                l_position += 1

                        # Populate the data array
                        if l_position < max_coeffs_per_l:
                            data[l, l_position] = coeffs_set[atom_idx, coeff_idx].item()

                # Create the heatmap
                im = ax.imshow(data, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

                # Add a colorbar
                divider = make_axes_locatable(ax)
                cax = divider.append_axes("right", size="5%", pad=0.05)
                fig.colorbar(im, cax=cax, orientation="vertical")

                # Label the y-axis with L values
                ax.set_yticks(np.arange(max_l + 1))
                ax.set_yticklabels([f"L={l}" for l in range(max_l + 1)])

                # Label the x-axis
                ax.set_xlabel("Coefficient Index within L value")

                # Add title
                if multi_coeffs:
                    ax.set_title(
                        f"Atom {atom_idx} ({atom_symbol}) - {set_names[set_idx]}"
                    )
                else:
                    ax.set_title(f"Atom {atom_idx} ({atom_symbol})")

                # Add coefficient count text for each L value
                for l, count in l_value_counts.items():
                    if count > 0:
                        ax.text(
                            -0.5,
                            l,
                            f"({count})",
                            verticalalignment="center",
                            fontsize=8,
                        )

            else:
                # If no L mapping, just display a placeholder
                ax.text(
                    0.5,
                    0.5,
                    f"No L mapping for atom type {atom_type}",
                    horizontalalignment="center",
                    verticalalignment="center",
                    transform=ax.transAxes,
                )
                ax.set_xticks([])
                ax.set_yticks([])

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename)
    print(f"\nCoefficients by angular momentum saved as '{filename}'")


def plot_atom_coefficient_comparison_heatmap(
    all_coeffs,
    atom_types=None,
    atom_indices=None,
    atom_type_filter=None,
    transformation_names=None,
    title="Coefficient Comparison for Selected Atoms",
    filename="atom_coeff_comparison_heatmap.png",
    cmap="viridis",
    vmin=None,
    vmax=None,
):
    """
    Create a heatmap comparing coefficient values across different transformations
    for a specific atom or atoms of a specific type.

    Args:
        all_coeffs: Tensor of shape [num_transformations, num_atoms, num_coeffs] with all coefficients
        atom_types: Tensor of atom types
        atom_indices: List of specific atom indices to include (e.g., [0, 2, 5])
                     If None, will use atom_type_filter instead
        atom_type_filter: Atomic number to filter by (e.g., 6 for Carbon)
                          Only used if atom_indices is None
        transformation_names: List of names for each transformation
        title: Title for the plot
        filename: Filename to save the plot
        cmap: Colormap to use
        vmin: Minimum value for colormap scaling
        vmax: Maximum value for colormap scaling
    """
    if isinstance(all_coeffs, list):
        all_coeffs = torch.stack(all_coeffs)

    # Get dimensions
    num_transformations, num_atoms, num_coeffs = all_coeffs.shape

    # Determine which atoms to include
    if atom_indices is not None:
        selected_indices = atom_indices
    elif atom_type_filter is not None and atom_types is not None:
        # Filter atoms by type
        selected_indices = [
            i for i in range(num_atoms) if atom_types[i].item() == atom_type_filter
        ]
    else:
        # Default to first atom if no filter specified
        selected_indices = [0]

    if not selected_indices:
        print(f"No atoms found matching the filter criteria!")
        return

    num_selected_atoms = len(selected_indices)

    # Create transformation names if not provided
    if transformation_names is None:
        transformation_names = [f"Transform {i}" for i in range(num_transformations)]

    # Create one plot per selected atom
    for idx, atom_idx in enumerate(selected_indices):
        # Create a coefficient vs coefficient grid for each pair of transformations
        plt.figure(figsize=(12, 10))

        # Create a grid of num_transforms x num_transforms subplots
        grid_size = num_transformations

        # For atom label
        if atom_types is not None:
            atom_type = atom_types[atom_idx].item()
            atom_symbol = (
                chemical_symbols[atom_type]
                if atom_type < len(chemical_symbols)
                else f"Z{atom_type}"
            )
            atom_label = f"Atom {atom_idx} ({atom_symbol})"
        else:
            atom_label = f"Atom {atom_idx}"

        # Create 2D array for the grid
        # Each cell (i,j) will contain coefficient_i vs coefficient_j correlation
        grid_data = np.zeros((num_coeffs, num_coeffs))

        # For a single atom, let's compare each coefficient across all transformations
        for i in range(num_coeffs):
            for j in range(num_coeffs):
                # Calculate correlation between coefficient i and j across all transformations
                coeff_i_values = all_coeffs[:, atom_idx, i].numpy()
                coeff_j_values = all_coeffs[:, atom_idx, j].numpy()

                # Calculate correlation or another metric
                # Using covariance for this heatmap
                grid_data[i, j] = np.cov(coeff_i_values, coeff_j_values)[0, 1]

        # Find global min/max for consistent color scaling
        if vmin is None:
            vmin = np.nanmin(grid_data)
        if vmax is None:
            vmax = np.nanmax(grid_data)

        # Create the heatmap
        plt.imshow(grid_data, cmap=cmap, vmin=vmin, vmax=vmax)
        plt.colorbar(label="Covariance")

        # Add labels
        plt.xlabel("Coefficient Index", fontsize=14)
        plt.ylabel("Coefficient Index", fontsize=14)
        plt.title(f"{title}\n{atom_label}", fontsize=16)

        # Add grid lines
        plt.grid(False)

        # Save the plot
        atom_filename = filename.replace(".png", f"_atom{atom_idx}.png")
        plt.tight_layout()
        plt.savefig(atom_filename)
        print(f"\nAtom coefficient comparison heatmap saved as '{atom_filename}'")

        # Now create a second plot showing coefficient values across transformations
        plt.figure(figsize=(12, 8))

        # Extract coefficient data for this atom across all transformations
        atom_data = all_coeffs[:, atom_idx, :].numpy()

        # Create a heatmap with transformations on y-axis and coefficients on x-axis
        im = plt.imshow(atom_data, aspect="auto", cmap="plasma")
        plt.colorbar(im, label="Coefficient Value")

        # Add labels
        plt.xlabel("Coefficient Index", fontsize=14)
        plt.ylabel("Transformation", fontsize=14)
        plt.title(
            f"Coefficient Values Across Transformations\n{atom_label}", fontsize=16
        )

        # Set y-ticks to transformation names
        plt.yticks(range(num_transformations), transformation_names)

        # Save this plot too
        values_filename = filename.replace(".png", f"_atom{atom_idx}_values.png")
        plt.tight_layout()
        plt.savefig(values_filename)
        print(
            f"\nCoefficient values across transformations saved as '{values_filename}'"
        )
        plt.close()


def plot_embedding_comparison(
    model, 
    all_embeddings,
    atom_types=None,
    transformation_names=None,
    title="Embedding Comparison Across Rotations",
    filename="embedding_comparison.png",
    max_atoms_to_plot=6,
    marker_size=3,
    lmax_list=None,
    mmax_list=None,
    rotation_matrices=None,
):
    """
    Create a grid of scatter plots comparing embeddings between different rotations,
    with one plot per atom. Each point represents a position in the embedding vector.
    Applies Wigner-D transformation to the spherical harmonic embeddings before comparison.

    Args:
        all_embeddings: List of embedding tensors, each of shape [num_transformations, num_atoms, num_channels, num_coeffs]
        atom_types: Optional tensor of atom types for labeling
        transformation_names: List of names for each transformation/rotation
        title: Title for the plot
        filename: Filename to save the plot
        max_atoms_to_plot: Maximum number of atoms to include in the grid
        marker_size: Size of the scatter plot markers
        lmax_list: List of maximum degrees for each resolution
        mmax_list: List of maximum orders for each resolution
        rotation_matrices: List of rotation matrices for each transformation
    """
    if isinstance(all_embeddings, list):
        all_embeddings = torch.stack(all_embeddings)

    # Print shape for debugging
    print(f"Embeddings shape: {all_embeddings.shape}")

    num_transformations, num_atoms, num_channels, num_coeffs = all_embeddings.shape

    # Limit the number of atoms to plot
    num_atoms_to_plot = min(num_atoms, max_atoms_to_plot)

    # Calculate grid dimensions
    grid_size = int(np.ceil(np.sqrt(num_atoms_to_plot)))

    # Create figure
    plt.figure(figsize=(grid_size * 4, grid_size * 4))

    # Create transformation names if not provided
    if transformation_names is None:
        transformation_names = [f"Transform {i}" for i in range(num_transformations)]

    # Create a plot for each atom
    for i in range(num_atoms_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)

        # Extract embeddings for this atom across all transformations
        atom_embeddings = all_embeddings[:, i, :, :].numpy()

        # Get the unrotated embeddings (first transformation)
        unrotated_embeddings = atom_embeddings[0]

        # Create scatter plot comparing theoretical vs actual transformations
        for j in range(1, num_transformations):
            if rotation_matrices is not None:
                # Apply Wigner-D transformation to unrotated embeddings
                rotation_matrix = rotation_matrices[j]
                
                # Convert numpy arrays to torch tensors
                unrotated_embeddings_torch = torch.tensor(
                    unrotated_embeddings, dtype=torch.float32
                )
                
                # Use predict_transformed_embedding_rotation directly
                
                # Apply transformation using the dedicated function
                transformed_embeddings = predict_transformed_embedding_rotation(
                    model=model,
                    embedding_original=unrotated_embeddings_torch,
                    rotation_matrix=rotation_matrix,
                    device="cuda"
                )
                
                # Convert back to numpy for plotting
                transformed_embeddings = transformed_embeddings.numpy()

                # Compare theoretical transformation with actual model output
                plt.scatter(
                    transformed_embeddings.flatten(),  # Theoretically transformed embeddings
                    atom_embeddings[j].flatten(),  # Actual model output
                    s=marker_size,
                    alpha=0.7,
                    label=transformation_names[j] if j == 1 else None,
                )
            else:
                # If no rotation matrices provided, just compare the actual embeddings
                plt.scatter(
                    unrotated_embeddings.flatten(),  # Unrotated embeddings
                    atom_embeddings[j].flatten(),  # Current transformation
                    s=marker_size,
                    alpha=0.7,
                    label=transformation_names[j] if j == 1 else None,
                )

        # Find global min/max for consistent axis limits
        global_min = min(atom_embeddings.min(), atom_embeddings.max())
        global_max = max(atom_embeddings.min(), atom_embeddings.max())

        # Add some padding to the limits
        range_padding = 0.05 * (global_max - global_min)
        ax_min = global_min - range_padding
        ax_max = global_max + range_padding

        # Add identity line (y=x)
        plt.plot([ax_min, ax_max], [ax_min, ax_max], "r--", alpha=0.5)

        # Set axis limits
        plt.xlim(ax_min, ax_max)
        plt.ylim(ax_min, ax_max)

        # Add atom label
        if atom_types is not None:
            atom_type = atom_types[i].item()
            atom_symbol = (
                chemical_symbols[atom_type]
                if atom_type < len(chemical_symbols)
                else f"Z{atom_type}"
            )
            plt.title(f"Atom {i} ({atom_symbol})")
        else:
            plt.title(f"Atom {i}")

        # Calculate correlation coefficient with first transformation
        correlations = []
        for j in range(1, num_transformations):
            if rotation_matrices is not None:
                # Compare theoretical transformation with actual model output
                correlation = np.corrcoef(
                    transformed_embeddings.flatten(), atom_embeddings[j].flatten()
                )[0, 1]
            else:
                correlation = np.corrcoef(
                    unrotated_embeddings.flatten(), atom_embeddings[j].flatten()
                )[0, 1]
            correlations.append(correlation)

        # Calculate RMSE with first transformation
        rmses = []
        for j in range(1, num_transformations):
            if rotation_matrices is not None:
                # Compare theoretical transformation with actual model output
                rmse = np.sqrt(
                    np.mean(
                        (
                            transformed_embeddings.flatten()
                            - atom_embeddings[j].flatten()
                        )
                        ** 2
                    )
                )
            else:
                rmse = np.sqrt(
                    np.mean(
                        (unrotated_embeddings.flatten() - atom_embeddings[j].flatten())
                        ** 2
                    )
                )
            rmses.append(rmse)

        # Add statistics text
        stats_text = "Correlations:\n"
        for j, corr in enumerate(correlations):
            stats_text += f"{transformation_names[j+1]}: {corr:.3f}\n"
        stats_text += "\nRMSEs:\n"
        for j, rmse in enumerate(rmses):
            stats_text += f"{transformation_names[j+1]}: {rmse:.3e}"

        plt.text(
            0.05,
            0.95,
            stats_text,
            transform=plt.gca().transAxes,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.7),
        )

        # Only show axis labels for bottom and left plots
        if i % grid_size == 0:
            plt.ylabel("Embedding Values")
        if i >= grid_size * (grid_size - 1):
            plt.xlabel("Embedding Values")

    # Add legend for the first subplot
    plt.subplot(grid_size, grid_size, 1)
    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")

    plt.tight_layout()
    plt.suptitle(title, fontsize=16)
    plt.subplots_adjust(top=0.95)

    plt.savefig(filename, bbox_inches="tight")
    print(f"\nEmbedding comparison saved as '{filename}'")
