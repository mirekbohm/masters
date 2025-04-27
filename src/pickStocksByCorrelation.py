import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime, timedelta
import itertools
import random


def pickStocksByCorrelation(corr_matrix: pd.DataFrame, num_stocks: int, corr_type: str, balance_factor: float = 0.4):
    """
    Pick a given number of stocks from the S&P 500 based on their pairwise correlations.

    Parameters:
        corr_matrix (pd.DataFrame): Correlation matrix with stock tickers as index and columns.
        num_stocks (int): The number of stocks to select.
        corr_type (str): One of 'high', 'mid', 'low', or 'mixed'.
                         - 'high': stocks must have pairwise correlation >= threshold.
                         - 'mid': stocks must have pairwise correlation within a mid-range.
                         - 'low': stocks must have pairwise correlation <= threshold.
                         - 'mixed': stocks are selected to create a balanced mix of high and low correlations.
        balance_factor (float): For 'mixed' type, the target proportion of high correlation pairs (0.0-1.0).
                                Default is 0.4 (40% high correlations, 60% low correlations).

    Returns:
        list: A list of stock tickers meeting the criteria, or an empty list if no valid group is found.
        dict: For 'mixed' type, returns additional statistics about the correlations.
    """
    # Set thresholds and conditions
    if corr_type == 'high':
        threshold = 0.7
        condition = lambda x: x >= threshold
    elif corr_type == 'low':
        threshold = 0.3
        condition = lambda x: x <= threshold
    elif corr_type == 'mid':
        lower_threshold = 0.3
        upper_threshold = 0.7
        condition = lambda x: lower_threshold <= x <= upper_threshold
    elif corr_type == 'mixed':
        low_threshold = 0.3
        high_threshold = 0.7
        # For mixed, every pair between the candidate and group must be at an extreme
        condition = lambda x: x <= low_threshold or x >= high_threshold
    else:
        raise ValueError("corr_type must be 'high', 'mid', 'low', or 'mixed'")

    tickers = list(corr_matrix.columns)

    # Helper function to check if a candidate stock meets the condition with every stock in the current group
    def is_valid_addition(group, candidate):
        for stock in group:
            corr = corr_matrix.loc[stock, candidate]
            if corr_type == 'mixed':
                if not (corr <= low_threshold or corr >= high_threshold):
                    return False
            else:
                if not condition(corr):
                    return False
        return True

    # For mixed correlations, calculate statistics about the correlation distribution
    def get_correlation_stats(group):
        if len(group) < 2:
            return {"high_ratio": 0, "low_ratio": 0, "total_pairs": 0}

        pairs = list(itertools.combinations(group, 2))
        correlations = [corr_matrix.loc[s1, s2] for s1, s2 in pairs]

        high_count = sum(1 for c in correlations if c >= high_threshold)
        low_count = sum(1 for c in correlations if c <= low_threshold)
        mid_count = len(correlations) - high_count - low_count

        return {
            "high_count": high_count,
            "low_count": low_count,
            "mid_count": mid_count,
            "high_ratio": high_count / len(correlations) if correlations else 0,
            "low_ratio": low_count / len(correlations) if correlations else 0,
            "total_pairs": len(correlations)
        }

    # Score a group based on how well it matches our desired balance of correlations
    def score_mixed_group(group):
        stats = get_correlation_stats(group)

        # If we don't have both high and low correlations, give a poor score
        if stats["high_count"] == 0 or stats["low_count"] == 0:
            return -1000

        # Calculate how close the actual high ratio is to our target balance factor
        balance_score = -abs(stats["high_ratio"] - balance_factor) * 10

        # Also favor groups with more extreme correlations (fewer mid-range correlations)
        extremeness_score = -stats["mid_count"] * 2

        return balance_score + extremeness_score

    # For mixed type, we'll use a more sophisticated approach to find balanced groups
    if corr_type == 'mixed':
        # Try multiple random seeds to increase chances of finding a good group
        best_group = []
        best_score = float('-inf')
        best_stats = {}

        # Number of random seeds to try
        num_attempts = min(50, len(tickers))
        seed_tickers = random.sample(tickers, num_attempts)

        for seed in seed_tickers:
            # Start with this seed ticker
            group = [seed]

            # Track candidates that could potentially be added
            valid_candidates = [t for t in tickers if t != seed and is_valid_addition(group, t)]

            # Greedily add stocks that meet our conditions
            while len(group) < num_stocks and valid_candidates:
                # Try each candidate and see which one gives the best balance
                best_candidate = None
                best_candidate_score = float('-inf')

                for candidate in valid_candidates[:]:
                    # Create a test group with this candidate
                    test_group = group + [candidate]

                    # If we're at the final size, score based on correlation balance
                    if len(test_group) == num_stocks:
                        score = score_mixed_group(test_group)
                        if score > best_candidate_score:
                            best_candidate = candidate
                            best_candidate_score = score
                    else:
                        # Otherwise, prefer candidates that keep more options open
                        remaining = [t for t in tickers if t not in test_group and is_valid_addition(test_group, t)]
                        score = len(remaining)
                        if score > best_candidate_score:
                            best_candidate = candidate
                            best_candidate_score = score

                if best_candidate:
                    group.append(best_candidate)
                    # Update valid candidates for the next round
                    valid_candidates = [t for t in tickers if t not in group and is_valid_addition(group, t)]
                else:
                    # No valid candidates left
                    break

            # Score the final group
            if len(group) == num_stocks:
                score = score_mixed_group(group)
                stats = get_correlation_stats(group)

                if score > best_score:
                    best_group = group
                    best_score = score
                    best_stats = stats

        if best_group:
            return best_group, best_stats
        return [], {}

    # For non-mixed types, use the original algorithm
    for seed in tickers:
        group = [seed]
        candidates = [t for t in tickers if t != seed]

        while len(group) < num_stocks and candidates:
            for candidate in candidates[:]:
                if is_valid_addition(group, candidate):
                    group.append(candidate)
                    candidates.remove(candidate)
                    break
                else:
                    candidates.remove(candidate)

            if len(group) == num_stocks:
                return group, {}

    return [], {}