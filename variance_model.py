"""
Enhanced Variance Modeling for NFL Predictions

This module implements Bayesian credible intervals and team-specific variance tracking
to improve confidence estimates for predicted spreads, especially in close games.
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import streamlit as st

# Import enhanced team consistency metrics
try:
    from team_consistency_metrics import analyze_team_consistency, get_variance_scale_factor, plot_team_consistency
    from team_mapping import get_current_team_abbr, get_all_team_abbrs
    ENHANCED_METRICS_AVAILABLE = True
    ENHANCED_METRICS_AVAILABLE = True
except ImportError:
    ENHANCED_METRICS_AVAILABLE = False

def calculate_team_consistency(pbp_df, team_abbr, min_games=8):
    """
    Calculates a consistency score for a team based on variance in EPA performance.
    Lower values indicate more consistent performance.
    
    Parameters:
    -----------
    pbp_df : DataFrame
        Play-by-play data
    team_abbr : str
        Team abbreviation
    min_games : int
        Minimum number of games required for calculation
    
    Returns:
    --------
    float
        Consistency score (lower = more consistent)
    """
    if pbp_df.empty:
        return 1.0  # Default average consistency
    
    # Filter for team and regular season
    team_games = pbp_df[(pbp_df['home_team'] == team_abbr) | (pbp_df['away_team'] == team_abbr)]
    team_games = team_games[team_games['season_type'] == 'REG']
    
    if team_games.empty:
        return 1.0
    
    # Calculate game-by-game EPA
    game_ids = team_games['game_id'].unique()
    
    if len(game_ids) < min_games:
        return 1.0  # Not enough games for a reliable estimate
    
    game_epa = []
    for game_id in game_ids:
        game_df = team_games[team_games['game_id'] == game_id]
        is_home = game_df['home_team'].iloc[0] == team_abbr
        
        # Get plays where this team had possession
        if is_home:
            team_plays = game_df[game_df['posteam'] == team_abbr]
        else:
            team_plays = game_df[game_df['posteam'] == team_abbr]
        
        if not team_plays.empty:
            avg_epa = team_plays['epa'].mean()
            game_epa.append(avg_epa)
    
    # Calculate variance of game EPA (higher variance = less consistent)
    if game_epa:
        # Use coefficient of variation to normalize (std / mean)
        mean_epa = np.mean(game_epa)
        if abs(mean_epa) < 0.01:  # Avoid division by near-zero
            return 1.0
        
        std_epa = np.std(game_epa)
        
        # Calculate raw consistency score
        raw_score = std_epa / abs(mean_epa)
        
        # Use a more nuanced scaling approach
        # Based on empirical analysis of NFL team consistency:
        # - Most consistent teams: ~0.6-0.8
        # - Average teams: ~0.9-1.1
        # - Inconsistent teams: ~1.2-1.4
        # - Very inconsistent teams: >1.4
        
        # Apply a sigmoid-like scaling to spread values more evenly
        if raw_score < 0.5:
            consistency_score = 0.7  # Very consistent
        elif raw_score < 1.0:
            consistency_score = 0.7 + (raw_score - 0.5) * 0.4  # 0.7-0.9 range
        elif raw_score < 2.0:
            consistency_score = 0.9 + (raw_score - 1.0) * 0.3  # 0.9-1.2 range
        else:
            # For very inconsistent teams, cap at 1.7
            consistency_score = min(1.2 + (raw_score - 2.0) * 0.25, 1.7)
            
        return consistency_score
    
    return 1.0

def calculate_matchup_variance(home_consistency, away_consistency, base_variance=2.5):
    """
    Calculates expected variance for a matchup based on team consistency scores.
    
    Parameters:
    -----------
    home_consistency : float
        Home team consistency score (0.7-1.7 scale)
    away_consistency : float
        Away team consistency score (0.7-1.7 scale)
    base_variance : float
        Base variance for NFL spread predictions (increased to 2.5 for more conservative estimates)
    
    Returns:
    --------
    float
        Expected variance for the matchup (scaled to reasonable NFL range)
    """
    # Normalize consistency scores to 0.5-1.5 range for variance calculation
    # This gives us better control over the final variance range
    norm_home = (home_consistency - 0.7) / (1.7 - 0.7) + 0.5
    norm_away = (away_consistency - 0.7) / (1.7 - 0.7) + 0.5
    
    # Combine team consistency scores (geometric mean)
    combined_consistency = (norm_home * norm_away) ** 0.5
    
    # Scale to a reasonable range
    # For two very consistent teams (0.7, 0.7): ~1.5 points variance
    # For two average teams (1.0, 1.0): ~2.0 points variance
    # For two inconsistent teams (1.5, 1.5): ~3.5 points variance
    # For two very inconsistent teams (1.7, 1.7): ~4.0 points variance
    scaled_variance = base_variance * combined_consistency
    
    # Add some interpretability info for debugging
    return scaled_variance

def calculate_credible_interval(predicted_spread, matchup_variance, confidence=0.95):
    """
    Calculates a Bayesian credible interval for the predicted spread.
    
    Parameters:
    -----------
    predicted_spread : float
        The model's predicted spread
    matchup_variance : float
        Estimated variance for this matchup
    confidence : float
        Confidence level (0-1)
    
    Returns:
    --------
    tuple
        (lower_bound, upper_bound) of the credible interval
    """
    # Standard deviation is square root of variance
    std_dev = np.sqrt(matchup_variance)
    
    # Calculate critical value for the desired confidence level
    critical_value = stats.norm.ppf((1 + confidence) / 2)
    
    # Calculate interval
    lower_bound = predicted_spread - critical_value * std_dev
    upper_bound = predicted_spread + critical_value * std_dev
    
    return (round(lower_bound, 1), round(upper_bound, 1))

def calculate_win_probability_with_variance(model_spread, vegas_spread, matchup_variance):
    """
    Calculates win probability considering the specific matchup variance.
    
    Parameters:
    -----------
    model_spread : float
        The model's predicted spread
    vegas_spread : float
        The Vegas spread for comparison
    matchup_variance : float
        Estimated variance for this matchup
    
    Returns:
    --------
    float
        Win probability adjusted for matchup-specific variance
    """
    std_dev = np.sqrt(matchup_variance)
    spread_diff = model_spread - vegas_spread
    
    # Calculate win probability based on which side of the Vegas line our prediction falls
    if spread_diff >= 0:  # Model says home team is favored by more (or underdog by less)
        # Probability that true spread is greater than Vegas spread
        win_prob = 1 - stats.norm.cdf(vegas_spread, model_spread, std_dev)
    else:  # Model says home team is favored by less (or underdog by more)
        # Probability that true spread is less than Vegas spread
        win_prob = stats.norm.cdf(vegas_spread, model_spread, std_dev)
    
    # Apply a more conservative calibration that matches historical results
    # Even large edges don't guarantee wins (market efficiency)
    # Higher variance should reduce confidence more significantly
    
    # First, apply variance penalty - higher variance = lower max win probability
    # Baseline variance is 2.5, so normalize against that
    variance_factor = 2.5 / matchup_variance if matchup_variance > 0 else 1.0
    
    # Cap the variance factor to a reasonable range
    variance_factor = min(max(variance_factor, 0.7), 1.2)
    
    # Apply the variance-adjusted scaling
    if win_prob > 0.5:
        # Variance impacts our max probability - with high variance, cap at ~67%
        # With low variance, can go up to ~73%
        max_prob = 0.5 + (0.23 * variance_factor)
        
        # Scale from 0.5-1.0 range to 0.5-max_prob range
        return 0.5 + (win_prob - 0.5) * ((max_prob - 0.5) / 0.5)
    else:
        # For probabilities below 0.5, apply similar scaling
        min_prob = 0.5 - (0.23 * variance_factor)
        return 0.5 - (0.5 - win_prob) * ((0.5 - min_prob) / 0.5)

def get_enhanced_confidence_rating(model_spread, vegas_spread, matchup_variance, samples=100):
    """
    Gets an enhanced confidence rating considering matchup-specific variance.
    
    Parameters:
    -----------
    model_spread : float
        The model's predicted spread
    vegas_spread : float
        The Vegas spread for comparison
    matchup_variance : float
        Estimated variance for this matchup
    samples : int
        Number of historical samples to consider similar
    
    Returns:
    --------
    tuple
        (stars, confidence_text, win_prob, recommendation, samples)
    """
    win_prob = calculate_win_probability_with_variance(model_spread, vegas_spread, matchup_variance)
    
    # Calculate absolute edge magnitude
    edge_magnitude = abs(model_spread - vegas_spread)
    
    # Variance-based penalty - larger penalties for higher variance
    # Base variance is 2.5, so normalize relative to that
    variance_ratio = matchup_variance / 2.5
    
    # Calculate variance penalty (0 for normal variance, increases with higher variance)
    # Higher variance = higher penalty
    if variance_ratio <= 1.0:
        # Normal or low variance = no penalty
        variance_penalty = 0
    elif variance_ratio <= 1.2:
        # Slightly high variance = small penalty
        variance_penalty = 0.5
    elif variance_ratio <= 1.4:
        # High variance = medium penalty
        variance_penalty = 1
    else:
        # Very high variance = large penalty
        variance_penalty = 1.5
    
    # Edge magnitude penalty - smaller edges get penalized more
    edge_penalty = 0
    if edge_magnitude < 2:
        edge_penalty = 1.5  # Very small edge
    elif edge_magnitude < 3:
        edge_penalty = 1    # Small edge
    elif edge_magnitude < 4:
        edge_penalty = 0.5  # Moderate edge
    
    # Calculate total penalty (fractional stars to be deducted)
    total_penalty = variance_penalty + edge_penalty
    
    # Fixed thresholds based on win probability - more conservative version
    # These align with backtesting results
    if win_prob < 0.55:  # Increased from 0.53
        base_stars = 1
        confidence_text = "Very Low"
        recommendation = "AVOID BETTING"
    elif win_prob < 0.58:  # Increased from 0.56
        base_stars = 2
        confidence_text = "Low"
        recommendation = "SMALL BET ONLY"
    elif win_prob < 0.62:  # Increased from 0.60
        base_stars = 3
        confidence_text = "Moderate"
        recommendation = "STANDARD UNIT"
    elif win_prob < 0.67:  # Increased from 0.65
        base_stars = 4
        confidence_text = "High"
        recommendation = "1.5-2 UNITS"
    else:  # 0.67+
        base_stars = 5
        confidence_text = "Very High"
        recommendation = "2-3 UNITS"
        
    # Apply penalties (but never go below 1 star)
    # Convert to float, apply penalty, then round to nearest 0.5
    adjusted_stars = max(1, round((base_stars - total_penalty) * 2) / 2)
    
    # Round to nearest integer for the final rating
    stars = round(adjusted_stars)
    
    # Update confidence text and recommendation based on potentially adjusted star rating
    confidence_text_map = {
        1: "Very Low",
        2: "Low",
        3: "Moderate", 
        4: "High",
        5: "Very High"
    }
    confidence_text = confidence_text_map[stars]
    
    # Update recommendation based on adjusted star rating
    if stars == 1:
        recommendation = "AVOID BETTING"
    elif stars == 2:
        recommendation = "SMALL BET ONLY"
    elif stars == 3:
        recommendation = "STANDARD UNIT"
    elif stars == 4:
        recommendation = "1.5-2 UNITS"
    else:  # stars == 5
        recommendation = "2-3 UNITS"
    
    return (stars, confidence_text, win_prob, recommendation, samples)

def plot_prediction_distribution(predicted_spread, matchup_variance, vegas_spread):
    """
    Creates a plot showing the probability distribution of predicted spreads.
    
    Parameters:
    -----------
    predicted_spread : float
        The model's predicted spread
    matchup_variance : float
        Estimated variance for this matchup
    vegas_spread : float
        The Vegas spread for comparison
    
    Returns:
    --------
    fig
        Matplotlib figure object with the distribution plot
    """
    std_dev = np.sqrt(matchup_variance)
    
    # Create range of values for x-axis
    x = np.linspace(predicted_spread - 3*std_dev, predicted_spread + 3*std_dev, 1000)
    
    # Calculate normal PDF
    pdf = stats.norm.pdf(x, predicted_spread, std_dev)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot distribution
    ax.plot(x, pdf, 'b-', linewidth=2, label='Model Distribution')
    
    # Add vertical lines for predicted spread and Vegas spread
    ax.axvline(x=predicted_spread, color='blue', linestyle='-', linewidth=2, label='Model Spread')
    ax.axvline(x=vegas_spread, color='red', linestyle='--', linewidth=2, label='Vegas Spread')
    
    # Calculate 95% credible interval
    lower, upper = calculate_credible_interval(predicted_spread, matchup_variance)
    
    # Shade the 95% credible interval
    ax.fill_between(x[(x >= lower) & (x <= upper)], pdf[(x >= lower) & (x <= upper)], 
                   color='blue', alpha=0.2, label='95% Credible Interval')
    
    # Calculate probability that the true line is on the right side of the Vegas line
    # This is the "probability of beating Vegas"
    win_prob = calculate_win_probability_with_variance(predicted_spread, vegas_spread, matchup_variance)
    
    # Shade the area representing the win probability
    if predicted_spread > vegas_spread:
        # Model has home team favored by more (or underdog by less)
        # Win if true spread > Vegas spread
        ax.fill_between(x[x > vegas_spread], 0, pdf[x > vegas_spread], 
                      color='green', alpha=0.3, label=f'Win Probability: {win_prob:.1%}')
    else:
        # Model has home team favored by less (or underdog by more)
        # Win if true spread < Vegas spread
        ax.fill_between(x[x < vegas_spread], 0, pdf[x < vegas_spread], 
                      color='green', alpha=0.3, label=f'Win Probability: {win_prob:.1%}')
    
    # Customize plot
    ax.set_title(f'Prediction Distribution\nProbability of beating Vegas: {win_prob:.1%}')
    ax.set_xlabel('Spread (Home Team Perspective)')
    ax.set_ylabel('Probability Density')
    
    # Add edge magnitude information
    edge_magnitude = abs(predicted_spread - vegas_spread)
    ax.text(0.02, 0.97, f"Edge: {edge_magnitude:.1f} points", transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add variance information
    variance_color = "#d63031" if matchup_variance > 3.0 else "#00b894"
    ax.text(0.02, 0.90, f"Variance: {matchup_variance:.1f} (±{std_dev:.1f} pts)", transform=ax.transAxes, 
           fontsize=10, verticalalignment='top', color=variance_color,
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.legend(loc='best')
    ax.grid(True, linestyle='--', alpha=0.7)
    
    return fig

def add_enhanced_confidence_to_streamlit(home_team, away_team, model_spread, vegas_spread, pbp_df, current_year):
    """
    Adds enhanced confidence visualization to a Streamlit app.
    
    Parameters:
    -----------
    home_team : str
        Home team abbreviation
    away_team : str
        Away team abbreviation
    model_spread : float
        Model predicted spread (from home team perspective)
    vegas_spread : float
        Vegas spread (from home team perspective)
    pbp_df : DataFrame
        Play-by-play data
    current_year : int
        Current year being analyzed
    """
    st.write("### Enhanced Variance Model (BETA)")
    
    # Use enhanced metrics if available
    if ENHANCED_METRICS_AVAILABLE:
        # Ensure we're using current team abbreviations
        current_home_team = get_current_team_abbr(home_team)
        current_away_team = get_current_team_abbr(away_team)
        
        # Calculate enhanced team consistency metrics
        home_metrics = analyze_team_consistency(pbp_df, current_home_team, return_details=True)
        away_metrics = analyze_team_consistency(pbp_df, current_away_team, return_details=True)
        
        # Get consistency scores
        home_consistency_score = home_metrics["consistency_score"]
        away_consistency_score = away_metrics["consistency_score"]
        
        # Calculate matchup variance
        home_variance = get_variance_scale_factor(home_metrics)
        away_variance = get_variance_scale_factor(away_metrics)
        matchup_variance = (home_variance * away_variance) ** 0.5
        
        # Display detailed consistency metrics
        col1, col2 = st.columns(2)
        
        with col1:
            # Home team consistency
            percentile = home_metrics["percentile"]
            if percentile > 80:
                consistency_color = "#00b894"  # Green (very consistent)
                consistency_label = "Very Consistent"
            elif percentile > 60:
                consistency_color = "#00cec9"  # Light blue-green (consistent)
                consistency_label = "Consistent"
            elif percentile > 40:
                consistency_color = "#fdcb6e"  # Yellow (average)
                consistency_label = "Average"
            elif percentile > 20:
                consistency_color = "#e17055"  # Light orange (inconsistent)
                consistency_label = "Inconsistent"
            else:
                consistency_color = "#d63031"  # Red (very inconsistent)
                consistency_label = "Very Inconsistent"
            
            # Add ATS information if available
            ats_record = home_metrics.get("ats_record", None)
            cover_rate = home_metrics.get("cover_rate", None)
            
            # Add EPA std dev information for more concrete context
            epa_std = home_metrics.get("std_dev", 0)
            epa_mean = home_metrics.get("mean_epa", 0)
            raw_score = home_metrics.get("consistency_score", 0)
            
            # Add ATS highlight if cover rate is exceptional
            ats_highlight = ""
            if cover_rate is not None and isinstance(cover_rate, (int, float)):
                if cover_rate > 0.65:
                    ats_highlight = f'<div style="margin-top:5px; padding:3px; background-color:#d5f5e3; border-radius:3px; font-weight:bold; color:#27ae60;">Strong ATS Team: {cover_rate*100:.1f}% Cover Rate</div>'
                elif cover_rate < 0.35:
                    ats_highlight = f'<div style="margin-top:5px; padding:3px; background-color:#fadbd8; border-radius:3px; font-weight:bold; color:#e74c3c;">Poor ATS Team: {cover_rate*100:.1f}% Cover Rate</div>'
            
            st.markdown(f"""
                **{home_team} Predictability:** <span style='color:{consistency_color}'>{percentile:.0f}th</span> percentile ({consistency_label})
                <div style="font-size:0.9em;">
                    EPA Std Dev: <b>{epa_std:.3f}</b> | Mean EPA: {epa_mean:.3f}
                </div>
                <div style="font-size:0.9em;">
                    {ats_record if ats_record else "No ATS data available"}
                </div>
                {ats_highlight}
                <div style="font-size:0.8em;">
                    Raw Predictability Score: {raw_score:.3f} → {percentile:.0f}th percentile
                </div>
                <div style="font-size:0.8em; color:gray; margin-top:5px;">
                    {consistency_label} teams show {"more" if percentile > 60 else "less"} predictable betting outcomes
                </div>
            """, unsafe_allow_html=True)
            
            # Add a "View Details" expander
            with st.expander(f"View {home_team} Consistency Details"):
                # Plot consistency metrics
                plot_team_consistency(home_metrics, home_team, st)
        
        with col2:
            # Away team consistency
            percentile = away_metrics["percentile"]
            if percentile > 80:
                consistency_color = "#00b894"  # Green (very consistent)
                consistency_label = "Very Consistent"
            elif percentile > 60:
                consistency_color = "#00cec9"  # Light blue-green (consistent)
                consistency_label = "Consistent"
            elif percentile > 40:
                consistency_color = "#fdcb6e"  # Yellow (average)
                consistency_label = "Average"
            elif percentile > 20:
                consistency_color = "#e17055"  # Light orange (inconsistent)
                consistency_label = "Inconsistent"
            else:
                consistency_color = "#d63031"  # Red (very inconsistent)
                consistency_label = "Very Inconsistent"
            
            # Add ATS information if available
            ats_record = away_metrics.get("ats_record", None)
            cover_rate = away_metrics.get("cover_rate", None)
            
            # Add EPA std dev information for more concrete context
            epa_std = away_metrics.get("std_dev", 0)
            epa_mean = away_metrics.get("mean_epa", 0)
            raw_score = away_metrics.get("consistency_score", 0)
            
            # Add ATS highlight if cover rate is exceptional
            ats_highlight = ""
            if cover_rate is not None and isinstance(cover_rate, (int, float)):
                if cover_rate > 0.65:
                    ats_highlight = f'<div style="margin-top:5px; padding:3px; background-color:#d5f5e3; border-radius:3px; font-weight:bold; color:#27ae60;">Strong ATS Team: {cover_rate*100:.1f}% Cover Rate</div>'
                elif cover_rate < 0.35:
                    ats_highlight = f'<div style="margin-top:5px; padding:3px; background-color:#fadbd8; border-radius:3px; font-weight:bold; color:#e74c3c;">Poor ATS Team: {cover_rate*100:.1f}% Cover Rate</div>'
            
            st.markdown(f"""
                **{away_team} Predictability:** <span style='color:{consistency_color}'>{percentile:.0f}th</span> percentile ({consistency_label})
                <div style="font-size:0.9em;">
                    EPA Std Dev: <b>{epa_std:.3f}</b> | Mean EPA: {epa_mean:.3f}
                </div>
                <div style="font-size:0.9em;">
                    {ats_record if ats_record else "No ATS data available"}
                </div>
                {ats_highlight}
                <div style="font-size:0.8em;">
                    Raw Predictability Score: {raw_score:.3f} → {percentile:.0f}th percentile
                </div>
                <div style="font-size:0.8em; color:gray; margin-top:5px;">
                    {consistency_label} teams show {"more" if percentile > 60 else "less"} predictable betting outcomes
                </div>
            """, unsafe_allow_html=True)
            
            # Add a "View Details" expander
            with st.expander(f"View {away_team} Consistency Details"):
                # Plot consistency metrics
                plot_team_consistency(away_metrics, away_team, st)
    else:
        # Fall back to original implementation
        home_consistency = calculate_team_consistency(pbp_df, home_team)
        away_consistency = calculate_team_consistency(pbp_df, away_team)
        
        # Calculate matchup variance
        matchup_variance = calculate_matchup_variance(home_consistency, away_consistency)
        
        # Display team consistency info
        col1, col2 = st.columns(2)
        with col1:
            # More granular color coding
            if home_consistency < 0.8:
                consistency_color = "#00b894"  # Mint green (very consistent)
                consistency_label = "Very Consistent"
            elif home_consistency < 1.0:
                consistency_color = "#00cec9"  # Light blue-green (consistent)
                consistency_label = "Consistent"
            elif home_consistency < 1.2:
                consistency_color = "#fdcb6e"  # Yellow (average)
                consistency_label = "Average"
            elif home_consistency < 1.4:
                consistency_color = "#e17055"  # Light orange (inconsistent)
                consistency_label = "Inconsistent"
            else:
                consistency_color = "#d63031"  # Red (very inconsistent)
                consistency_label = "Very Inconsistent"
                
            st.markdown(f"**{home_team} Consistency:** <span style='color:{consistency_color}'>{home_consistency:.2f}</span> ({consistency_label})", unsafe_allow_html=True)
        
        with col2:
            # More granular color coding
            if away_consistency < 0.8:
                consistency_color = "#00b894"  # Mint green (very consistent)
                consistency_label = "Very Consistent"
            elif away_consistency < 1.0:
                consistency_color = "#00cec9"  # Light blue-green (consistent)
                consistency_label = "Consistent"
            elif away_consistency < 1.2:
                consistency_color = "#fdcb6e"  # Yellow (average)
                consistency_label = "Average"
            elif away_consistency < 1.4:
                consistency_color = "#e17055"  # Light orange (inconsistent)
                consistency_label = "Inconsistent"
            else:
                consistency_color = "#d63031"  # Red (very inconsistent)
                consistency_label = "Very Inconsistent"
                
            st.markdown(f"**{away_team} Consistency:** <span style='color:{consistency_color}'>{away_consistency:.2f}</span> ({consistency_label})", unsafe_allow_html=True)
    
    # Display matchup variance with better context
    if ENHANCED_METRICS_AVAILABLE:
        # For enhanced metrics
        if matchup_variance < 2.0:
            variance_color = "#00b894"  # Mint green
            variance_label = "Very Low"
        elif matchup_variance < 2.5:
            variance_color = "#00cec9"  # Light blue-green
            variance_label = "Low"
        elif matchup_variance < 3.0:
            variance_color = "#fdcb6e"  # Yellow
            variance_label = "Average"
        elif matchup_variance < 3.5:
            variance_color = "#e17055"  # Light orange
            variance_label = "High"
        else:
            variance_color = "#d63031"  # Red
            variance_label = "Very High"
    else:
        # For legacy metrics
        if matchup_variance < 2.0:
            variance_color = "#00b894"  # Mint green
            variance_label = "Low"
        elif matchup_variance < 2.5:
            variance_color = "#00cec9"  # Light blue-green
            variance_label = "Below Average"
        elif matchup_variance < 3.0:
            variance_color = "#fdcb6e"  # Yellow
            variance_label = "Average"
        elif matchup_variance < 3.5:
            variance_color = "#e17055"  # Light orange
            variance_label = "Above Average"
        else:
            variance_color = "#d63031"  # Red
            variance_label = "High"
        
    # Show variance with standard deviation for easier interpretation
    std_dev = np.sqrt(matchup_variance)
    
    # Add a reference point to make the variance more meaningful
    if matchup_variance < 2.5:
        reference = "Lower than typical NFL matchups (more predictable outcome)"
    elif matchup_variance > 3.5:
        reference = "Higher than typical NFL matchups (less predictable outcome)"
    else:
        reference = "Typical for NFL matchups"
    
    # Get ATS records if available to add context
    home_ats = home_metrics.get("ats_record", "N/A") if ENHANCED_METRICS_AVAILABLE else "N/A"
    away_ats = away_metrics.get("ats_record", "N/A") if ENHANCED_METRICS_AVAILABLE else "N/A"
    
    # Check for strong ATS teams
    home_cover_rate = home_metrics.get("cover_rate", 0.5) if ENHANCED_METRICS_AVAILABLE else 0.5
    away_cover_rate = away_metrics.get("cover_rate", 0.5) if ENHANCED_METRICS_AVAILABLE else 0.5
    
    # Make sure cover rates are not None before using them
    if home_cover_rate is None:
        home_cover_rate = 0.5
    if away_cover_rate is None:
        away_cover_rate = 0.5
    
    ats_note = ""
    if home_cover_rate > 0.65 and away_cover_rate > 0.65:
        ats_note = '<p style="color:#27ae60;"><b>Note:</b> Both teams have strong ATS records</p>'
    elif home_cover_rate > 0.65:
        ats_note = f'<p style="color:#27ae60;"><b>Note:</b> {home_team} has a strong ATS record</p>'
    elif away_cover_rate > 0.65:
        ats_note = f'<p style="color:#27ae60;"><b>Note:</b> {away_team} has a strong ATS record</p>'
    elif home_cover_rate < 0.35 and away_cover_rate < 0.35:
        ats_note = '<p style="color:#e74c3c;"><b>Note:</b> Both teams have poor ATS records</p>'
    elif home_cover_rate < 0.35:
        ats_note = f'<p style="color:#e74c3c;"><b>Note:</b> {home_team} has a poor ATS record</p>'
    elif away_cover_rate < 0.35:
        ats_note = f'<p style="color:#e74c3c;"><b>Note:</b> {away_team} has a poor ATS record</p>'
    
    st.markdown(
        f"""
        <div style="padding: 10px; border: 1px solid {variance_color}; border-radius: 5px; margin-bottom: 10px;">
            <h4 style="margin-top: 0;">Matchup Predictability</h4>
            <p><b>Variance:</b> <span style='color:{variance_color}'>{matchup_variance:.2f}</span> points² 
            (<span style='color:{variance_color}'>{variance_label}</span>)</p>
            <p><b>Standard Deviation:</b> ±{std_dev:.1f} points</p>
            <p style="font-size:0.9em;"><i>{reference}</i></p>
            <p style="font-size:0.9em;"><b>ATS Records:</b> {home_team}: {home_ats} | {away_team}: {away_ats}</p>
            {ats_note}
            <p style="font-size:0.8em;">Lower variance = more predictable outcome = more confidence in the prediction (all else equal)</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Calculate credible interval
    lower, upper = calculate_credible_interval(model_spread, matchup_variance)
    
    # Display credible interval
    st.markdown(f"**95% Credible Interval:** {lower} to {upper} points")
    
    # Calculate spread difference
    spread_diff = abs(model_spread - vegas_spread)
    
    # Calculate enhanced confidence rating
    stars, confidence_text, win_prob, recommendation, samples = get_enhanced_confidence_rating(
        model_spread, vegas_spread, matchup_variance
    )
    
    # Create a visually prominent confidence indicator
    st.divider()
    confidence_color = {
        1: "#ff6b6b",  # Red for very low confidence
        2: "#ffa26b",  # Orange for low confidence
        3: "#ffd56b",  # Yellow for moderate confidence
        4: "#7bed9f",  # Light green for high confidence
        5: "#2ed573"   # Dark green for very high confidence
    }
    
    # Calculate key factors that influenced the rating
    edge_magnitude = abs(model_spread - vegas_spread)
    avg_team_percentile = (home_metrics["percentile"] + away_metrics["percentile"]) / 2 if ENHANCED_METRICS_AVAILABLE else 50
    
    # Determine which factors most influenced the confidence rating
    confidence_drivers = []
    if edge_magnitude < 3:
        confidence_drivers.append(f"<span style='color:red'>Small edge ({edge_magnitude:.1f} points)</span>")
    elif edge_magnitude > 5:
        confidence_drivers.append(f"<span style='color:green'>Large edge ({edge_magnitude:.1f} points)</span>")
    
    if ENHANCED_METRICS_AVAILABLE:
        if avg_team_percentile > 70:
            confidence_drivers.append("<span style='color:green'>High team consistency</span>")
        elif avg_team_percentile < 30:
            confidence_drivers.append("<span style='color:red'>Low team consistency</span>")
            
        if matchup_variance < 2.5:
            confidence_drivers.append("<span style='color:green'>Low matchup variance</span>")
        elif matchup_variance > 3.5:
            confidence_drivers.append("<span style='color:red'>High matchup variance</span>")
    
    # Add default text if no specific drivers were identified
    if not confidence_drivers:
        confidence_drivers.append("Average edge and team consistency")
    
    st.markdown(
        f"""
        <div style="background-color: {confidence_color[stars]}; padding: 10px; border-radius: 10px; margin-bottom: 15px;">
            <h3 style="text-align: center; margin: 0; color: black;">Enhanced Confidence Rating: {"★" * stars}{"☆" * (5-stars)}</h3>
            <h4 style="text-align: center; margin: 5px 0; color: black;">{confidence_text.upper()} CONFIDENCE ({win_prob*100:.1f}%)</h4>
            <p style="text-align: center; margin: 0; font-weight: bold; color: black;">{recommendation}</p>
            <p style="text-align: center; margin: 5px 0; color: black;">Key factors: {" • ".join(confidence_drivers)}</p>
            <p style="text-align: center; margin: 0; font-size: 0.8em; color: black;">Edge: {edge_magnitude:.1f} points | Matchup Variance: {matchup_variance:.2f}</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Plot the prediction distribution
    fig = plot_prediction_distribution(model_spread, matchup_variance, vegas_spread)
    st.pyplot(fig)
    
    # Explanation of the enhanced model
    with st.expander("How the Enhanced Variance Model Works (and Its Limitations)"):
        st.write("""
        ### Enhanced Variance Model Methodology
        
        The Enhanced Variance Model attempts to improve confidence ratings by:
        
        1. **Team Consistency Analysis**: Measures how consistently each team performs from game to game
        2. **Matchup-Specific Variance**: Calculates expected variance for this specific matchup
        3. **Bayesian Credible Intervals**: Shows the range where the true spread likely falls
        4. **Adjusted Confidence Thresholds**: More conservative thresholds that require 3+ point edges for high confidence
        
        ### Important Limitations
        
        This model is deliberately calibrated to be conservative, with win probabilities capped at 75% (matching historical results).
        
        **Key limitations to be aware of:**
        
        - **Theoretical Foundation**: The model assumes that team consistency is predictive of future performance and betting outcomes
        - **Limited Backtesting**: The variance-based adjustments haven't been extensively validated against historical results
        - **Simplified Consistency Metric**: Team consistency is measured using a simplified formula that may not capture all relevant factors
        
        ### How This Compares to the Standard Model
        
        The standard model uses direct empirical data from past betting results to establish confidence levels based purely on edge size.
        This enhanced model adds a theoretical adjustment layer based on team consistency.
        
        **In general:**
        - For standard betting decisions, the standard model's empirically-derived confidence ratings may be more reliable
        - The enhanced model provides additional context about team consistency that might be useful for specific matchups
        
        Remember that even the best models can't guarantee outcomes - this is a probability-based approach to find value.
        """)
        
        if ENHANCED_METRICS_AVAILABLE:
            st.write("""
            ### Enhanced Metrics (BETA)
            
            This model uses advanced consistency metrics that analyze:
            
            - **Game-by-Game EPA Stability**: How consistently a team performs from game to game
            - **Success Rate Correlation**: How well success rate correlates with EPA (more stable teams show stronger correlation)
            - **Performance Trends**: Whether a team shows consistent improvement or decline
            
            The consistency percentile shows how a team ranks relative to the rest of the league:
            
            - **0-20th percentile**: Very inconsistent teams with high game-to-game variance
            - **20-40th percentile**: Below-average consistency
            - **40-60th percentile**: Average NFL team consistency
            - **60-80th percentile**: Above-average consistency
            - **80-100th percentile**: Very consistent teams with predictable performance
            
            Teams with higher consistency percentiles tend to be more predictable, potentially making the model's predictions more reliable for these teams.
            """)
            
        st.write("""
        ### Interpreting the Results
        
        - **Variance**: Lower variance = more confident prediction (all else equal)
        - **Credible Interval**: 95% chance the true spread falls in this range
        - **Confidence Rating**: Considers both edge size AND team consistency
        
        The enhanced model may show higher confidence than the standard model for consistent teams, or lower confidence for inconsistent teams, even at the same edge size.
        """)
    
    # Add a note about beta status
    st.caption("*Note: Enhanced Variance Model is in BETA. Feedback welcome.*")

# For standalone testing
if __name__ == "__main__":
    print("Enhanced Variance Model module loaded.")
