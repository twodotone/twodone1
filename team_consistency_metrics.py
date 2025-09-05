"""
Enhanced Team Predictability Metrics for NFL Betting

This module provides metrics focused on how predictably teams perform
relative to expectations, which is more relevant for betting decisions.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from scipy import stats
from team_mapping import get_all_team_abbrs

def analyze_team_consistency(pbp_df, team_abbr, min_games=8, return_details=False):
    """
    Provides a betting-focused analysis of team predictability.
    
    Parameters:
    -----------
    pbp_df : DataFrame
        Play-by-play data
    team_abbr : str
        Team abbreviation
    min_games : int
        Minimum number of games required for calculation
    return_details : bool
        Whether to return detailed metrics
        
    Returns:
    --------
    dict
        A dictionary containing predictability metrics relevant for betting
    """
    if pbp_df.empty or len(pbp_df) < 100:  # Need enough plays for meaningful analysis
        return {
            "consistency_score": 0.5,
            "percentile": 50,
            "std_dev": None,
            "game_metrics": None,
            "predictability_rating": "Average"
        }
    
    # Get all team abbreviations (current and historical)
    all_team_abbrs = get_all_team_abbrs(team_abbr)
    
    # Filter for team and regular season
    team_games = pbp_df[
        ((pbp_df['home_team'].isin(all_team_abbrs)) | 
         (pbp_df['away_team'].isin(all_team_abbrs)))
    ]
    team_games = team_games[team_games['season_type'] == 'REG']
    
    if team_games.empty:
        return {
            "consistency_score": 0.5,
            "percentile": 50,
            "std_dev": None,
            "game_metrics": None,
            "predictability_rating": "Average"
        }
    
    # Calculate game-by-game metrics
    game_ids = team_games['game_id'].unique()
    
    if len(game_ids) < min_games:
        return {
            "consistency_score": 0.5,
            "percentile": 50,
            "std_dev": None,
            "game_metrics": None,
            "predictability_rating": "Average"
        }
    
    # Collect game-by-game metrics
    game_metrics = []
    for game_id in game_ids:
        game_df = team_games[team_games['game_id'] == game_id]
        
        # Check if this team is home using any of its abbreviations
        is_home = any(game_df['home_team'].iloc[0] == abbr for abbr in all_team_abbrs)
        
        # Get the team abbreviation used in this specific game
        game_team_abbr = game_df['home_team'].iloc[0] if is_home else game_df['away_team'].iloc[0]
        
        # Get plays where this team had possession
        team_plays = game_df[game_df['posteam'] == game_team_abbr]
        opponent_abbr = game_df['away_team'].iloc[0] if is_home else game_df['home_team'].iloc[0]
        
        if not team_plays.empty:
            # EPA metrics
            avg_epa = team_plays['epa'].mean()
            
            # Get spread information if available - to measure performance vs. expectations
            spread_line = None
            if 'spread_line' in game_df.columns:
                try:
                    # Negative means home team is favored
                    spread_line = game_df['spread_line'].iloc[0]
                    if not is_home:  # Convert to away team perspective
                        spread_line = -spread_line
                except:
                    spread_line = None
            
            # Get actual game result if available
            actual_margin = None
            if 'result' in game_df.columns:
                try:
                    result = game_df['result'].iloc[0]  # Home team margin
                    actual_margin = result if is_home else -result
                except:
                    actual_margin = None
            
            # Additional metrics
            success_rate = (team_plays['epa'] > 0).mean() if not team_plays.empty else 0
            
            # Add to game metrics
            game_metrics.append({
                'game_id': game_id,
                'opponent': opponent_abbr,
                'is_home': is_home,
                'week': game_df['week'].iloc[0],
                'season': game_df['season'].iloc[0],
                'avg_epa': avg_epa,
                'success_rate': success_rate,
                'spread_line': spread_line,
                'actual_margin': actual_margin,
                'plays': len(team_plays)
            })
    
    if not game_metrics:
        return {
            "consistency_score": 0.5,
            "percentile": 50,
            "std_dev": None,
            "game_metrics": None,
            "predictability_rating": "Average"
        }
    
    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(game_metrics)
    
    # Calculate EPA consistency metrics
    epa_std = metrics_df['avg_epa'].std()
    epa_mean = metrics_df['avg_epa'].mean()
    
    # For betting purposes, we're interested in two primary components:
    # 1. Game-to-game EPA consistency (relative to team's own baseline)
    # 2. Performance vs. expectations (if spread data available)
    
    # First, calculate normalized EPA variability (adjusted for team quality)
    # We use coefficient of variation for positive EPA teams, standard deviation for negative
    if epa_mean > 0.05:
        # For positive EPA teams, use coefficient of variation (std/mean)
        cv = epa_std / abs(epa_mean)
        # Lower CV = more consistent performance
        # 0.4 is very consistent, 0.8+ is inconsistent (based on NFL data)
        epa_consistency = max(0, 1 - (cv / 0.8))
    else:
        # For teams with near-zero or negative EPA, use standard deviation directly
        # 0.15 is very consistent, 0.35+ is inconsistent (based on NFL data)
        epa_consistency = max(0, 1 - (epa_std / 0.35))
    
    # Now analyze spread-based predictability if data is available
    spread_predictability = 0.5  # Default to average
    ats_record = None
    cover_rate = 0.5  # Default to average
    
    if ('spread_line' in metrics_df.columns and 'actual_margin' in metrics_df.columns and
        not metrics_df['spread_line'].isna().all() and not metrics_df['actual_margin'].isna().all()):
        
        # Filter to games with both spread and result data
        valid_games = metrics_df[metrics_df['spread_line'].notna() & metrics_df['actual_margin'].notna()]
        
        if len(valid_games) >= 5:  # Need enough games for meaningful analysis
            # Calculate against-the-spread performance
            valid_games['ats_margin'] = valid_games['actual_margin'] + valid_games['spread_line']
            valid_games['covered'] = valid_games['ats_margin'] > 0
            
            # Calculate consistency of ATS performance
            ats_std = valid_games['ats_margin'].std()
            
            # Calculate cover rate (most important metric for betting)
            cover_rate = valid_games['covered'].mean()
            
            # For betting purposes, a team that consistently covers (very high cover rate)
            # or consistently fails to cover (very low cover rate) is actually more predictable
            # than a team that hovers around 50%.
            # 
            # Define predictability as distance from 50% cover rate PLUS consistency of margins
            
            # 1. Distance from 50% (higher is better)
            cover_distance = abs(cover_rate - 0.5)
            
            # 2. Low standard deviation of ATS margins (lower is better)
            # 7 points is very predictable, 14+ is unpredictable (based on NFL data)
            margin_consistency = max(0, 1 - (ats_std / 14))
            
            # 3. Combine both factors
            # Weight cover rate distance higher (0.7) than margin consistency (0.3)
            # Best predictability: Teams that consistently cover or fade by large margins
            spread_predictability = (cover_distance * 2.0) + (margin_consistency * 0.3)
            
            # Ensure we stay in the 0-1 range
            spread_predictability = min(max(spread_predictability, 0), 1)
            
            # Calculate ATS record (useful information)
            covers = valid_games['covered'].sum()
            ats_record = f"{covers}-{len(valid_games) - covers} ({cover_rate*100:.1f}%)"
    
    # Combine metrics - weight spread predictability much higher when available
    if ats_record:
        # When we have spread data, it's MUCH more important for betting
        consistency_score = 0.2 * epa_consistency + 0.8 * spread_predictability
    else:
        # Without spread data, rely more on basic consistency
        consistency_score = epa_consistency
    
    # Create a well-distributed range of NFL team percentiles
    # Adjust these to match the actual distribution we're seeing in our data
    # These thresholds are more realistic based on actual NFL team betting patterns
    team_percentiles = {
        5: 0.10,   # Bottom 5% of teams (extremely unpredictable)
        10: 0.15,
        20: 0.20,
        30: 0.25,
        40: 0.30,
        50: 0.35,  # Median NFL team
        60: 0.40,
        70: 0.45,
        80: 0.50,
        90: 0.55,
        95: 0.60   # Top 5% of teams (extremely predictable)
    }
    
    # Find percentile using linear interpolation
    percentile = 0
    for p in sorted(team_percentiles.keys()):
        threshold = team_percentiles[p]
        if consistency_score <= threshold:
            if p > min(team_percentiles.keys()):
                # Find previous percentile point
                prev_percentiles = [x for x in team_percentiles.keys() if x < p]
                prev_p = max(prev_percentiles)
                prev_threshold = team_percentiles[prev_p]
                
                # Linear interpolation
                if threshold > prev_threshold:  # Avoid division by zero
                    ratio = (consistency_score - prev_threshold) / (threshold - prev_threshold)
                    percentile = prev_p + ratio * (p - prev_p)
                else:
                    percentile = prev_p
            else:
                percentile = p
            break
    else:
        # If score is higher than all thresholds
        percentile = 100
    
    # Assign a predictability rating
    if percentile >= 80:
        predictability = "High"
        if ats_record and cover_rate > 0.7:
            betting_note = f"Strong ATS performer ({cover_rate*100:.1f}% cover rate)"
        elif ats_record and cover_rate < 0.3:
            betting_note = f"Consistently fails to cover ({cover_rate*100:.1f}% cover rate)"
        else:
            betting_note = "This team tends to perform predictably relative to expectations"
    elif percentile >= 40:
        predictability = "Average"
        betting_note = "This team shows typical game-to-game variability"
    else:
        predictability = "Low"
        betting_note = "This team shows high game-to-game variability"
    
    # Build result dictionary with meaningful betting context
    result = {
        "consistency_score": consistency_score,
        "percentile": percentile,
        "std_dev": epa_std,
        "mean_epa": epa_mean,
        "epa_consistency": epa_consistency,
        "spread_predictability": spread_predictability if ats_record else None,
        "ats_record": ats_record,
        "cover_rate": cover_rate if ats_record else None,
        "predictability_rating": predictability,
        "betting_note": betting_note
    }
    
    # Add detailed metrics if requested
    if return_details:
        result["game_metrics"] = metrics_df.to_dict('records')
    
    return result

def get_variance_scale_factor(consistency_data, baseline_variance=2.5):
    """
    Converts predictability metrics into a variance scale factor for prediction intervals.
    
    Parameters:
    -----------
    consistency_data : dict
        Dictionary from analyze_team_consistency
    baseline_variance : float
        Base variance for NFL spread predictions
        
    Returns:
    --------
    float
        Variance scale factor
    """
    # Extract percentile (higher = more predictable)
    percentile = consistency_data.get("percentile", 50)
    
    # Create a more meaningful mapping from percentile to variance scale
    # Teams at low percentiles (unpredictable) should have much higher variance
    # Teams at high percentiles (predictable) should have lower variance
    if percentile < 20:
        # Very unpredictable teams: 1.3-1.5x baseline variance
        variance_scale = 1.5 - (percentile / 20) * 0.2
    elif percentile < 50:
        # Below average teams: 1.1-1.3x baseline variance
        variance_scale = 1.3 - ((percentile - 20) / 30) * 0.2
    elif percentile < 80:
        # Above average teams: 0.9-1.1x baseline variance
        variance_scale = 1.1 - ((percentile - 50) / 30) * 0.2
    else:
        # Very predictable teams: 0.7-0.9x baseline variance
        variance_scale = 0.9 - ((percentile - 80) / 20) * 0.2
    
    # Apply the variance scale to the baseline
    return variance_scale * baseline_variance

def plot_team_consistency(consistency_data, team_abbr, st_container=None):
    """
    Creates a visualization of team predictability metrics focused on betting relevance.
    
    Parameters:
    -----------
    consistency_data : dict
        Dictionary from analyze_team_consistency with return_details=True
    team_abbr : str
        Team abbreviation
    st_container : streamlit container, optional
        Streamlit container to display the plot in
    
    Returns:
    --------
    fig
        Matplotlib figure object with the predictability plot
    """
    if not consistency_data.get("game_metrics"):
        # Create simple placeholder figure
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "Insufficient data for predictability analysis", 
                ha='center', va='center', fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        
        if st_container:
            st_container.pyplot(fig)
        
        return fig
    
    # Convert game metrics to DataFrame
    metrics_df = pd.DataFrame(consistency_data["game_metrics"])
    
    # Sort by season and week
    metrics_df = metrics_df.sort_values(by=['season', 'week'])
    
    # Create figure with subplots - adjust based on available data
    has_spread_data = ('spread_line' in metrics_df.columns and 
                       'actual_margin' in metrics_df.columns and
                       not metrics_df['spread_line'].isna().all() and 
                       not metrics_df['actual_margin'].isna().all())
    
    if has_spread_data:
        # Three panels when we have spread data
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), 
                                          gridspec_kw={'height_ratios': [2, 1, 1]})
    else:
        # Two panels without spread data
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), 
                                     gridspec_kw={'height_ratios': [2, 1]})
    
    # Plot EPA over time
    weeks = list(range(len(metrics_df)))
    ax1.plot(weeks, metrics_df['avg_epa'], 'b-', marker='o', label='EPA per Play')
    
    # Add mean and standard deviation lines
    mean_epa = metrics_df['avg_epa'].mean()
    std_epa = metrics_df['avg_epa'].std()
    ax1.axhline(y=mean_epa, color='r', linestyle='--', label=f'Mean EPA: {mean_epa:.3f}')
    ax1.axhline(y=mean_epa + std_epa, color='g', linestyle=':', label=f'±1 Std Dev: {std_epa:.3f}')
    ax1.axhline(y=mean_epa - std_epa, color='g', linestyle=':')
    
    # Shade the area between mean ± std dev to highlight consistency range
    ax1.fill_between(weeks, mean_epa - std_epa, mean_epa + std_epa, color='g', alpha=0.1)
    
    # Add opponent labels with coloring based on performance
    for i, row in enumerate(metrics_df.iterrows()):
        opponent = row[1]['opponent']
        is_home = row[1]['is_home']
        location = 'vs' if is_home else '@'
        epa = row[1]['avg_epa']
        
        # Color code based on performance relative to mean
        if epa > mean_epa + 0.5*std_epa:
            color = 'green'  # Good game
        elif epa < mean_epa - 0.5*std_epa:
            color = 'red'    # Bad game
        else:
            color = 'black'  # Average game
            
        ax1.annotate(f"{location} {opponent}", 
                    (i, epa),
                    textcoords="offset points",
                    xytext=(0, 10 if i % 2 == 0 else -20),
                    ha='center',
                    fontsize=8,
                    rotation=45,
                    color=color)
    
    # Plot success rate in second subplot
    ax2.plot(weeks, metrics_df['success_rate'], 'g-', marker='s', label='Success Rate')
    ax2.axhline(y=metrics_df['success_rate'].mean(), color='r', linestyle='--', 
               label=f'Mean: {metrics_df["success_rate"].mean():.3f}')
    
    # Add ATS performance subplot if data available
    if has_spread_data:
        # Filter to games with both spread and result data
        valid_games = metrics_df[metrics_df['spread_line'].notna() & metrics_df['actual_margin'].notna()]
        
        if not valid_games.empty:
            # Calculate against-the-spread performance
            valid_games['ats_margin'] = valid_games['actual_margin'] + valid_games['spread_line']
            valid_games['covered'] = valid_games['ats_margin'] > 0
            valid_indices = [metrics_df.index.get_loc(idx) for idx in valid_games.index]
            
            # Plot ATS margins
            ax3.bar(valid_indices, valid_games['ats_margin'], color=[
                'green' if x > 0 else 'red' for x in valid_games['ats_margin']
            ])
            ax3.axhline(y=0, color='black', linestyle='-')
            
            # Add context line
            ax3.axhline(y=valid_games['ats_margin'].mean(), color='blue', linestyle='--',
                      label=f'Avg ATS margin: {valid_games["ats_margin"].mean():.1f}')
            
            # Calculate ATS consistency
            ats_std = valid_games['ats_margin'].std()
            
            # Add cover percentage
            cover_pct = valid_games['covered'].mean() * 100
            
            # Color the title based on cover rate
            if cover_pct > 65:
                title_color = 'green'
                title_note = "STRONG ATS TEAM"
            elif cover_pct < 35:
                title_color = 'red'
                title_note = "POOR ATS TEAM"
            else:
                title_color = 'black'
                title_note = ""
                
            ax3.set_title(f"Against The Spread Performance: {cover_pct:.1f}% Cover Rate, {ats_std:.1f} pts Std Dev", 
                         color=title_color, fontweight='bold' if title_note else 'normal')
            
            # Add a text annotation for exceptional teams
            if title_note:
                ax3.text(0.5, 0.9, title_note, transform=ax3.transAxes,
                        fontsize=12, ha='center', fontweight='bold', color=title_color,
                        bbox=dict(facecolor='white', alpha=0.8, boxstyle='round'))
                
            ax3.set_ylabel("ATS Margin (pts)")
    
    # Customize plots
    ax1.set_title(f"{team_abbr} Game-by-Game Performance Predictability")
    ax1.set_ylabel("EPA per Play")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc='best')
    
    ax2.set_ylabel("Success Rate")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='best')
    
    if has_spread_data:
        ax3.grid(True, alpha=0.3)
        ax3.legend(loc='best')
        ax3.set_xlabel("Game Number")
    else:
        ax2.set_xlabel("Game Number")
    
    # Add predictability metrics as text with better betting context
    consistency_score = consistency_data.get("consistency_score", 0)
    percentile = consistency_data.get("percentile", 0)
    predictability = consistency_data.get("predictability_rating", "Average")
    betting_note = consistency_data.get("betting_note", "")
    ats_record = consistency_data.get("ats_record", None)
    cover_rate = consistency_data.get("cover_rate", 0.5)
    
    # Create a more betting-focused interpretation
    if ats_record:
        # If we have a strong ATS team, highlight this prominently
        if cover_rate > 0.65:
            betting_context = f"ATS Record: {ats_record}\n⭐ STRONG ATS PERFORMER ⭐\n{betting_note}"
        elif cover_rate < 0.35:
            betting_context = f"ATS Record: {ats_record}\n⚠️ CONSISTENTLY FAILS TO COVER ⚠️\n{betting_note}"
        else:
            betting_context = f"ATS Record: {ats_record}\n{betting_note}"
    else:
        betting_context = betting_note
    
    # Format based on predictability rating
    if predictability == "High":
        pred_note = "BETTING IMPACT: More reliable predictions possible"
    elif predictability == "Low":
        pred_note = "BETTING IMPACT: Use caution - highly variable performance"
    else:
        pred_note = "BETTING IMPACT: Typical prediction reliability"
    
    consistency_text = (
        f"Predictability Rating: {predictability.upper()} ({percentile:.0f}th percentile)\n" +
        f"Game-to-Game EPA Variability: {std_epa:.3f}\n\n" +
        f"{betting_context}\n\n" +
        f"{pred_note}"
    )
    
    # Add text box with interpretation
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.4)
    fig.text(0.02, 0.02, consistency_text, fontsize=9, verticalalignment='bottom', 
             bbox=props)
    
    fig.tight_layout(rect=[0, 0.08, 1, 1])  # Make room for the text at bottom
    
    # Display in Streamlit if container provided
    if st_container:
        st_container.pyplot(fig)
    
    return fig

# Example usage when run as script
if __name__ == "__main__":
    print("Team Predictability Metrics module loaded.")
