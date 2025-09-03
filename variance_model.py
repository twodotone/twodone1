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
        if mean_epa == 0:
            return 1.0
        
        std_epa = np.std(game_epa)
        # Scale to a reasonable range (0.5 to 1.5)
        consistency_score = min(max(std_epa / abs(mean_epa) if mean_epa != 0 else 1.0, 0.5), 1.5)
        return consistency_score
    
    return 1.0

def calculate_matchup_variance(home_consistency, away_consistency, base_variance=2.0):
    """
    Calculates expected variance for a matchup based on team consistency scores.
    
    Parameters:
    -----------
    home_consistency : float
        Home team consistency score
    away_consistency : float
        Away team consistency score
    base_variance : float
        Base variance for NFL spread predictions
    
    Returns:
    --------
    float
        Expected variance for the matchup
    """
    # Combine team consistency scores (multiplicative model)
    combined_consistency = (home_consistency * away_consistency) ** 0.5
    
    # Scale to reasonable range (typical NFL spread std dev is around 2-3 points)
    return base_variance * combined_consistency

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

def calculate_win_probability_with_variance(spread_diff, matchup_variance):
    """
    Calculates win probability considering the specific matchup variance.
    
    Parameters:
    -----------
    spread_diff : float
        Difference between model spread and Vegas spread
    matchup_variance : float
        Estimated variance for this matchup
    
    Returns:
    --------
    float
        Win probability adjusted for matchup-specific variance
    """
    # Convert spread difference to Z-score using matchup-specific std dev
    std_dev = np.sqrt(matchup_variance)
    z_score = spread_diff / std_dev if std_dev > 0 else 0
    
    # Convert Z-score to probability using normal CDF
    win_prob = stats.norm.cdf(z_score)
    
    return win_prob

def get_enhanced_confidence_rating(spread_diff, matchup_variance, samples=100):
    """
    Gets an enhanced confidence rating considering matchup-specific variance.
    
    Parameters:
    -----------
    spread_diff : float
        Difference between model spread and Vegas spread
    matchup_variance : float
        Estimated variance for this matchup
    samples : int
        Number of historical samples to consider similar
    
    Returns:
    --------
    tuple
        (stars, confidence_text, win_prob, recommendation)
    """
    win_prob = calculate_win_probability_with_variance(spread_diff, matchup_variance)
    
    # Adjust thresholds based on matchup variance
    # Lower variance = more confident at smaller spreads
    variance_factor = np.sqrt(matchup_variance) / np.sqrt(4.0)  # Normalize to a baseline variance of 4.0
    
    # Base thresholds from confidence_ratings.py
    thresholds = {
        1: 0.53,  # Very low confidence (avoid betting)
        2: 0.56,  # Low confidence (small bets only)
        3: 0.60,  # Moderate confidence (standard unit)
        4: 0.65,  # High confidence (1.5-2 units)
        5: 0.70   # Very high confidence (2-3 units)
    }
    
    # Adjust thresholds based on variance
    adjusted_thresholds = {k: v / variance_factor for k, v in thresholds.items()}
    
    # Determine confidence level
    stars = 1  # Default to lowest confidence
    for level in range(5, 0, -1):
        if win_prob >= adjusted_thresholds[level]:
            stars = level
            break
    
    # Confidence text
    confidence_text_map = {
        1: "Very Low",
        2: "Low",
        3: "Moderate", 
        4: "High",
        5: "Very High"
    }
    confidence_text = confidence_text_map[stars]
    
    # Betting recommendation
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
    
    # Calculate probability of beating Vegas spread
    win_prob = calculate_win_probability_with_variance(predicted_spread - vegas_spread, matchup_variance)
    
    # Customize plot
    ax.set_title(f'Prediction Distribution\nProbability of beating Vegas: {win_prob:.1%}')
    ax.set_xlabel('Spread (Home Team Perspective)')
    ax.set_ylabel('Probability Density')
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
    
    # Calculate team consistency scores
    home_consistency = calculate_team_consistency(pbp_df, home_team)
    away_consistency = calculate_team_consistency(pbp_df, away_team)
    
    # Calculate matchup variance
    matchup_variance = calculate_matchup_variance(home_consistency, away_consistency)
    
    # Calculate credible interval
    lower, upper = calculate_credible_interval(model_spread, matchup_variance)
    
    # Display team consistency info
    col1, col2 = st.columns(2)
    with col1:
        consistency_color = "green" if home_consistency < 1.0 else ("orange" if home_consistency < 1.2 else "red")
        st.markdown(f"**{home_team} Consistency:** <span style='color:{consistency_color}'>{home_consistency:.2f}</span> (lower is better)", unsafe_allow_html=True)
    
    with col2:
        consistency_color = "green" if away_consistency < 1.0 else ("orange" if away_consistency < 1.2 else "red")
        st.markdown(f"**{away_team} Consistency:** <span style='color:{consistency_color}'>{away_consistency:.2f}</span> (lower is better)", unsafe_allow_html=True)
    
    # Display matchup variance
    variance_color = "green" if matchup_variance < 3.0 else ("orange" if matchup_variance < 5.0 else "red")
    st.markdown(f"**Matchup Variance:** <span style='color:{variance_color}'>{matchup_variance:.2f}</span> points²", unsafe_allow_html=True)
    
    # Display credible interval
    st.markdown(f"**95% Credible Interval:** {lower} to {upper} points")
    
    # Calculate spread difference
    spread_diff = abs(model_spread - vegas_spread)
    
    # Calculate enhanced confidence rating
    stars, confidence_text, win_prob, recommendation, samples = get_enhanced_confidence_rating(spread_diff, matchup_variance)
    
    # Create a visually prominent confidence indicator
    st.divider()
    confidence_color = {
        1: "#ff6b6b",  # Red for very low confidence
        2: "#ffa26b",  # Orange for low confidence
        3: "#ffd56b",  # Yellow for moderate confidence
        4: "#7bed9f",  # Light green for high confidence
        5: "#2ed573"   # Dark green for very high confidence
    }
    
    st.markdown(
        f"""
        <div style="background-color: {confidence_color[stars]}; padding: 10px; border-radius: 10px; margin-bottom: 15px;">
            <h3 style="text-align: center; margin: 0; color: black;">Enhanced Confidence Rating: {"★" * stars}{"☆" * (5-stars)}</h3>
            <h4 style="text-align: center; margin: 5px 0; color: black;">{confidence_text.upper()} CONFIDENCE ({win_prob*100:.1f}%)</h4>
            <p style="text-align: center; margin: 0; font-weight: bold; color: black;">{recommendation}</p>
            <p style="text-align: center; margin: 0; font-size: 0.8em; color: black;">Adjusted for team consistency and matchup variance</p>
        </div>
        """, 
        unsafe_allow_html=True
    )
    
    # Plot the prediction distribution
    fig = plot_prediction_distribution(model_spread, matchup_variance, vegas_spread)
    st.pyplot(fig)
    
    # Explanation of the enhanced model
    with st.expander("How the Enhanced Variance Model Works"):
        st.write("""
        The Enhanced Variance Model improves confidence ratings by:
        
        1. **Team Consistency Analysis**: Measures how consistently each team performs (lower = more consistent)
        2. **Matchup-Specific Variance**: Calculates expected variance for this specific matchup
        3. **Bayesian Credible Intervals**: Shows the range where the true spread likely falls
        4. **Adjusted Confidence Thresholds**: Teams with consistent performance get higher confidence at smaller edges
        
        This helps identify situations where a smaller edge may still be significant if it involves highly consistent teams.
        """)

# For standalone testing
if __name__ == "__main__":
    print("Enhanced Variance Model module loaded.")
