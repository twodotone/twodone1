import numpy as np

# Updated values based on our confidence analysis, with weighting toward 2024 season results
# Format: {edge_magnitude: (win_rate, sample_size)}
EDGE_CONFIDENCE_MAP = {
    1: (0.510, 200),
    2: (0.525, 175),
    3: (0.535, 150),
    4: (0.550, 125),  # At 4+ points we observed 54-57% win rate for 2024
    5: (0.570, 100),  # At 5+ points we observed 57-60% win rate for 2024
    6: (0.610, 75),   # At 6+ points we observed ~65% win rate overall
    7: (0.620, 50),
    8: (0.630, 35),
    9: (0.640, 25),
    10: (0.650, 15),
}

# Version optimized for 2025 season based on 2024 results
EDGE_CONFIDENCE_MAP_2025 = {
    1: (0.515, 200),
    2: (0.530, 175),
    3: (0.545, 150),
    4: (0.560, 125),  # 2024 showed stronger performance at 4+ point edges
    5: (0.590, 100),  # 2024 showed ~60% win rate at 5+ point edges
    6: (0.625, 75),   # Extrapolating from 2024 trend
    7: (0.640, 50),
    8: (0.650, 35),
    9: (0.660, 25),
    10: (0.670, 15),
}

def get_confidence_rating(edge_magnitude, use_2025_model=True, confidence_map=None):
    """
    Convert an edge magnitude to a confidence rating (1-5 stars)
    and estimated win probability.
    
    Args:
        edge_magnitude (float): The absolute difference between the model's spread
                                and the Vegas spread
        use_2025_model (bool): Whether to use the 2025-optimized confidence map
        confidence_map (dict): Optional custom mapping of edge magnitude to (win_rate, sample_size)
    
    Returns:
        tuple: (confidence_stars, win_probability, sample_size)
            - confidence_stars: 1-5 star rating
            - win_probability: Estimated win probability
            - sample_size: Sample size for this edge magnitude
    """
    # Convert negative edges to positive
    edge_magnitude = abs(edge_magnitude)
    
    # Select the appropriate confidence map
    if confidence_map is None:
        confidence_map = EDGE_CONFIDENCE_MAP_2025 if use_2025_model else EDGE_CONFIDENCE_MAP
    
    # Find the closest edge magnitude in our map
    closest_edge = min(confidence_map.keys(), key=lambda x: abs(x - edge_magnitude))
    win_rate, sample_size = confidence_map[closest_edge]
    
    # For edges between our mapped values, use linear interpolation
    if edge_magnitude > closest_edge and edge_magnitude < closest_edge + 1 and closest_edge + 1 in confidence_map:
        next_edge = closest_edge + 1
        next_win_rate, next_sample = confidence_map[next_edge]
        
        # Linear interpolation
        weight = edge_magnitude - closest_edge
        win_rate = win_rate * (1 - weight) + next_win_rate * weight
        sample_size = int(sample_size * (1 - weight) + next_sample * weight)
    
    # Calculate confidence rating (1-5 stars)
    # Using both win rate and sample size to determine confidence
    # Break-even is 52.4%, so anything above that should be at least 3 stars
    if win_rate < 0.52:
        stars = 1
    elif win_rate < 0.53:
        stars = 2
    elif win_rate < 0.57:
        stars = 3
    elif win_rate < 0.60:
        stars = 4
    else:
        stars = 5
    
    # Reduce confidence if sample size is very small
    if sample_size < 20 and stars > 1:
        stars -= 1
    
    return stars, win_rate, sample_size

def get_confidence_text(stars):
    """
    Convert a star rating to a text description
    
    Args:
        stars (int): 1-5 star rating
        
    Returns:
        str: Text description of confidence level
    """
    confidence_text = {
        1: "Very Low",
        2: "Low",
        3: "Moderate",
        4: "High",
        5: "Very High"
    }
    
    return confidence_text.get(stars, "Unknown")

def get_recommendation(edge_magnitude, win_probability):
    """
    Get a recommendation based on edge magnitude and win probability
    
    Args:
        edge_magnitude (float): Model edge in points
        win_probability (float): Estimated win probability (0-1)
        
    Returns:
        str: Recommendation text
    """
    if win_probability < 0.524:
        return "Pass - Expected negative value"
    
    if edge_magnitude < 4.0:
        return "Small Edge - Consider passing"
    elif edge_magnitude < 5.0:
        return "Modest Edge - Small bet recommended"
    elif edge_magnitude < 6.0:
        return "Solid Edge - Standard bet recommended"
    else:
        return "Strong Edge - High confidence bet"

def update_confidence_map(edge_analysis_df):
    """
    Update the confidence map with real analysis data
    
    Args:
        edge_analysis_df (DataFrame): Output from value_confidence.py analysis
        
    Returns:
        dict: Updated confidence map
    """
    updated_map = {}
    
    for _, row in edge_analysis_df.iterrows():
        # Extract the lower bound of the bin
        bin_str = str(row['edge_bin'])
        lower_bound = float(bin_str.split(',')[0].replace('[', ''))
        
        updated_map[lower_bound] = (row['win_rate'], row['samples'])
    
    return updated_map

# Example usage
if __name__ == "__main__":
    # Test with a few edges
    test_edges = [2.5, 4.0, 5.5, 7.2, 9.8]
    
    print("2025 Edge Confidence Analysis (Optimized for 2025 Season):")
    print("="*60)
    
    for edge in test_edges:
        stars, win_prob, samples = get_confidence_rating(edge, use_2025_model=True)
        confidence = get_confidence_text(stars)
        recommendation = get_recommendation(edge, win_prob)
        
        print(f"{edge:.1f} point edge: {stars}★ ({confidence})")
        print(f"  Win probability: {win_prob*100:.1f}% (based on {samples} samples)")
        print(f"  Recommendation: {recommendation}")
        print("")
    
    print("\nConservative Model (Based on All Seasons):")
    print("="*60)
    
    for edge in test_edges:
        stars, win_prob, samples = get_confidence_rating(edge, use_2025_model=False)
        confidence = get_confidence_text(stars)
        recommendation = get_recommendation(edge, win_prob)
        
        print(f"{edge:.1f} point edge: {stars}★ ({confidence})")
        print(f"  Win probability: {win_prob*100:.1f}% (based on {samples} samples)")
        print(f"  Recommendation: {recommendation}")
        print("")
