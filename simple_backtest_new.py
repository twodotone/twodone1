"""
Simple Backtesting Module

Clean, rigorous backtesting for the simple NFL model.
Focuses on proper validation without data leakage.
"""

import pandas as pd
import numpy as np
import nfl_data_py as nfl
from simple_model import SimpleNFLModel, calculate_edge_and_confidence
from typing import List, Dict, Tuple
import warnings
warnings.filterwarnings('ignore')


class SimpleBacktester:
    """
    Rigorous backtesting for the simple NFL model.
    
    Key principles:
    1. Strict temporal splits (no future data leakage)
    2. Realistic betting scenarios (account for vig)
    3. Proper statistical validation
    4. Clear performance metrics
    """
    
    def __init__(self, vig: float = 0.1):
        """
        Initialize backtester.
        
        Args:
            vig: Sportsbook vig/juice (10% = 0.1)
        """
        self.vig = vig
        self.results = []
        
    def run_backtest(self, 
                    test_years: List[int],
                    training_years: List[int], 
                    weeks: List[int],
                    edge_thresholds: List[float] = [2.0, 3.0, 4.0, 5.0],
                    model_params: Dict = None) -> pd.DataFrame:
        """
        Run a comprehensive backtest.
        
        Args:
            test_years: Years to test on
            training_years: Years to train on (must be before test years)
            weeks: Weeks to include in testing
            edge_thresholds: Minimum edge sizes to test
            model_params: Model parameters override
            
        Returns:
            DataFrame with detailed results
        """
        print(f"Running backtest...")
        print(f"Training years: {training_years}")
        print(f"Test years: {test_years}")
        print(f"Weeks: {weeks[0]}-{weeks[-1]}")
        print(f"Edge thresholds: {edge_thresholds}")
        print("="*60)
        
        # Validate that training years come before test years
        if max(training_years) >= min(test_years):
            raise ValueError("Training years must come before test years")
            
        all_results = []
        
        for test_year in test_years:
            print(f"\nTesting {test_year}...")
            
            # Initialize model with training data only
            model = SimpleNFLModel()
            if model_params:
                for param, value in model_params.items():
                    setattr(model, param, value)
                    
            model.load_data(training_years)
            
            # Load test year schedule and results
            schedule = nfl.import_schedules([test_year])
            schedule = schedule[schedule['week'].isin(weeks)]
            
            year_results = []
            
            for _, game in schedule.iterrows():
                if pd.isna(game['result']):
                    continue  # Skip games without results
                    
                try:
                    # Get model prediction (using only training data)
                    pred_spread, details = model.predict_spread(
                        game['home_team'], 
                        game['away_team'],
                        game['week'],
                        test_year
                    )
                    
                    # Get Vegas spread (negative = home favored)
                    vegas_spread = self._get_vegas_spread(game)
                    if vegas_spread is None:
                        continue
                        
                    # Calculate edge and determine if we would bet
                    edge, confidence = calculate_edge_and_confidence(pred_spread, vegas_spread)
                    
                    # Test each edge threshold
                    for threshold in edge_thresholds:
                        if edge >= threshold:
                            # Determine our pick
                            if pred_spread < vegas_spread:
                                # Model likes home team more than Vegas
                                pick = game['home_team']
                                bet_spread = vegas_spread
                            else:
                                # Model likes away team more than Vegas
                                pick = game['away_team'] 
                                bet_spread = -vegas_spread
                                
                            # Determine actual outcome
                            actual_margin = game['result']  # Home team margin
                            
                            # Check if our pick covered
                            if pick == game['home_team']:
                                # We bet home team
                                covered = (actual_margin + bet_spread) > 0
                            else:
                                # We bet away team
                                covered = (actual_margin + bet_spread) < 0
                                
                            # Calculate profit/loss (account for vig)
                            if covered:
                                profit = 1.0 - self.vig  # Win $0.90 on $1 bet with 10% vig
                            else:
                                profit = -1.0  # Lose $1
                                
                            result = {
                                'year': test_year,
                                'week': game['week'],
                                'home_team': game['home_team'],
                                'away_team': game['away_team'],
                                'model_spread': pred_spread,
                                'vegas_spread': vegas_spread,
                                'edge': edge,
                                'threshold': threshold,
                                'confidence': confidence,
                                'pick': pick,
                                'actual_margin': actual_margin,
                                'covered': covered,
                                'profit': profit,
                                'bet_amount': 1.0  # Standard unit
                            }
                            
                            year_results.append(result)
                            
                except Exception as e:
                    print(f"Error processing game {game['home_team']} vs {game['away_team']}: {e}")
                    continue
                    
            all_results.extend(year_results)
            print(f"  Processed {len([r for r in year_results if r['threshold'] == edge_thresholds[0]])} games")
            
        results_df = pd.DataFrame(all_results)
        print(f"\nBacktest complete. Total bets across all thresholds: {len(results_df)}")
        
        return results_df
    
    def _get_vegas_spread(self, game) -> float:
        """Extract Vegas spread from game data."""
        try:
            # Use moneyline to determine favorite if available
            home_ml = getattr(game, 'home_moneyline', None)
            away_ml = getattr(game, 'away_moneyline', None)
            spread_magnitude = abs(getattr(game, 'spread_line', 0))
            
            if home_ml is not None and away_ml is not None:
                if home_ml < away_ml:  # Home team favored
                    return -spread_magnitude
                else:  # Away team favored
                    return spread_magnitude
            else:
                # Fallback to spread_line if moneylines not available
                return getattr(game, 'spread_line', None)
                
        except:
            return None
    
    def analyze_results(self, results_df: pd.DataFrame) -> Dict:
        """
        Analyze backtest results and provide comprehensive statistics.
        
        Args:
            results_df: Results from run_backtest()
            
        Returns:
            Dictionary with analysis results
        """
        if results_df.empty:
            return {"error": "No results to analyze"}
            
        analysis = {}
        
        # Overall performance by threshold
        threshold_analysis = {}
        
        for threshold in results_df['threshold'].unique():
            threshold_data = results_df[results_df['threshold'] == threshold]
            
            total_bets = len(threshold_data)
            wins = threshold_data['covered'].sum()
            total_profit = threshold_data['profit'].sum()
            
            if total_bets > 0:
                win_rate = wins / total_bets
                roi = total_profit / total_bets
                
                # Calculate confidence interval for win rate
                se = np.sqrt(win_rate * (1 - win_rate) / total_bets)
                ci_lower = win_rate - 1.96 * se
                ci_upper = win_rate + 1.96 * se
                
                threshold_analysis[threshold] = {
                    'total_bets': total_bets,
                    'wins': int(wins),
                    'losses': total_bets - int(wins),
                    'win_rate': win_rate,
                    'roi': roi,
                    'total_profit': total_profit,
                    'win_rate_ci_lower': ci_lower,
                    'win_rate_ci_upper': ci_upper,
                    'breakeven_rate': 0.5 + self.vig/2  # Rate needed to break even with vig
                }
                
        analysis['by_threshold'] = threshold_analysis
        
        # Performance by year
        year_analysis = {}
        for year in results_df['year'].unique():
            year_data = results_df[results_df['year'] == year]
            
            year_summary = {}
            for threshold in year_data['threshold'].unique():
                threshold_year_data = year_data[year_data['threshold'] == threshold]
                
                total_bets = len(threshold_year_data)
                wins = threshold_year_data['covered'].sum()
                
                if total_bets > 0:
                    year_summary[threshold] = {
                        'total_bets': total_bets,
                        'wins': int(wins),
                        'win_rate': wins / total_bets,
                        'roi': threshold_year_data['profit'].sum() / total_bets
                    }
                    
            year_analysis[year] = year_summary
            
        analysis['by_year'] = year_analysis
        
        # Edge size analysis (how well does larger edge predict success?)
        edge_bins = [0, 2, 3, 4, 5, 6, 10, 20]
        edge_analysis = {}
        
        for i in range(len(edge_bins)-1):
            bin_data = results_df[
                (results_df['edge'] >= edge_bins[i]) & 
                (results_df['edge'] < edge_bins[i+1])
            ]
            
            if len(bin_data) > 0:
                wins = bin_data['covered'].sum()
                total = len(bin_data)
                
                edge_analysis[f"{edge_bins[i]}-{edge_bins[i+1]}"] = {
                    'total_bets': total,
                    'wins': int(wins),
                    'win_rate': wins / total,
                    'avg_edge': bin_data['edge'].mean(),
                    'roi': bin_data['profit'].sum() / total
                }
                
        analysis['by_edge_size'] = edge_analysis
        
        return analysis
    
    def print_summary(self, analysis: Dict) -> None:
        """Print a formatted summary of backtest results."""
        print("\n" + "="*80)
        print("BACKTEST RESULTS SUMMARY")
        print("="*80)
        
        if 'error' in analysis:
            print(f"Error: {analysis['error']}")
            return
            
        # Performance by threshold
        print("\nPERFORMANCE BY EDGE THRESHOLD:")
        print("-" * 50)
        print(f"{'Threshold':<10} {'Bets':<6} {'W-L':<8} {'Win%':<8} {'ROI%':<8} {'Profit':<8}")
        print("-" * 50)
        
        for threshold, stats in analysis['by_threshold'].items():
            win_rate_pct = stats['win_rate'] * 100
            roi_pct = stats['roi'] * 100
            
            # Color coding for performance
            performance_marker = ""
            if stats['win_rate'] > stats['breakeven_rate']:
                performance_marker = " ✓"
            elif stats['win_rate'] < 0.5:
                performance_marker = " ✗"
                
            print(f"{threshold:<10} {stats['total_bets']:<6} "
                  f"{stats['wins']}-{stats['losses']:<4} "
                  f"{win_rate_pct:<7.1f} {roi_pct:<7.1f} "
                  f"{stats['total_profit']:<7.1f}{performance_marker}")
                  
        # Statistical significance note
        print(f"\nNote: Need >{analysis['by_threshold'][2.0]['breakeven_rate']*100:.1f}% win rate to beat {self.vig*100:.0f}% vig")
        
        # Performance by year
        print("\nPERFORMANCE BY YEAR (3+ point edges):")
        print("-" * 40)
        for year, year_stats in analysis['by_year'].items():
            if 3.0 in year_stats:
                stats = year_stats[3.0]
                win_rate_pct = stats['win_rate'] * 100
                roi_pct = stats['roi'] * 100
                print(f"{year}: {stats['wins']}-{stats['total_bets']-stats['wins']} "
                      f"({win_rate_pct:.1f}%, ROI: {roi_pct:+.1f}%)")
                      
        # Edge size effectiveness
        print("\nEDGE SIZE EFFECTIVENESS:")
        print("-" * 50)
        print(f"{'Edge Range':<12} {'Bets':<6} {'Win%':<8} {'Avg Edge':<10} {'ROI%':<8}")
        print("-" * 50)
        
        for edge_range, stats in analysis['by_edge_size'].items():
            win_rate_pct = stats['win_rate'] * 100
            roi_pct = stats['roi'] * 100
            avg_edge = stats['avg_edge']
            
            print(f"{edge_range:<12} {stats['total_bets']:<6} "
                  f"{win_rate_pct:<7.1f} {avg_edge:<9.1f} {roi_pct:<7.1f}")


def run_simple_backtest():
    """Run a basic backtest example."""
    backtester = SimpleBacktester(vig=0.1)  # 10% vig
    
    # Test 2024 using 2022-2023 training data
    results = backtester.run_backtest(
        test_years=[2024],
        training_years=[2022, 2023],
        weeks=list(range(4, 18)),  # Weeks 4-17
        edge_thresholds=[2.0, 3.0, 4.0, 5.0]
    )
    
    # Analyze results
    analysis = backtester.analyze_results(results)
    backtester.print_summary(analysis)
    
    return results, analysis


if __name__ == "__main__":
    # Run example backtest
    results, analysis = run_simple_backtest()
