import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import chi2_contingency, ks_2samp
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_and_prepare_data(filepath):
    """Load JSON data and convert to DataFrame"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    # Convert to DataFrame
    df_list = []
    for question in data:
        df_temp = pd.DataFrame({
            'question_id': question['id'],
            'question_text': question['text'],
            'answer': question['answers']
        })
        df_temp['respondent_id'] = range(len(question['answers']))
        df_list.append(df_temp)
    
    df = pd.concat(df_list, ignore_index=True)
    return df, data

def identify_likert_questions(df):
    """Identify Likert scale questions for each prototype"""
    likert_questions = {
        'A': [
            ('1763429742172', 'Understand Tasks'),
            ('1763429933708', 'Understand How to Use'),
            ('1763429972190', 'Easy to Use'),
            ('1763430192770', 'Has Features Needed'),
            ('1763430096428', 'Superior to Existing Tools')
        ],
        'B': [
            ('1763431473322', 'Understand Tasks'),
            ('1763431486831', 'Understand How to Use'),
            ('1763431509699', 'Easy to Use'),
            ('1763431525649', 'Has Features Needed'),
            ('1763431567705', 'Superior to Existing Tools')
        ],
        'C': [
            ('1763433215212', 'Understand Tasks'),
            ('1763433230680', 'Understand How to Use'),
            ('1763433260707', 'Easy to Use'),
            ('1763433293713', 'Has Features Needed'),
            ('1763433312274', 'Superior to Existing Tools')
        ]
    }
    return likert_questions

def get_likert_data(df, likert_questions):
    """Extract and organize Likert scale data"""
    likert_data = {}
    
    for prototype, questions in likert_questions.items():
        prototype_data = {}
        for q_id, q_label in questions:
            # Get answers for this question
            answers = df[df['question_id'] == q_id]['answer'].values
            # Convert to numeric
            numeric_answers = [int(a) for a in answers if a.isdigit()]
            prototype_data[q_label] = numeric_answers
        likert_data[prototype] = prototype_data
    
    return likert_data

def calculate_descriptive_stats(likert_data):
    """Calculate descriptive statistics for each prototype and question"""
    stats_results = {}
    
    for prototype, questions in likert_data.items():
        prototype_stats = {}
        for question, values in questions.items():
            prototype_stats[question] = {
                'mean': np.mean(values),
                'median': np.median(values),
                'std': np.std(values, ddof=1),
                'min': np.min(values),
                'max': np.max(values),
                'n': len(values)
            }
        stats_results[prototype] = prototype_stats
    
    return stats_results

def create_bar_charts(likert_data, stats_results):
    """Create bar charts for comparing prototypes"""
    questions = list(next(iter(likert_data.values())).keys())
    n_questions = len(questions)
    
    # Create subplots for mean scores
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, question in enumerate(questions):
        means = [stats_results[p][question]['mean'] for p in ['A', 'B', 'C']]
        stds = [stats_results[p][question]['std'] for p in ['A', 'B', 'C']]
        
        ax = axes[idx]
        bars = ax.bar(['Prototype A', 'Prototype B', 'Prototype C'], means, 
                      yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel('Mean Score (1-5)')
        ax.set_title(f'{question}')
        ax.set_ylim(0, 5.5)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for bar, mean in zip(bars, means):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                   f'{mean:.2f}', ha='center', va='bottom')
    
    # Remove extra subplot if any
    if idx < len(axes) - 1:
        fig.delaxes(axes[-1])
    
    plt.suptitle('Mean Scores Comparison Across Prototypes', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./mean_scores_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create distribution plots
    fig, axes = plt.subplots(n_questions, 3, figsize=(15, n_questions * 3))
    
    for q_idx, question in enumerate(questions):
        for p_idx, prototype in enumerate(['A', 'B', 'C']):
            ax = axes[q_idx, p_idx] if n_questions > 1 else axes[p_idx]
            
            data = likert_data[prototype][question]
            # Create histogram with counts
            counts = [data.count(i) for i in range(1, 6)]
            bars = ax.bar(range(1, 6), counts, color=f'C{p_idx}')
            
            ax.set_xlabel('Rating')
            ax.set_ylabel('Frequency')
            ax.set_title(f'Prototype {prototype}: {question}')
            ax.set_xticks(range(1, 6))
            ax.set_xlim(0.5, 5.5)
            
            # Add value labels on bars
            for bar, count in zip(bars, counts):
                if count > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., count + 0.1,
                           str(count), ha='center', va='bottom')
    
    plt.suptitle('Response Distribution for Each Prototype and Question', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./distribution_plots.png', dpi=300, bbox_inches='tight')
    plt.show()

def perform_chi_squared_omnibus(likert_data):
    """Perform chi-squared omnibus test for each question across all prototypes"""
    questions = list(next(iter(likert_data.values())).keys())
    omnibus_results = {}
    
    for question in questions:
        # Create contingency table
        # Rows: Prototypes, Columns: Ratings (1-5)
        contingency_table = []
        
        for prototype in ['A', 'B', 'C']:
            data = likert_data[prototype][question]
            row = [data.count(i) for i in range(1, 6)]
            contingency_table.append(row)
        
        contingency_table = np.array(contingency_table)
        
        # Check if we have any empty columns (ratings not used)
        # If so, we'll remove them for the test
        non_zero_cols = contingency_table.sum(axis=0) > 0
        filtered_table = contingency_table[:, non_zero_cols]
        
        # Only perform test if we have variation
        if filtered_table.size > 0 and filtered_table.sum() > 0:
            try:
                chi2, p_value, dof, expected = chi2_contingency(filtered_table)
                
                omnibus_results[question] = {
                    'chi2': chi2,
                    'p_value': p_value,
                    'dof': dof,
                    'significant': p_value < 0.05,
                    'contingency_table': contingency_table
                }
            except ValueError:
                # If chi-squared test fails, use alternative approach
                omnibus_results[question] = {
                    'chi2': np.nan,
                    'p_value': 1.0,
                    'dof': 0,
                    'significant': False,
                    'contingency_table': contingency_table,
                    'note': 'Chi-squared test not applicable due to sparse data'
                }
        else:
            omnibus_results[question] = {
                'chi2': np.nan,
                'p_value': 1.0,
                'dof': 0,
                'significant': False,
                'contingency_table': contingency_table,
                'note': 'No variation in data'
            }
    
    return omnibus_results

def perform_pairwise_ks_tests(likert_data, omnibus_results):
    """Perform pairwise Kolmogorov-Smirnov tests for significant omnibus results"""
    pairwise_results = {}
    
    for question, omnibus in omnibus_results.items():
        if omnibus['significant']:
            pairs = [('A', 'B'), ('A', 'C'), ('B', 'C')]
            pairwise_results[question] = {}
            
            for p1, p2 in pairs:
                data1 = likert_data[p1][question]
                data2 = likert_data[p2][question]
                
                # Perform KS test
                ks_stat, p_value = ks_2samp(data1, data2)
                
                # Apply Bonferroni correction (3 comparisons)
                adjusted_p = p_value * 3
                
                pairwise_results[question][f'{p1} vs {p2}'] = {
                    'ks_statistic': ks_stat,
                    'p_value': p_value,
                    'adjusted_p_value': min(adjusted_p, 1.0),
                    'significant': adjusted_p < 0.05
                }
    
    return pairwise_results

def analyze_ranking_questions(df):
    """Analyze the ranking questions (BEST, SECOND BEST, WORST)"""
    ranking_questions = {
        '1763433609862': 'BEST',
        '1763433957657': 'SECOND BEST',
        '1763434721787': 'WORST'
    }
    
    ranking_results = {}
    
    for q_id, label in ranking_questions.items():
        # Get answers
        answers = df[df['question_id'] == q_id]['answer'].values
        
        # Count frequencies
        prototype_counts = {
            'Prototype A': sum(1 for a in answers if 'Prototype A' in a),
            'Prototype B': sum(1 for a in answers if 'Prototype B' in a),
            'Prototype C': sum(1 for a in answers if 'Prototype C' in a)
        }
        
        # Expected frequencies (equal distribution)
        n_total = sum(prototype_counts.values())
        expected = n_total / 3
        
        # Chi-squared test
        observed = list(prototype_counts.values())
        chi2, p_value = stats.chisquare(observed, [expected] * 3)
        
        ranking_results[label] = {
            'counts': prototype_counts,
            'chi2': chi2,
            'p_value': p_value,
            'significant': p_value < 0.05,
            'total_responses': n_total
        }
    
    return ranking_results

def create_ranking_visualization(ranking_results):
    """Create visualization for ranking results"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, (label, results) in enumerate(ranking_results.items()):
        ax = axes[idx]
        
        prototypes = list(results['counts'].keys())
        counts = list(results['counts'].values())
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        bars = ax.bar(prototypes, counts, color=colors)
        
        ax.set_ylabel('Number of Votes')
        ax.set_title(f'{label} Interface\n(p={results["p_value"]:.3f}{"*" if results["significant"] else ""})')
        ax.set_ylim(0, max(counts) * 1.2)
        
        # Add value labels on bars
        for bar, count in zip(bars, counts):
            ax.text(bar.get_x() + bar.get_width()/2., count + 0.5,
                   str(count), ha='center', va='bottom', fontweight='bold')
        
        # Add horizontal line for expected value
        expected = results['total_responses'] / 3
        ax.axhline(y=expected, color='red', linestyle='--', alpha=0.5, label='Expected (equal)')
        ax.legend()
    
    plt.suptitle('Prototype Rankings Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('./ranking_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_results(stats_results, omnibus_results, pairwise_results, ranking_results):
    """Print formatted results"""
    print("=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    
    for prototype in ['A', 'B', 'C']:
        print(f"\n--- Prototype {prototype} ---")
        for question, stats in stats_results[prototype].items():
            print(f"\n{question}:")
            print(f"  Mean: {stats['mean']:.2f} (SD: {stats['std']:.2f})")
            print(f"  Median: {stats['median']:.1f}")
            print(f"  Range: {stats['min']}-{stats['max']}")
            print(f"  N: {stats['n']}")
    
    print("\n" + "=" * 80)
    print("CHI-SQUARED OMNIBUS TESTS (Across All Three Prototypes)")
    print("=" * 80)
    
    for question, results in omnibus_results.items():
        print(f"\n{question}:")
        if 'note' in results:
            print(f"  {results['note']}")
        else:
            print(f"  χ² = {results['chi2']:.3f}, p = {results['p_value']:.4f}")
            print(f"  Significant: {'YES' if results['significant'] else 'NO'}")
            if results['significant']:
                print("  → Proceeding with pairwise comparisons")
    
    print("\n" + "=" * 80)
    print("PAIRWISE KOLMOGOROV-SMIRNOV TESTS (Bonferroni Corrected)")
    print("=" * 80)
    
    for question, pairs in pairwise_results.items():
        print(f"\n{question}:")
        for pair, results in pairs.items():
            print(f"  {pair}:")
            print(f"    KS statistic: {results['ks_statistic']:.3f}")
            print(f"    p-value (raw): {results['p_value']:.4f}")
            print(f"    p-value (adjusted): {results['adjusted_p_value']:.4f}")
            print(f"    Significant: {'YES' if results['significant'] else 'NO'}")
    
    print("\n" + "=" * 80)
    print("RANKING QUESTIONS ANALYSIS")
    print("=" * 80)
    
    for label, results in ranking_results.items():
        print(f"\n{label}:")
        for prototype, count in results['counts'].items():
            print(f"  {prototype}: {count} votes")
        print(f"  χ² = {results['chi2']:.3f}, p = {results['p_value']:.4f}")
        print(f"  Significant: {'YES' if results['significant'] else 'NO'}")

def create_summary_table(stats_results):
    """Create a summary table of mean scores"""
    questions = list(next(iter(stats_results.values())).keys())
    
    summary_data = []
    for question in questions:
        row = {'Question': question}
        for prototype in ['A', 'B', 'C']:
            mean = stats_results[prototype][question]['mean']
            std = stats_results[prototype][question]['std']
            row[f'Prototype {prototype}'] = f"{mean:.2f} ± {std:.2f}"
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    
    # Create a nice looking table
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=summary_df.values,
                    colLabels=summary_df.columns,
                    cellLoc='center',
                    loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Style the header
    for i in range(len(summary_df.columns)):
        table[(0, i)].set_facecolor('#40466e')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(summary_df) + 1):
        for j in range(len(summary_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    plt.title('Summary of Mean Scores ± Standard Deviation', fontsize=14, fontweight='bold')
    plt.savefig('./summary_table.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return summary_df

def main():
    # Load data
    print("Loading data...")
    df, raw_data = load_and_prepare_data('./survey_data.json')
    
    # Identify and extract Likert scale questions
    print("Processing Likert scale questions...")
    likert_questions = identify_likert_questions(df)
    likert_data = get_likert_data(df, likert_questions)
    
    # Calculate descriptive statistics
    print("Calculating descriptive statistics...")
    stats_results = calculate_descriptive_stats(likert_data)
    
    # Create visualizations
    print("Creating visualizations...")
    create_bar_charts(likert_data, stats_results)
    
    # Perform statistical tests
    print("Performing chi-squared omnibus tests...")
    omnibus_results = perform_chi_squared_omnibus(likert_data)
    
    print("Performing pairwise Kolmogorov-Smirnov tests...")
    pairwise_results = perform_pairwise_ks_tests(likert_data, omnibus_results)
    
    # Analyze ranking questions
    print("Analyzing ranking questions...")
    ranking_results = analyze_ranking_questions(df)
    create_ranking_visualization(ranking_results)
    
    # Create summary table
    print("Creating summary table...")
    summary_df = create_summary_table(stats_results)
    
    # Print all results
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE - RESULTS SUMMARY")
    print_results(stats_results, omnibus_results, pairwise_results, ranking_results)
    
    # Save results to file
    print("\nSaving results to file...")
    with open('./analysis_results.txt', 'w') as f:
        import sys
        original_stdout = sys.stdout
        sys.stdout = f
        print_results(stats_results, omnibus_results, pairwise_results, ranking_results)
        sys.stdout = original_stdout
    
    print("\nAnalysis complete! Check the generated plots and analysis_results.txt for detailed results.")
    
    return stats_results, omnibus_results, pairwise_results, ranking_results

