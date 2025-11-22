#!/usr/bin/env python3
"""
Survey Data Analysis Script
Generates descriptive statistics and visualizations for survey responses
"""

import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import numpy as np
import textwrap

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_survey_data(filepath):
    """Load survey data from JSON file"""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data

def identify_question_type(question_data):
    """Identify if question is multiple choice, select-all, or free text"""
    question_text = question_data['text'].lower()
    
    # Skip consent question
    if 'consent' in question_text:
        return 'skip'
    
    # Free text questions to skip
    if any(phrase in question_text for phrase in [
        'briefly describe', 
        'if you selected "other"',
        'if you could magically conjure'
    ]):
        return 'free_text'
    
    # Select all that apply questions
    if 'select all that apply' in question_text:
        return 'select_all'
    
    # Everything else is multiple choice
    return 'multiple_choice'

def process_select_all_question(answers):
    """Process select-all-that-apply questions"""
    all_options = []
    for answer in answers:
        if answer and answer.lower() not in ['n/a', 'na', 'none']:
            options = answer.split(';')
            all_options.extend([opt.strip() for opt in options])
    return Counter(all_options)

def process_multiple_choice_question(answers):
    """Process multiple choice questions"""
    cleaned_answers = [ans for ans in answers if ans and ans.lower() not in ['n/a', 'na']]
    return Counter(cleaned_answers)

def wrap_labels(labels, width=15):
    """Wrap long labels for better display"""
    return ['\n'.join(textwrap.wrap(label, width)) for label in labels]

def get_custom_order(question_text):
    """Return custom ordering for specific question types"""
    question_lower = question_text.lower()
    
    # Trip/travel frequency questions
    if 'how many trips' in question_lower or 'how many times' in question_lower:
        # Check which format is being used by looking at the question text
        if 'how many times' in question_lower or 'away from your home city' in question_lower:
            # This question uses "times"
            return ["0 times", "1-2 times", "3-5 times", "6-10 times", "11+ times"]
        else:
            # These questions use "trips"
            return ["0 trips", "1-2 trips", "3-5 trips", "6-10 trips", "11+ trips"]
    
    # Tool usage consistency questions
    if 'same basic tools/processes' in question_lower:
        return ["I use completely different tools/services/processes for different trips",
                "I mostly use different tools/services/processes for different trips", 
                "Not really sure/It depends",
                "I mostly use the same tools/services/processes for both kinds of trips",
                "I use almost exactly the same tools/services/processes for both kinds of trips"]
    
    return None

def order_data_for_chart(data, question_text, include_zeros=False):
    """Order data according to custom rules or by frequency"""
    custom_order = get_custom_order(question_text)
    
    if custom_order:
        # Use custom ordering
        ordered_data = {}
        
        # For frequency questions, include all categories even if zero
        question_lower = question_text.lower()
        if include_zeros and ('how many trips' in question_lower or 'how many times' in question_lower):
            # Add all categories with zero counts first
            for item in custom_order:
                ordered_data[item] = data.get(item, 0)
        else:
            # Only include categories that exist in data (original behavior)
            for item in custom_order:
                if item in data:
                    ordered_data[item] = data[item]
            # Add any items not in custom order at the end
            for item, count in data.items():
                if item not in ordered_data:
                    ordered_data[item] = count
        
        return ordered_data
    else:
        # Default: sort by frequency (descending)
        return dict(sorted(data.items(), key=lambda x: x[1], reverse=True))

def create_bar_chart(data, title, xlabel, ylabel, question_text="", figsize=(12, 6), rotation=45, wrap_width=20):
    """Create a bar chart for categorical data"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Order data based on question type - include zeros for frequency questions in bar charts
    sorted_data = order_data_for_chart(data, question_text, include_zeros=True)
    
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    
    # Wrap long labels
    wrapped_labels = wrap_labels(labels, wrap_width)
    
    # Create bar chart
    bars = ax.bar(range(len(labels)), values)
    
    # Color bars with gradient - but make zero-value bars a lighter color
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(bars)))
    for bar, color, value in zip(bars, colors, values):
        if value == 0:
            bar.set_color('lightgray')  # Make zero bars gray
            bar.set_edgecolor('darkgray')  # Add edge for visibility
            bar.set_linewidth(1)
        else:
            bar.set_color(color)
    
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(wrapped_labels, rotation=rotation, ha='right')
    
    # Add value labels on bars (including zeros)
    for i, (bar, value) in enumerate(zip(bars, values)):
        height = bar.get_height()
        if value > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value}', ha='center', va='bottom', fontsize=10)
        else:
            # For zero values, place the "0" just above the x-axis
            ax.text(bar.get_x() + bar.get_width()/2., 0.1,
                    '0', ha='center', va='bottom', fontsize=10, color='darkgray')
    
    # Add grid for better readability
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    
    plt.tight_layout()
    return fig

def create_pie_chart(data, title, question_text="", figsize=(10, 8)):
    """Create a pie chart for categorical data"""
    fig, ax = plt.subplots(figsize=figsize)
    
    # Order data based on question type - DO NOT include zeros for pie charts
    sorted_data = order_data_for_chart(data, question_text, include_zeros=False)
    
    if len(sorted_data) > 8:
        # Group smaller categories as "Other"
        top_items = dict(list(sorted_data.items())[:7])
        other_count = sum(list(sorted_data.values())[7:])
        if other_count > 0:
            top_items['Other Categories'] = other_count
        sorted_data = top_items
    
    labels = list(sorted_data.keys())
    values = list(sorted_data.values())
    
    # Create pie chart with better styling
    colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
    wedges, texts, autotexts = ax.pie(values, labels=labels, autopct='%1.1f%%',
                                       colors=colors, startangle=90,
                                       pctdistance=0.85)
    
    # Beautify the text
    for text in texts:
        text.set_fontsize(10)
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontweight('bold')
        autotext.set_fontsize(10)
    
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    return fig

def analyze_survey(data):
    """Main function to analyze survey data"""
    results = {}
    figures = []
    chart_number = 1
    
    print("=" * 80)
    print("SURVEY DATA ANALYSIS")
    print("=" * 80)
    print(f"\nTotal respondents: {len(data[0]['answers'])}\n")
    
    for i, question_data in enumerate(data):
        question_type = identify_question_type(question_data)
        
        if question_type == 'skip' or question_type == 'free_text':
            continue
        
        question_text = question_data['text']
        question_id = question_data['id']
        answers = question_data['answers']
        
        print(f"\nQuestion {i+1} (ID: {question_id})")
        print("-" * 60)
        print(f"Question: {question_text[:100]}..." if len(question_text) > 100 else f"Question: {question_text}")
        print(f"Type: {question_type.replace('_', ' ').title()}")
        
        if question_type == 'select_all':
            counter = process_select_all_question(answers)
        else:
            counter = process_multiple_choice_question(answers)
        
        if counter:
            # Print statistics
            print(f"Total responses: {len([a for a in answers if a and a.lower() not in ['n/a', 'na', 'none']])}")
            print(f"Unique options: {len(counter)}")
            print("\nTop responses:")
            for option, count in counter.most_common(5):
                percentage = (count / len(answers)) * 100
                print(f"  - {option[:50]}{'...' if len(option) > 50 else ''}: {count} ({percentage:.1f}%)")
            
            # Create visualizations
            short_title = question_text[:60] + "..." if len(question_text) > 60 else question_text
            
            # Bar chart for all questions
            fig_bar = create_bar_chart(
                counter, 
                f"Q{i+1}: {short_title}",
                "Response Options",
                "Frequency",
                question_text=question_text,  # Pass the full question text for ordering
                figsize=(14, 7),
                rotation=45,
                wrap_width=25
            )
            # Save the bar chart
            fig_bar.savefig(f'chart_{chart_number:02d}_q{i+1}_bar.png', dpi=100, bbox_inches='tight')
            print(f"  Saved: chart_{chart_number:02d}_q{i+1}_bar.png")
            chart_number += 1
            figures.append(fig_bar)
            plt.close(fig_bar)
            
            # Pie chart for multiple choice with fewer options
            if question_type == 'multiple_choice' and len(counter) <= 10:
                fig_pie = create_pie_chart(
                    counter,
                    f"Q{i+1}: {short_title}",
                    question_text=question_text,  # Pass the full question text for ordering
                    figsize=(10, 8)
                )
                # Save the pie chart
                fig_pie.savefig(f'chart_{chart_number:02d}_q{i+1}_pie.png', dpi=100, bbox_inches='tight')
                print(f"  Saved: chart_{chart_number:02d}_q{i+1}_pie.png")
                chart_number += 1
                figures.append(fig_pie)
                plt.close(fig_pie)
            
            results[question_id] = {
                'question': question_text,
                'type': question_type,
                'statistics': counter,
                'total_responses': len([a for a in answers if a and a.lower() not in ['n/a', 'na', 'none']])
            }
    
    return results, figures

def save_results(results, output_file='survey_results.json'):
    """Save analysis results to JSON file"""
    # Convert Counter objects to dict for JSON serialization
    json_results = {}
    for key, value in results.items():
        json_results[key] = {
            'question': value['question'],
            'type': value['type'],
            'statistics': dict(value['statistics']),
            'total_responses': value['total_responses']
        }
    
    with open(output_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\nResults saved to {output_file}")

def main():
    """Main execution function"""
    # Load data
    data = load_survey_data('survey_data.json')
    
    # Analyze survey
    results, figures = analyze_survey(data)
    
    # Save results
    save_results(results)
    
    print("\n" + "=" * 80)
    print("Analysis complete! Charts have been saved as PNG files.")
    print("Numerical results have been saved to 'survey_results.json'")
    print("=" * 80)

if __name__ == "__main__":
    main()
