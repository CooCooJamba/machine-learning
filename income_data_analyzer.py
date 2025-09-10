#!/usr/bin/env python
# coding: utf-8

# In[ ]:


"""
Data Analysis and Time Series Processing Module

Contains functions for sliding window operations and analysis of the UCI Adult dataset
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sliding_window(x_array, window_size, step=1):
    """
    Sliding window function to transform 1D time series into a matrix
    
    Parameters:
    -----------
    x_array : numpy.ndarray
        1D input data array
    window_size : int
        Width of the sliding window
    step : int, optional
        Step size for window movement (default: 1)
    
    Returns:
    --------
    result_matrix : numpy.ndarray
        Sliding window matrix of shape (n_windows, window_size)
    
    Examples:
    ---------
    >>> x = np.array([1, 2, 3, 4, 5])
    >>> sliding_window(x, window_size=3, step=1)
    array([[1, 2, 3],
           [2, 3, 4],
           [3, 4, 5]])
    """
    if len(x_array) < window_size:
        raise ValueError("Array length must be greater than or equal to window size")
    
    n_windows = (len(x_array) - window_size) // step + 1
    indices = np.arange(window_size) + np.arange(0, n_windows * step, step).reshape(-1, 1)
    
    return x_array[indices]


# Test sliding_window function
if __name__ == "__main__":
    # Test case 1
    window_size = 3
    step_size = 1
    x1 = np.array([8, 1, 4, 5, -2, 5, 9, 0])
    expected_A1 = np.array([[8, 1, 4],
                           [1, 4, 5],
                           [4, 5, -2],
                           [5, -2, 5],
                           [-2, 5, 9],
                           [5, 9, 0]])

    print("Test 1 passed:", np.array_equal(sliding_window(x1, window_size, step_size), expected_A1))

    # Test case 2
    window_size = 2
    step_size = 4
    x2 = np.array([8, 3, 4, 1, -6, 5, 9, 2, 10, 11, -14, 0])
    expected_A2 = np.array([[8, 3],
                           [-6, 5],
                           [10, 11]])

    print("Test 2 passed:", np.array_equal(sliding_window(x2, window_size, step_size), expected_A2))


class AdultDataAnalyzer:
    """
    Class for analyzing income data from the UCI Adult dataset
    
    Original material by: Yury Kashnitsky (@yorko in ODS Slack)
    License: Creative Commons CC BY-NC-SA 4.0
    """
    
    def __init__(self, data_url=None):
        """
        Initialize data analyzer
        
        Parameters:
        -----------
        data_url : str, optional
            URL for data download (default: standard UCI URL)
        """
        self.data_url = data_url or "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"
        self.data = None
        self.load_data()
    
    def load_data(self):
        """Load and preprocess data"""
        column_names = [
            "age", "workclass", "fnlwgt", "education", "education-num",
            "marital-status", "occupation", "relationship", "race", "sex",
            "capital-gain", "capital-loss", "hours-per-week", "native-country", "salary"
        ]
        
        self.data = pd.read_csv(
            self.data_url,
            sep=", ",
            header=None,
            names=column_names,
            engine='python',
            na_values='?'
        )
        
        # Convert string columns
        string_columns = self.data.select_dtypes(include=['object']).columns
        self.data[string_columns] = self.data[string_columns].astype("string")
        
        # Clean data
        self.data = self.data.dropna()
    
    def get_gender_distribution(self):
        """Get gender distribution with visualization"""
        gender_counts = self.data['sex'].value_counts()
        
        plt.figure(figsize=(8, 6))
        plt.bar(gender_counts.index, gender_counts.values, color=['lightblue', 'lightpink'])
        plt.title('Gender Distribution in Adult Dataset')
        plt.xlabel('Gender')
        plt.ylabel('Count')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        return gender_counts
    
    def get_average_age_by_gender(self, gender='Female'):
        """Calculate average age by gender"""
        return self.data[self.data['sex'] == gender]['age'].mean()
    
    def get_country_proportion(self, country='Germany'):
        """Calculate proportion of citizens from specific country"""
        country_count = (self.data['native-country'] == country).sum()
        total_count = len(self.data)
        return country_count / total_count
    
    def analyze_age_by_income(self):
        """Analyze age statistics by income level"""
        income_groups = self.data.groupby('salary')['age']
        
        means = income_groups.mean()
        stds = income_groups.std()
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        ax1.bar(means.index, means.values, color=['orange', 'green'])
        ax1.set_title('Average Age by Income Level')
        ax1.set_ylabel('Age')
        ax1.grid(axis='y', alpha=0.3)
        
        ax2.bar(stds.index, stds.values, color=['orange', 'green'])
        ax2.set_title('Age Standard Deviation by Income')
        ax2.set_ylabel('Standard Deviation')
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return means, stds
    
    def check_education_requirement(self):
        """Check if high earners have at least higher education"""
        higher_education = ['Bachelors', 'Prof-school', 'Assoc-acdm', 
                           'Assoc-voc', 'Masters', 'Doctorate']
        
        high_earners = self.data[self.data['salary'] == '>50K']
        non_higher_ed_high_earners = high_earners[~high_earners['education'].isin(higher_education)]
        
        return len(non_higher_ed_high_earners) == 0
    
    def get_age_statistics_by_demographics(self):
        """Get detailed age statistics by race and gender"""
        stats = self.data.groupby(['race', 'sex'])['age'].describe()
        return stats
    
    def analyze_income_by_marital_status(self):
        """Analyze income distribution by marital status for men"""
        male_data = self.data[self.data['sex'] == 'Male'].copy()
        
        married_statuses = ['Married-civ-spouse', 'Married-spouse-absent', 'Married-AF-spouse']
        male_data['is_married'] = male_data['marital-status'].isin(married_statuses)
        
        # Calculate proportions
        married_high_earners = len(male_data[male_data['is_married'] & (male_data['salary'] == '>50K')])
        single_high_earners = len(male_data[~male_data['is_married'] & (male_data['salary'] == '>50K')])
        
        married_total = len(male_data[male_data['is_married']])
        single_total = len(male_data[~male_data['is_married']])
        
        married_proportion = married_high_earners / married_total if married_total > 0 else 0
        single_proportion = single_high_earners / single_total if single_total > 0 else 0
        
        # Visualization
        plt.figure(figsize=(8, 6))
        plt.bar(['Single', 'Married'], [single_proportion, married_proportion], 
                color=['lightcoral', 'lightseagreen'])
        plt.title('Proportion of High Earners Among Men')
        plt.ylabel('Proportion with Income >50K')
        plt.grid(axis='y', alpha=0.3)
        plt.show()
        
        return {
            'married_proportion': married_proportion,
            'single_proportion': single_proportion,
            'married_count': married_high_earners,
            'single_count': single_high_earners
        }
    
    def analyze_working_hours(self):
        """Analyze maximum working hours and income distribution"""
        max_hours = self.data['hours-per-week'].max()
        max_hours_workers = self.data[self.data['hours-per-week'] == max_hours]
        
        total_max_hours = len(max_hours_workers)
        high_earners_max_hours = len(max_hours_workers[max_hours_workers['salary'] == '>50K'])
        proportion = high_earners_max_hours / total_max_hours if total_max_hours > 0 else 0
        
        return {
            'max_hours': max_hours,
            'total_workers': total_max_hours,
            'high_earners': high_earners_max_hours,
            'proportion': proportion
        }
    
    def analyze_working_hours_by_country(self):
        """Analyze average working hours by country and income level"""
        stats = self.data.groupby(['salary', 'native-country'])['hours-per-week'].mean().unstack(level=0)
        
        # Visualization
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
        
        stats['<=50K'].plot.bar(ax=ax1, color='orange', alpha=0.7)
        ax1.set_title('Average Working Hours (Income <=50K)')
        ax1.set_ylabel('Hours per Week')
        ax1.tick_params(axis='x', rotation=45)
        ax1.grid(axis='y', alpha=0.3)
        
        stats['>50K'].plot.bar(ax=ax2, color='green', alpha=0.7)
        ax2.set_title('Average Working Hours (Income >50K)')
        ax2.set_ylabel('Hours per Week')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return stats


# Example usage
if __name__ == "__main__":
    # Create analyzer instance
    analyzer = AdultDataAnalyzer()
    
    # Perform analyses
    print("1. Gender Distribution:")
    print(analyzer.get_gender_distribution())
    
    print("\n2. Average Age of Women:")
    print(f"{analyzer.get_average_age_by_gender('Female'):.2f} years")
    
    print("\n3. Proportion of German Citizens:")
    print(f"{analyzer.get_country_proportion('Germany'):.4f}")
    
    print("\n4-5. Age Statistics by Income Level:")
    means, stds = analyzer.analyze_age_by_income()
    print(f"Means: {means.to_dict()}")
    print(f"Standard Deviations: {stds.to_dict()}")

