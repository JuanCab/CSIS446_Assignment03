"""
ID3 Decision Tree Algorithm Implementation with Continuous Feature Support

This module implements the ID3 (Iterative Dichotomiser 3) decision tree
algorithm for classification tasks with support for both categorical and 
continuous features. For continuous features, it finds optimal split points 
using the method described in the textbook.

The ID3 algorithm builds decision trees by recursively selecting the feature 
that provides the highest information gain at each node.

The algorithm works by:
1. Calculating entropy for the current dataset
2. For each feature, calculating information gain (or alternative criteria)
3. Selecting the feature with maximum information gain (or alternative)
4. Creating branches for each possible value of the selected feature
5. Recursively applying the process to each branch

This enhanced version supports:
- Three splitting criteria: Information Gain (default), Information Gain Ratio, Gini Index
- Both categorical and continuous features
- Automatic continuous feature detection
- Comprehensive tree visualization

Author: Juan Cabanela (with significant aid of Claude Sonnet 4)
Date: January 2026
"""

import numpy as np
from collections import Counter


class ID3:
    """
    ID3 Decision Tree Classifier with multiple splitting criteria and continuous feature support.

    The ID3 algorithm is a classic decision tree learning algorithm that
    builds decision trees by selecting attributes that maximize a splitting
    criterion. This enhanced version supports both categorical and continuous features.

    Key characteristics:
    - Uses categorical and continuous features
    - Greedy algorithm (makes locally optimal choices)
    - Can overfit without proper stopping criteria
    - Creates multiway splits for categorical features, binary splits for continuous

    Attributes:
        tree (dict): The root node of the decision tree structure
        max_depth (int): Maximum depth limit for the tree (None = unlimited)
        criterion (str): Splitting criterion - 'information_gain', 'gain_ratio', or 'gini'
        feature_names (list): Names of the input features
        classes_ (list): Unique class labels in the training data
        feature_values (dict): All possible values for each categorical feature index
        continuous_features (list): Indices of continuous features
        current_splits (dict): Temporary storage for current node's splits
    """

    def __init__(self, max_depth=None, criterion='information_gain', continuous_features=None):
        """
        Initialize the ID3 decision tree classifier.

        Args:
            max_depth (int, optional): Maximum depth of the tree to prevent
                                     overfitting. If None, nodes are expanded
                                     until all leaves are pure or no features
                                     remain. Defaults to None.
            criterion (str, optional): Splitting criterion to use. Options are:
                                     - 'information_gain': Standard entropy-based IG (default)
                                     - 'gain_ratio': IG normalized by split information
                                     - 'gini': Gini impurity index
                                     Defaults to 'information_gain'.
            continuous_features (list, optional): List of feature indices or names that are 
                                                continuous. If None, will auto-detect based 
                                                on data types. Defaults to None.
        """
        self.tree = None  # Will store the trained decision tree
        self.max_depth = max_depth  # Depth limit for regularization
        self.criterion = criterion  # Splitting criterion to use
        self.feature_names = None  # Human-readable feature names
        self.classes_ = None  # Unique class labels from training data
        # Store all possible values for each categorical feature (important for unseen values)
        self.feature_values = {}
        # List of continuous feature indices
        self.continuous_features = continuous_features or []
        # Temporary storage for current node's splits
        self.current_splits = {}
        
        # Validate criterion parameter
        valid_criteria = ['information_gain', 'gain_ratio', 'gini']
        if criterion not in valid_criteria:
            raise ValueError(f"Invalid criterion '{criterion}'. Must be one of {valid_criteria}")

    def _detect_continuous_features(self, X):
        """
        Automatically detect which features are continuous based on data types and unique values.

        Args:
            X (numpy.ndarray): Feature matrix

        Returns:
            list: Indices of features detected as continuous

        Note:
            Uses heuristic: features with >10 unique values or floating point values
            are considered continuous.
        """
        continuous_features = []

        for i in range(X.shape[1]):
            # Check if feature has numeric type and many unique values
            feature_values = X[:, i]

            try:
                # Try to convert to float - if successful, could be continuous
                numeric_values = feature_values.astype(float)
                unique_values = len(np.unique(numeric_values))

                # Heuristic: if more than 10 unique values or if values are floating point,
                # consider it continuous
                if unique_values > 10 or any(val != int(val) for val in numeric_values if not np.isnan(val)):
                    continuous_features.append(i)

            except (ValueError, TypeError):
                # If conversion fails, it's categorical
                continue

        return continuous_features

    def _find_optimal_continuous_split(self, X, y, feature_idx):
        """
        Find the optimal split point for a continuous feature.

        For continuous features, the optimal threshold must lie at a boundary between
        adjacent examples with different class labels when sorted by feature value.

        Args:
            X (numpy.ndarray): Feature matrix
            y (numpy.ndarray): Target labels
            feature_idx (int): Index of the continuous feature

        Returns:
            tuple: (best_split_point, best_information_gain)

        Note:
            Only considers split points between examples with different class labels,
            as these are the only potentially optimal thresholds.
        """
        # Get feature values and corresponding labels
        feature_values = X[:, feature_idx].astype(float)

        # Create sorted indices
        sorted_indices = np.argsort(feature_values)
        sorted_values = feature_values[sorted_indices]
        sorted_labels = y[sorted_indices]

        best_split = None
        best_gain = -np.inf

        # Calculate initial entropy/impurity
        if self.criterion == 'gini':
            initial_impurity = self.gini_impurity(y)
        else:
            initial_impurity = self.entropy(y)

        # Try each potential split point
        for i in range(1, len(sorted_values)):
            # Only consider splits between different class labels
            if sorted_labels[i-1] != sorted_labels[i]:
                # Calculate split point as midpoint
                split_point = (sorted_values[i-1] + sorted_values[i]) / 2

                # Split the data
                left_mask = feature_values <= split_point
                right_mask = feature_values > split_point

                left_labels = y[left_mask]
                right_labels = y[right_mask]

                if len(left_labels) == 0 or len(right_labels) == 0:
                    continue

                # Calculate weighted impurity after split
                n = len(y)
                left_weight = len(left_labels) / n
                right_weight = len(right_labels) / n

                if self.criterion == 'gini':
                    weighted_impurity = (left_weight * self.gini_impurity(left_labels) +
                                       right_weight * self.gini_impurity(right_labels))
                else:
                    weighted_impurity = (left_weight * self.entropy(left_labels) +
                                       right_weight * self.entropy(right_labels))

                # Calculate information gain
                information_gain = initial_impurity - weighted_impurity

                if information_gain > best_gain:
                    best_gain = information_gain
                    best_split = split_point

        return best_split, best_gain

    def entropy(self, y):
        """
        Calculate the entropy of a label array.

        Entropy measures the impurity or randomness in a set of labels.
        It's defined as: H(S) = -∑(p_i * log2(p_i))
        where p_i is the probability of class i.

        Properties:
        - Entropy = 0 when all samples belong to the same class (pure)
        - Entropy is maximized when classes are equally distributed
        - Range: [0, log2(num_classes)]

        Args:
            y (array-like): Array of class labels

        Returns:
            float: Entropy value. Returns 0 for empty arrays.

        Example:
            >>> entropy([1, 1, 0, 0])  # Two classes, equal distribution
            1.0
            >>> entropy([1, 1, 1, 1])  # One class, pure
            0.0
        """
        if len(y) == 0:
            return 0  # Empty set has zero entropy by convention

        # Count occurrences of each class label
        counts = Counter(y)
        
        # Calculate probability of each class
        probabilities = [count / len(y) for count in counts.values()]
        
        # Calculate entropy: -∑(p * log2(p))
        # Note: We check p > 0 to avoid log(0) which is undefined
        return -sum(p * np.log2(p) for p in probabilities if p > 0)

    def gini_impurity(self, y):
        """
        Calculate the Gini impurity of a label array.

        Gini impurity measures the probability of incorrectly classifying
        a randomly chosen element if it was randomly labeled according to
        the distribution of labels in the subset.
        
        It's defined as: Gini(S) = 1 - ∑(p_i^2)
        where p_i is the probability of class i.

        Properties:
        - Gini = 0 when all samples belong to the same class (pure)
        - Gini is maximized when classes are equally distributed
        - Range: [0, 1 - 1/num_classes]
        - Computationally simpler than entropy (no logarithms)

        Args:
            y (array-like): Array of class labels

        Returns:
            float: Gini impurity value. Returns 0 for empty arrays.

        Example:
            >>> gini_impurity([1, 1, 0, 0])  # Two classes, equal distribution
            0.5
            >>> gini_impurity([1, 1, 1, 1])  # One class, pure
            0.0
        """
        if len(y) == 0:
            return 0

        # Count occurrences of each class label
        counts = Counter(y)
        
        # Calculate probability of each class
        probabilities = [count / len(y) for count in counts.values()]
        
        # Calculate Gini impurity: 1 - ∑(p^2)
        return 1 - sum(p ** 2 for p in probabilities)

    def split_information(self, X, feature_idx):
        """
        Calculate the split information (intrinsic value) for a feature.

        Split information represents the entropy of the distribution of
        samples across the different values of a feature. It's used to
        normalize information gain to create the gain ratio.

        For categorical features:
        SplitInfo(S, A) = -∑((|S_v| / |S|) * log2(|S_v| / |S|))

        For continuous features:
        SplitInfo is always 1.0 (binary split)

        Args:
            X (numpy.ndarray): Feature matrix (samples x features)
            feature_idx (int): Index of the feature to evaluate

        Returns:
            float: Split information value. Returns a small epsilon value
                   (1e-10) for zero split information to avoid division by zero.
        """
        if feature_idx in self.continuous_features:
            # For continuous features with binary splits, split info is always log2(2) = 1
            # unless all values are the same
            if len(set(X[:, feature_idx])) <= 1:
                return 1e-10  # Avoid division by zero
            return 1.0  # Binary split
        else:
            # Original categorical logic
            values = set(X[:, feature_idx])
            split_info = 0
            n = len(X)
            
            for value in values:
                subset_size = np.sum(X[:, feature_idx] == value)
                if subset_size > 0:
                    proportion = subset_size / n
                    split_info -= proportion * np.log2(proportion)
            
            # Return small epsilon instead of 0 to avoid division by zero
            return split_info if split_info > 0 else 1e-10

    def information_gain(self, X, y, feature_idx):
        """
        Calculate information gain for splitting on a specific feature.

        For categorical features:
        Information gain measures how much uncertainty is reduced by splitting
        on a particular feature. It's calculated as:
        IG(S, A) = H(S) - ∑((|S_v| / |S|) * H(S_v))

        For continuous features:
        Information gain is calculated using the optimal split point found
        by _find_optimal_continuous_split method.

        Args:
            X (numpy.ndarray): Feature matrix (samples x features)
            y (numpy.ndarray): Target labels
            feature_idx (int): Index of the feature to evaluate

        Returns:
            float: Information gain value. Higher values indicate better splits.
        """
        if feature_idx in self.continuous_features:
            # For continuous features, find optimal split
            split_point, gain = self._find_optimal_continuous_split(X, y, feature_idx)

            # Store the optimal split for this evaluation
            if split_point is not None:
                self.current_splits[feature_idx] = split_point
                return gain
            else:
                return 0
        else:
            # Original categorical logic
            if self.criterion == 'gini':
                total_impurity = self.gini_impurity(y)
            else:
                total_impurity = self.entropy(y)

            values = set(X[:, feature_idx])
            weighted_impurity = 0

            for value in values:
                subset_indices = X[:, feature_idx] == value
                subset_y = y[subset_indices]
                weight = len(subset_y) / len(y)

                if self.criterion == 'gini':
                    weighted_impurity += weight * self.gini_impurity(subset_y)
                else:
                    weighted_impurity += weight * self.entropy(subset_y)

            return total_impurity - weighted_impurity

    def gain_ratio(self, X, y, feature_idx):
        """
        Calculate gain ratio for splitting on a specific feature.

        Gain ratio is a normalized version of information gain that addresses
        the bias toward features with many distinct values. It's calculated as:
        GainRatio(S, A) = IG(S, A) / SplitInfo(S, A)

        By dividing by split information, gain ratio penalizes features
        that create many small, fragmented subsets.

        Args:
            X (numpy.ndarray): Feature matrix (samples x features)
            y (numpy.ndarray): Target labels
            feature_idx (int): Index of the feature to evaluate

        Returns:
            float: Gain ratio value. Higher values indicate better splits.

        Note:
            Gain ratio is particularly useful when features have different
            numbers of possible values, as it reduces bias toward high-cardinality
            features.
        """
        # Calculate standard information gain
        ig = self.information_gain(X, y, feature_idx)
        
        # Calculate split information
        split_info = self.split_information(X, feature_idx)
        
        # Return normalized gain ratio
        return ig / split_info

    def gini_gain(self, X, y, feature_idx):
        """
        Calculate Gini gain (reduction in Gini impurity) for splitting on a feature.

        Gini gain measures how much the Gini impurity is reduced by splitting
        on a particular feature. Uses the same calculation as information_gain
        but with Gini impurity as the base measure.

        Args:
            X (numpy.ndarray): Feature matrix (samples x features)
            y (numpy.ndarray): Target labels
            feature_idx (int): Index of the feature to evaluate

        Returns:
            float: Gini gain value. Higher values indicate better splits.

        Note:
            Gini index is computationally simpler than entropy-based measures
            and often produces similar results in practice.
        """
        return self.information_gain(X, y, feature_idx)  # Same calculation, different base impurity

    def calculate_split_criterion(self, X, y, feature_idx):
        """
        Calculate the splitting criterion value for a feature based on the
        selected criterion type.

        This method delegates to the appropriate criterion calculation method
        based on self.criterion.

        Args:
            X (numpy.ndarray): Feature matrix (samples x features)
            y (numpy.ndarray): Target labels
            feature_idx (int): Index of the feature to evaluate

        Returns:
            float: Criterion value for the feature. Higher is better.

        Raises:
            ValueError: If criterion is not recognized (should not happen
                       due to validation in __init__)
        """
        if self.criterion == 'information_gain':
            return self.information_gain(X, y, feature_idx)
        elif self.criterion == 'gain_ratio':
            return self.gain_ratio(X, y, feature_idx)
        elif self.criterion == 'gini':
            return self.gini_gain(X, y, feature_idx)
        else:
            raise ValueError(f"Unknown criterion: {self.criterion}")

    def majority_class(self, y):
        """
        Return the most common class label in the array.

        This is used when we need to make a prediction at a leaf node
        or when we reach stopping criteria and need to assign a class.

        Args:
            y (array-like): Array of class labels

        Returns:
            The most frequently occurring class label

        Note:
            In case of ties, Counter.most_common() returns one of the
            tied values arbitrarily.
        """
        return Counter(y).most_common(1)[0][0]

    def build_tree(self, X, y, features, depth=0, global_majority=None):
        """
        Recursively build the ID3 decision tree with support for continuous features.

        This is the core algorithm that constructs the tree by:
        1. Checking stopping criteria (pure node, no features, max depth)
        2. Finding the best feature to split on (highest criterion value)
        3. Creating branches for categorical features or binary split for continuous
        4. Recursively building subtrees for each branch

        Args:
            X (numpy.ndarray): Feature matrix for current subset
            y (numpy.ndarray): Labels for current subset
            features (list): Available feature indices for splitting
            depth (int): Current depth in the tree (for max_depth check)
            global_majority (str): Most common class in entire dataset
                                 (used for empty branches)

        Returns:
            dict: Tree node structure with either:
                - {'class': label} for leaf nodes
                - {'feature': idx, 'feature_name': name, 'branches': {}, 
                   'is_continuous': bool, 'split_point': float} for decision nodes

        Note:
            The tree structure uses dictionaries to represent nodes:
            - Leaf nodes contain only a 'class' key
            - Decision nodes contain 'feature', 'feature_name', 'branches', and 
              for continuous features: 'is_continuous' and 'split_point'
            - Branches map feature values (categorical) or conditions (continuous) to child nodes
        """
        # Set global majority class on the first call (root level)
        if global_majority is None:
            global_majority = self.majority_class(y)

        # === STOPPING CRITERIA ===
        
        # Stop if all samples have the same class (pure node)
        if len(set(y)) == 1:
            return {'class': y[0]}

        # Stop if no more features are available for splitting
        if len(features) == 0:
            return {'class': self.majority_class(y)}

        # Stop if maximum depth is reached (regularization)
        if self.max_depth is not None and depth >= self.max_depth:
            return {'class': self.majority_class(y)}

        # === FEATURE SELECTION ===
        
        # Clear previous splits and evaluate all features
        self.current_splits = {}
        criterion_values = []
        for f in features:
            criterion_values.append(self.calculate_split_criterion(X, y, f))

        best_feature_idx = features[np.argmax(criterion_values)]
        best_feature_name = (
            self.feature_names[best_feature_idx]
            if self.feature_names else best_feature_idx
        )

        # === TREE NODE CREATION ===
        
        # Create decision node structure
        tree = {
            'feature': best_feature_idx,
            'feature_name': best_feature_name,
            'branches': {}
        }

        # Handle continuous vs categorical features differently
        if best_feature_idx in self.continuous_features:
            # Continuous feature: binary split
            split_point = self.current_splits.get(best_feature_idx)
            if split_point is None:
                # Fallback if no split found
                return {'class': self.majority_class(y)}

            tree['split_point'] = split_point
            tree['is_continuous'] = True

            # Create binary branches
            left_mask = X[:, best_feature_idx].astype(float) <= split_point
            right_mask = ~left_mask

            # Left branch (<= split_point)
            left_X, left_y = X[left_mask], y[left_mask]
            if len(left_y) == 0:
                tree['branches'][f'<= {split_point:.2f}'] = {'class': global_majority}
            else:
                # Continuous features can be reused, so don't remove from features
                tree['branches'][f'<= {split_point:.2f}'] = self.build_tree(
                    left_X, left_y, features, depth + 1, global_majority
                )

            # Right branch (> split_point)
            right_X, right_y = X[right_mask], y[right_mask]
            if len(right_y) == 0:
                tree['branches'][f'> {split_point:.2f}'] = {'class': global_majority}
            else:
                tree['branches'][f'> {split_point:.2f}'] = self.build_tree(
                    right_X, right_y, features, depth + 1, global_majority
                )

        else:
            # Categorical feature: multiway split
            tree['is_continuous'] = False
            all_values = self.feature_values[best_feature_idx]
            remaining_features = [f for f in features if f != best_feature_idx]

            for value in all_values:
                subset_indices = X[:, best_feature_idx] == value
                subset_X = X[subset_indices]
                subset_y = y[subset_indices]

                if len(subset_y) == 0:
                    tree['branches'][value] = {'class': global_majority}
                else:
                    tree['branches'][value] = self.build_tree(
                        subset_X, subset_y, remaining_features,
                        depth + 1, global_majority
                    )

        return tree

    def fit(self, X, y, feature_names=None):
        """
        Fit the ID3 decision tree to training data.

        This method prepares the data and initiates the tree building process.
        It stores metadata about features and classes, detects continuous features,
        then calls build_tree() to construct the actual decision tree.

        Args:
            X (array-like): Training feature matrix (n_samples, n_features)
            y (array-like): Training target labels (n_samples,)
            feature_names (list, optional): Names for features. If None,
                                          indices will be used as names.

        Returns:
            self: Returns the fitted estimator instance

        Note:
            - Input arrays are converted to numpy arrays for consistency
            - Continuous features are detected or converted from names to indices
            - All possible values for each categorical feature are stored to handle
              unseen values during prediction
        """
        # Convert inputs to numpy arrays for consistent handling
        if isinstance(X, list):
            X = np.array(X)
        if isinstance(y, list):
            y = np.array(y)

        # Store feature names for interpretability
        self.feature_names = feature_names
        
        # Store unique class labels
        self.classes_ = list(set(y))

        # Convert continuous_features from names to indices if needed
        if self.feature_names and self.continuous_features:
            continuous_indices = []
            for cf in self.continuous_features:
                if isinstance(cf, str):
                    # It's a feature name, convert to index
                    try:
                        idx = self.feature_names.index(cf)
                        continuous_indices.append(idx)
                    except ValueError:
                        print(f"Warning: Feature name '{cf}' not found in feature_names")
                else:
                    # It's already an index
                    continuous_indices.append(cf)
            self.continuous_features = continuous_indices

        # Auto-detect continuous features if not specified
        if not self.continuous_features:
            self.continuous_features = self._detect_continuous_features(X)

        # Store all possible values for categorical features
        for i in range(X.shape[1]):
            if i not in self.continuous_features:
                self.feature_values[i] = set(X[:, i])

        # Create list of all feature indices
        features = list(range(X.shape[1]))
        
        # Build the decision tree
        self.tree = self.build_tree(X, y, features)
        
        return self

    def predict_sample(self, x, tree):
        """
        Predict the class for a single sample by traversing the tree.

        This method recursively follows the decision path through the tree
        based on the sample's feature values until it reaches a leaf node.
        Handles both categorical and continuous features.

        Args:
            x (array-like): Single sample feature vector
            tree (dict): Current tree node to evaluate

        Returns:
            Class label prediction for the sample

        Note:
            - For categorical features: checks exact value match
            - For continuous features: compares against split point
            - If a feature value wasn't seen during training, falls back to 
              the most common class among current node's descendants
        """
        # Base case: reached a leaf node
        if 'class' in tree:
            return tree['class']

        # Get the feature index for this decision node
        feature_idx = tree['feature']
        feature_value = x[feature_idx]

        if tree.get('is_continuous', False):
            # Continuous feature: use split point
            split_point = tree['split_point']
            if float(feature_value) <= split_point:
                branch_key = f'<= {split_point:.2f}'
            else:
                branch_key = f'> {split_point:.2f}'
        else:
            # Categorical feature
            branch_key = feature_value

        if branch_key in tree['branches']:
            return self.predict_sample(x, tree['branches'][branch_key])
        else:
            # Handle unseen value
            classes = self._get_leaf_classes(tree)
            return Counter(classes).most_common(1)[0][0]

    def _get_leaf_classes(self, tree):
        """
        Get all class labels from leaf nodes in a subtree.

        This helper method collects all class predictions that could be
        made from the current subtree. Used for handling unseen feature values.

        Args:
            tree (dict): Root node of subtree to examine

        Returns:
            list: All class labels found in leaf nodes of the subtree
        """
        # Base case: this is a leaf node
        if 'class' in tree:
            return [tree['class']]
        
        # Recursively collect classes from all branches
        classes = []
        for branch in tree['branches'].values():
            classes.extend(self._get_leaf_classes(branch))
        
        return classes

    def predict(self, X):
        """
        Predict class labels for multiple samples.

        Args:
            X (array-like): Feature matrix (n_samples, n_features)

        Returns:
            numpy.ndarray: Predicted class labels (n_samples,)

        Raises:
            Exception: If the tree hasn't been fitted yet
        """
        # Convert input to numpy array for consistency
        if isinstance(X, list):
            X = np.array(X)

        # Predict each sample individually and return as numpy array
        return np.array([self.predict_sample(x, self.tree) for x in X])

    def print_tree(self, tree=None, indent=""):
        """
        Print a text representation of the decision tree.

        This method provides a human-readable view of the tree structure,
        showing the decision path and leaf classifications. Enhanced to show
        continuous feature split points.

        Args:
            tree (dict, optional): Tree node to print. If None, prints
                                 the entire tree from the root.
            indent (str): Current indentation level for formatting

        Example Output:
            Feature: elevation
              elevation <= 1250.5:
                Class: Oak
              elevation > 1250.5:
                Feature: stream
                  stream = False:
                    Class: Pine
        """
        # Use the root tree if none specified
        if tree is None:
            tree = self.tree

        # Print leaf node
        if 'class' in tree:
            print(f"{indent}Class: {tree['class']}")
        else:
            # Print decision node
            feature_name = tree['feature_name']
            if tree.get('is_continuous', False):
                print(f"{indent}Feature: {feature_name} (continuous, split: {tree['split_point']:.2f})")
            else:
                print(f"{indent}Feature: {feature_name}")

            # Print each branch
            for value, subtree in tree['branches'].items():
                if tree.get('is_continuous', False):
                    print(f"{indent}  {value}:")
                else:
                    print(f"{indent}  {feature_name} = {value}:")
                self.print_tree(subtree, indent + "    ")

    def _add_nodes_edges(self, tree, graph, parent_id=None, edge_label=None):
        """
        Recursively add nodes and edges to a Graphviz graph.

        This helper method builds a visual representation of the tree
        by adding nodes (decision points and leaves) and edges (branches)
        to a Graphviz graph object. Enhanced to handle continuous features.

        Args:
            tree (dict): Current tree node to process
            graph (Digraph): Graphviz graph object to modify
            parent_id (str, optional): ID of parent node (for edge creation)
            edge_label (str, optional): Label for edge from parent

        Note:
            - Leaf nodes are drawn as green rectangles
            - Decision nodes are drawn as blue ovals
            - Continuous features show split points in node labels
            - Edges are labeled with feature values or conditions
        """
        # Generate unique node ID using object memory address
        node_id = str(id(tree))

        if 'class' in tree:
            # Leaf node: rectangular shape, green fill
            graph.node(node_id, tree['class'], shape='box',
                      style='filled', fillcolor='lightgreen')
        else:
            # Decision node: oval shape, blue fill
            feature_name = tree['feature_name']
            if tree.get('is_continuous', False):
                label = f"{feature_name}\\nSplit: {tree['split_point']:.2f}"
            else:
                label = str(feature_name)

            graph.node(node_id, label, shape='ellipse',
                      style='filled', fillcolor='lightblue')

            # Recursively add child nodes and their edges
            for value, subtree in sorted(tree['branches'].items()):
                self._add_nodes_edges(subtree, graph, node_id, str(value))

        # Add edge from parent to current node (skip for root)
        if parent_id is not None:
            graph.edge(parent_id, node_id, label=edge_label)

    def plot_tree(self, filename='decision_tree_enhanced', view=True):
        """
        Generate a visual plot of the decision tree using Graphviz.

        Creates a graphical representation of the tree structure with
        decision nodes as ovals and leaf nodes as rectangles, connected
        by labeled edges. Enhanced to show continuous feature split points.

        Args:
            filename (str): Base name for output file (without extension)
            view (bool): Whether to automatically open the generated image
                        In Jupyter notebooks, this controls whether to display
                        the image inline (True) or just save it (False)

        Returns:
            Digraph object if successful, None if Graphviz not available

        Raises:
            ImportError: If Graphviz is not installed

        Note:
            Requires both Python graphviz package and system Graphviz
            installation. The tree must be fitted before plotting.
            In Jupyter notebooks, displays PNG format to avoid LaTeX export issues.
        """
        # Check for Graphviz dependency
        try:
            from graphviz import Digraph
        except ImportError:
            print("Please install graphviz: pip install graphviz")
            print("Also install the system graphviz: https://graphviz.org/download/")
            return None

        # Check if tree has been trained
        if self.tree is None:
            print("Tree has not been fitted yet!")
            return None

        # Check if we're running in a Jupyter notebook
        def _is_notebook():
            try:
                from IPython import get_ipython
                if (get_ipython() is not None and
                        get_ipython().__class__.__name__ == 'ZMQInteractiveShell'):
                    return True
                return False
            except ImportError:
                return False

        # Handle Jupyter vs non-Jupyter environments differently
        if _is_notebook():
            # In Jupyter: Use subprocess to avoid Graphviz object display issues
            import subprocess
            import tempfile
            from pathlib import Path

            # Create DOT source code as string
            dot_source = self._generate_dot_source()

            try:
                # Write DOT source to temporary file
                with tempfile.NamedTemporaryFile(mode='w', suffix='.dot',
                                                 delete=False) as f:
                    f.write(dot_source)
                    temp_dot_file = f.name

                # Use command line dot to generate PNG
                png_filename = f"{filename}.png"
                result = subprocess.run([
                    'dot', '-Tpng', temp_dot_file, '-o', png_filename
                ], capture_output=True, text=True, check=True)

                # Clean up temporary file
                Path(temp_dot_file).unlink()

                print(f"Enhanced tree saved as '{png_filename}'")

                # Display inline if view=True
                if view:
                    try:
                        from IPython.display import Image, display
                        display(Image(png_filename))
                    except ImportError:
                        print(f"PNG saved as '{png_filename}' "
                              "(inline display unavailable)")

                return None  # Return None to prevent any object display

            except subprocess.CalledProcessError as e:
                print(f"Error generating PNG: {e}")
                print("Falling back to standard method...")
                # Fall through to standard method
            except FileNotFoundError:
                print("Command-line 'dot' not found. "
                      "Please install Graphviz system package.")
                print("Falling back to standard method...")
                # Fall through to standard method
            except Exception as e:
                print(f"Unexpected error: {e}")
                print("Falling back to standard method...")
                # Fall through to standard method

        # Standard method (outside Jupyter or if subprocess method failed)
        graph = Digraph(comment='Enhanced ID3 Decision Tree')
        graph.attr(rankdir='TB')  # Top to bottom layout

        # Build the graph structure
        self._add_nodes_edges(self.tree, graph)

        # Render and save the graph
        graph.render(filename, format='png', cleanup=True, view=view)
        print(f"Enhanced tree saved as '{filename}.png'")

        return graph

    def _generate_dot_source(self):
        """
        Generate DOT source code for the decision tree with continuous feature support.

        Returns:
            str: DOT format string representing the tree
        """
        lines = ['digraph "Enhanced ID3 Decision Tree" {', 'rankdir=TB;']

        def add_node_to_dot(tree, node_id):
            if 'class' in tree:
                # Leaf node
                lines.append(
                    f'{node_id} [label="{tree["class"]}" shape=box '
                    'style=filled fillcolor=lightgreen];'
                )
            else:
                # Decision node
                feature_name = tree['feature_name']
                if tree.get('is_continuous', False):
                    label = f"{feature_name}\\nSplit: {tree['split_point']:.2f}"
                else:
                    label = str(feature_name)

                lines.append(
                    f'{node_id} [label="{label}" shape=ellipse '
                    'style=filled fillcolor=lightblue];'
                )

                # Add edges to children
                for i, (value, subtree) in enumerate(
                        sorted(tree['branches'].items())):
                    child_id = f"{node_id}_{i}"
                    lines.append(f'{node_id} -> {child_id} [label="{value}"];')
                    add_node_to_dot(subtree, child_id)

        add_node_to_dot(self.tree, "root")
        lines.append('}')
        return '\n'.join(lines)