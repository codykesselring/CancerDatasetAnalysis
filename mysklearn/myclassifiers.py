from mysklearn import myutils

from mysklearn import myutils
from mysklearn.mysimplelinearregressor import MySimpleLinearRegressor

class MySimpleLinearRegressionClassifier:
    """Represents a simple linear regression classifier that discretizes
        predictions from a simple linear regressor (see MySimpleLinearRegressor).

    Attributes:
        discretizer(function): a function that discretizes a numeric value into
            a string label. The function's signature is func(obj) -> obj
        regressor(MySimpleLinearRegressor): the underlying regression model that
            fits a line to x and y data

    Notes:
        Terminology: instance = sample = row and attribute = feature = column
    """

    def __init__(self, discretizer, regressor=None):
        """Initializer for MySimpleLinearClassifier.

        Args:
            discretizer(function): a function that discretizes a numeric value into
                a string label. The function's signature is func(obj) -> obj
            regressor(MySimpleLinearRegressor): the underlying regression model that
                fits a line to x and y data (None if to be created in fit())
        """
        self.discretizer = discretizer
        self.regressor = regressor

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        if self.regressor is None:
            self.regressor = MySimpleLinearRegressor()
        
        self.regressor.fit(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test by applying discretizer
            to the numeric predictions from regressor.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        numeric_predictions = self.regressor.predict(X_test)
        
        # Apply discretizer to convert numeric predictions to class labels
        y_predicted = [self.discretizer(pred) for pred in numeric_predictions]
        
        return y_predicted

class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors=3):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        self.X_train = X_train
        self.y_train = y_train

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        neighbor_indices = []
        
        for test_instance in X_test:
            # Calculate distances
            instance_distances = []
            for i, train_instance in enumerate(self.X_train):
                dist = sum((test_instance[j] - train_instance[j]) ** 2 
                          for j in range(len(test_instance))) ** 0.5
                instance_distances.append((dist, i))
            
            # Sort by distance and get k nearest neighbors
            instance_distances.sort(key=lambda x: x[0])
            k_nearest = instance_distances[:self.n_neighbors]
            
            # Separate distances and indices
            k_distances = [dist for dist, idx in k_nearest]
            k_indices = [idx for dist, idx in k_nearest]
            
            distances.append(k_distances)
            neighbor_indices.append(k_indices)
        
        return distances, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        
        _, neighbor_indices = self.kneighbors(X_test)
        
        for indices in neighbor_indices:
            neighbor_labels = [self.y_train[i] for i in indices]
            
            # Find the most common label 
            label_counts = {}
            for label in neighbor_labels:
                label_counts[label] = label_counts.get(label, 0) + 1
            
            # Find the label with max count
            predicted_label = max(label_counts, key=label_counts.get)
            y_predicted.append(predicted_label)
        
        return y_predicted

class MyDummyClassifier:
    """Represents a "dummy" classifier using the "most_frequent" strategy.
        The most_frequent strategy is a Zero-R classifier, meaning it ignores
        X_train and produces zero "rules" from it. Instead, it only uses
        y_train to see what the most frequent class label is. That is
        always the dummy classifier's prediction, regardless of X_test.

    Attributes:
        most_common_label(obj): whatever the most frequent class label in the
            y_train passed into fit()

    Notes:
        Loosely based on sklearn's DummyClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
    """
    def __init__(self):
        """Initializer for DummyClassifier.

        """
        self.most_common_label = None

    def fit(self, X_train, y_train):
        """Fits a dummy classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Zero-R only predicts the most frequent class label, this method
                only saves the most frequent class label.
        """
        label_counts = {}
        for label in y_train:
            label_counts[label] = label_counts.get(label, 0) + 1
        
        # Find the label with maximum count
        self.most_common_label = max(label_counts, key=label_counts.get)

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        return [self.most_common_label for _ in X_test]
    
class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        priors(YOU CHOOSE THE MOST APPROPRIATE TYPE): The prior probabilities computed for each
            label in the training set.
        conditionals(YOU CHOOSE THE MOST APPROPRIATE TYPE): The conditional probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        You may add additional instance attributes if you would like, just be sure to update this docstring
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.
        """
        self.priors = None
        self.conditionals = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples)
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the conditional probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and conditionals.
        """
        self.priors = {}
        total_instances = len(y_train)

        for label in y_train:
            self.priors[label] = self.priors.get(label, 0) + 1

        for label in self.priors:
            self.priors[label] = self.priors[label] / total_instances

        # P(attribute_value | class)
        self.conditionals = {}

        for label in self.priors.keys():
            self.conditionals[label] = {}

            label_instances = [X_train[i] for i in range(len(y_train)) if y_train[i] == label]
            label_count = len(label_instances)

            n_features = len(X_train[0])
            for attr_idx in range(n_features):
                self.conditionals[label][attr_idx] = {}

                # Count occurrences of each attribute value for this class
                attr_values = [instance[attr_idx] for instance in label_instances]
                for value in attr_values:
                    self.conditionals[label][attr_idx][value] = \
                        self.conditionals[label][attr_idx].get(value, 0) + 1

                # Convert counts to probabilities
                for value in self.conditionals[label][attr_idx]:
                    self.conditionals[label][attr_idx][value] = \
                        self.conditionals[label][attr_idx][value] / label_count

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for test_instance in X_test:
            # Calculate posterior probability for each class
            posteriors = {}

            for label in self.priors.keys():
                # Start with prior probability
                posterior = self.priors[label]

                # Multiply by conditional probabilities for each attribute
                for attr_idx, attr_value in enumerate(test_instance):
                    if attr_value in self.conditionals[label][attr_idx]:
                        posterior *= self.conditionals[label][attr_idx][attr_value]
                    else:
                        posterior = 0
                        break

                posteriors[label] = posterior

            # predict the class with highest probability
            predicted_label = max(posteriors, key=posteriors.get)
            y_predicted.append(predicted_label)

        return y_predicted


class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train).
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier:
            https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.
        """
        self.X_train = None
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT
        (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples).
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            On a majority vote tie, choose first attribute value based on attribute domain ordering.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train

        available_attributes = list(range(len(X_train[0])))

        self.tree = self._tdidt(X_train, y_train, available_attributes, len(y_train))

    def _tdidt(self, X, y, available_attributes, parent_total):
        """Recursively builds a decision tree using the TDIDT algorithm.

        Args:
            X(list of list of obj): Current instances
            y(list of obj): Current labels
            available_attributes(list of int): Available attribute indices
            parent_total(int): Total instances in parent partition

        Returns:
            tree(nested list): The decision tree
        """
        #base case 1: All labels are the same 
        if len(set(y)) == 1:
            return ["Leaf", y[0], len(y), parent_total]

        #base case 2: clash
        if len(available_attributes) == 0:
            return ["Leaf", self._majority_vote(y), len(y), parent_total]

        best_attr = self._select_attribute(X, y, available_attributes)

        remaining_attributes = [attr for attr in available_attributes if attr != best_attr]

        tree = ["Attribute", f"att{best_attr}"]

        partitions = self._partition_by_attribute(X, y, best_attr)

        sorted_values = sorted(partitions.keys())

        for value in sorted_values:
            X_partition = partitions[value]['X']
            y_partition = partitions[value]['y']

            subtree = self._tdidt(X_partition, y_partition, remaining_attributes, len(y))

            # Add value branch
            tree.append(["Value", value, subtree])

        return tree

    def _select_attribute(self, X, y, available_attributes):
        """Select the attribute with the highest information gain.

        Args:
            X(list of list of obj): Current instances
            y(list of obj): Current labels
            available_attributes(list of int): Available attribute indices

        Returns:
            best_attr(int): Index of attribute with highest information gain
        """
        best_attr = available_attributes[0]
        best_gain = -1

        for attr in available_attributes:
            gain = self._information_gain(X, y, attr)
            if gain > best_gain:
                best_gain = gain
                best_attr = attr

        return best_attr

    def _information_gain(self, X, y, attr_index):
        """Calculate information gain for an attribute.

        Args:
            X(list of list of obj): Current instances
            y(list of obj): Current labels
            attr_index(int): Attribute index

        Returns:
            gain(float): Information gain
        """
        total_entropy = self._entropy(y)

        partitions = self._partition_by_attribute(X, y, attr_index)

        weighted_entropy = 0
        for partition in partitions.values():
            weight = len(partition['y']) / len(y)
            weighted_entropy += weight * self._entropy(partition['y'])

        return total_entropy - weighted_entropy

    def _entropy(self, labels):
        """Calculate entropy of a list of labels.

        Args:
            labels(list of obj): List of class labels

        Returns:
            entropy(float): Entropy value
        """
        if not labels:
            return 0

        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        total = len(labels)
        entropy = 0

        import math
        for count in label_counts.values():
            if count > 0:
                p = count / total
                if p > 0:
                    entropy -= p * math.log2(p)

        return entropy

    def _partition_by_attribute(self, X, y, attr_index):
        """Partition instances by attribute values.

        Args:
            X(list of list of obj): Current instances
            y(list of obj): Current labels
            attr_index(int): Attribute index

        Returns:
            partitions(dict): Dictionary mapping values to {'X': [...], 'y': [...]}
        """
        partitions = {}

        for i, instance in enumerate(X):
            value = instance[attr_index]
            if value not in partitions:
                partitions[value] = {'X': [], 'y': []}
            partitions[value]['X'].append(instance)
            partitions[value]['y'].append(y[i])

        return partitions

    def _majority_vote(self, labels):
        """Find the majority class label, with ties broken alphabetically.

        Args:
            labels(list of obj): List of class labels

        Returns:
            majority(obj): Majority class label
        """
        label_counts = {}
        for label in labels:
            label_counts[label] = label_counts.get(label, 0) + 1

        max_count = max(label_counts.values())

        max_labels = [label for label, count in label_counts.items() if count == max_count]

        return sorted(max_labels)[0]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []

        for instance in X_test:
            prediction = self._predict_instance(instance, self.tree)
            y_predicted.append(prediction)

        return y_predicted

    def _predict_instance(self, instance, tree):
        """Predict class label for a single instance by traversing the tree.

        Args:
            instance(list of obj): A single test instance
            tree(nested list): The decision tree

        Returns:
            prediction(obj): Predicted class label
        """
        # Base case
        if tree[0] == "Leaf":
            return tree[1]  # Return the class label

        # internal node
        attr_name = tree[1]
        attr_index = int(attr_name[3:])

        instance_value = instance[attr_index]

        for i in range(2, len(tree)):
            value_branch = tree[i]
            if value_branch[1] == instance_value:
                return self._predict_instance(instance, value_branch[2])

        # If no matching branch found, return majority class from available branches
        # This handles unseen attribute values
        majority_label = self._get_majority_from_tree(tree)
        return majority_label

    def _get_majority_from_tree(self, tree):
        """Get the majority class from a tree or subtree.

        Args:
            tree(nested list): The decision tree or subtree

        Returns:
            label(obj): The majority class label
        """
        if tree[0] == "Leaf":
            return tree[1]

        # Collect all labels from leaf nodes in this subtree
        labels = []
        self._collect_leaf_labels(tree, labels)

        # Return majority vote
        if labels:
            return self._majority_vote(labels)
        return None

    def _collect_leaf_labels(self, tree, labels):
        """Recursively collect all leaf labels from a tree.

        Args:
            tree(nested list): The decision tree or subtree
            labels(list): List to collect labels into
        """
        if tree[0] == "Leaf":
            labels.append(tree[1])
            return

        # Recurse through value branches
        for i in range(2, len(tree)):
            value_branch = tree[i]
            subtree = value_branch[2]
            self._collect_leaf_labels(subtree, labels)

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format
        "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """
        self._print_rules_recursive(self.tree, [], attribute_names, class_name)

    def _print_rules_recursive(self, tree, rule_path, attribute_names, class_name):
        """Recursively traverse tree and print decision rules.

        Args:
            tree(nested list): Current subtree
            rule_path(list): Current path of conditions
            attribute_names(list of str or None): Attribute names
            class_name(str): Class name for rules
        """
        #base case
        if tree[0] == "Leaf":
            class_label = tree[1]
            if rule_path:
                rule_str = "IF " + " AND ".join(rule_path) + f" THEN {class_name} = {class_label}"
            else:
                rule_str = f"IF True THEN {class_name} = {class_label}"
            print(rule_str)
            return

        attr_name = tree[1]  # e.g., "att0"
        attr_index = int(attr_name[3:])  # Extract index

        if attribute_names:
            display_name = attribute_names[attr_index]
        else:
            display_name = attr_name

        # Traverse each value branch
        for i in range(2, len(tree)):
            value_branch = tree[i]

            value = value_branch[1]
            subtree = value_branch[2]

            new_rule_path = rule_path + [f"{display_name} == {value}"]
            self._print_rules_recursive(subtree, new_rule_path, attribute_names, class_name)

    # BONUS method
    def visualize_tree(self, dot_fname, pdf_fname, attribute_names=None):
        """BONUS: Visualizes a tree via the open source Graphviz graph visualization package and
        its DOT graph language (produces .dot and .pdf files).

        Args:
            dot_fname(str): The name of the .dot output file.
            pdf_fname(str): The name of the .pdf output file generated from the .dot file.
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes
                (e.g. "att0", "att1", ...) should be used).

        Notes:
            Graphviz: https://graphviz.org/
            DOT language: https://graphviz.org/doc/info/lang.html
            You will need to install graphviz in the Docker container as shown in class to complete this method.
        """
        pass # TODO: (BONUS) fix this

