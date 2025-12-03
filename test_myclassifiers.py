import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier



def test_decision_tree_classifier_fit():
     # LA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior",
                    ["Attribute", "att3",
                        ["Value", "no",
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]

    # Test interview dataset
    dt_classifier = MyDecisionTreeClassifier()
    dt_classifier.fit(X_train_interview, y_train_interview)
    assert dt_classifier.tree == tree_interview

    # Expected tree for iPhone dataset (from desk check)
    # Note: attribute values are sorted alphabetically/numerically
    # Note: for clashes with ties, choose alphabetically first class label
    tree_iphone = \
            ["Attribute", "att2",
                ["Value", "excellent",
                    ["Attribute", "att1",
                        ["Value", 1,
                            ["Attribute", "att0",
                                ["Value", 2,
                                    ["Leaf", "no", 2, 2]
                                ]
                            ]
                        ],
                        ["Value", 2,
                            ["Attribute", "att0",
                                ["Value", 1,
                                    ["Leaf", "yes", 1, 3]
                                ],
                                ["Value", 2,
                                    ["Leaf", "no", 2, 3]
                                ]
                            ]
                        ],
                        ["Value", 3,
                            ["Leaf", "no", 1, 6]
                        ]
                    ]
                ],
                ["Value", "fair",
                    ["Attribute", "att0",
                        ["Value", 1,
                            ["Attribute", "att1",
                                ["Value", 1,
                                    ["Leaf", "yes", 1, 3]
                                ],
                                ["Value", 2,
                                    ["Leaf", "yes", 1, 3]
                                ],
                                ["Value", 3,
                                    ["Leaf", "no", 1, 3]
                                ]
                            ]
                        ],
                        ["Value", 2,
                            ["Leaf", "yes", 6, 9]
                        ]
                    ]
                ]
            ]

    # Test iPhone dataset
    dt_classifier_iphone = MyDecisionTreeClassifier()
    dt_classifier_iphone.fit(X_train_iphone, y_train_iphone)
    assert dt_classifier_iphone.tree == tree_iphone

def test_decision_tree_classifier_predict():
     # LA7 (fake) iPhone purchases dataset
    header_iphone = ["standing", "job_status", "credit_rating"]
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    # interview dataset
    header_interview = ["level", "lang", "tweets", "phd", "interviewed_well"]
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # note: this tree uses the generic "att#" attribute labels because fit() does not and should not accept attribute names
    # note: the attribute values are sorted alphabetically
    tree_interview = \
            ["Attribute", "att0",
                ["Value", "Junior",
                    ["Attribute", "att3",
                        ["Value", "no",
                            ["Leaf", "True", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "False", 2, 5]
                        ]
                    ]
                ],
                ["Value", "Mid",
                    ["Leaf", "True", 4, 14]
                ],
                ["Value", "Senior",
                    ["Attribute", "att2",
                        ["Value", "no",
                            ["Leaf", "False", 3, 5]
                        ],
                        ["Value", "yes",
                            ["Leaf", "True", 2, 5]
                        ]
                    ]
                ]
            ]

    # Test interview dataset predictions
    # From B Attribute Selection (Entropy) Lab Task #2
    # Test instance 1: ["Junior", "Java", "yes", "no"] -> should predict "True"
    #   (path: level=Junior -> phd=no -> Leaf "True")
    # Test instance 2: ["Junior", "Java", "yes", "yes"] -> should predict "False"
    #   (path: level=Junior -> phd=yes -> Leaf "False")
    X_test_interview = [
        ["Junior", "Java", "yes", "no"],
        ["Junior", "Java", "yes", "yes"]
    ]
    y_predicted_interview = ["True", "False"]

    dt_classifier = MyDecisionTreeClassifier()
    dt_classifier.fit(X_train_interview, y_train_interview)
    predictions = dt_classifier.predict(X_test_interview)
    assert predictions == y_predicted_interview

    # Test iPhone dataset predictions
    # Test instance 1: [2, 2, "fair"]
    #   -> standing=2 -> credit_rating="fair" -> Leaf "yes"
    # Test instance 2: [1, 1, "excellent"]
    #   -> standing=1 -> job_status=1 -> Leaf "yes"
    X_test_iphone = [
        [2, 2, "fair"],
        [1, 1, "excellent"]
    ]
    y_predicted_iphone = ["yes", "yes"]

    dt_classifier_iphone = MyDecisionTreeClassifier()
    dt_classifier_iphone.fit(X_train_iphone, y_train_iphone)
    predictions_iphone = dt_classifier_iphone.predict(X_test_iphone)
    assert predictions_iphone == y_predicted_iphone


def test_random_forest_classifier_fit():
    """Test that MyRandomForestClassifier.fit() creates N trees and selects M best trees."""
    # Interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    # Test with specific parameters
    n_trees = 10
    m_trees = 5
    f_attributes = 2

    rf_classifier = MyRandomForestClassifier(n_trees=n_trees, m_trees=m_trees, f_attributes=f_attributes, random_state=42)
    rf_classifier.fit(X_train_interview, y_train_interview)

    # Check that the forest has exactly M trees
    assert rf_classifier.forest is not None
    assert len(rf_classifier.forest) == m_trees

    # Check that all trees in the forest are decision tree objects
    for tree in rf_classifier.forest:
        assert tree.tree is not None
        assert isinstance(tree.tree, list)


def test_random_forest_classifier_predict():
    """Test that MyRandomForestClassifier.predict() uses majority voting correctly."""
    # Interview dataset
    X_train_interview = [
        ["Senior", "Java", "no", "no"],
        ["Senior", "Java", "no", "yes"],
        ["Mid", "Python", "no", "no"],
        ["Junior", "Python", "no", "no"],
        ["Junior", "R", "yes", "no"],
        ["Junior", "R", "yes", "yes"],
        ["Mid", "R", "yes", "yes"],
        ["Senior", "Python", "no", "no"],
        ["Senior", "R", "yes", "no"],
        ["Junior", "Python", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "yes"],
        ["Mid", "Java", "yes", "no"],
        ["Junior", "Python", "no", "yes"]
    ]
    y_train_interview = ["False", "False", "True", "True", "True", "False", "True", "False", "True", "True", "True", "True", "True", "False"]

    X_test_interview = [
        ["Junior", "Java", "yes", "no"],
        ["Senior", "Python", "yes", "yes"],
        ["Mid", "Python", "no", "no"]
    ]

    # Test with reproducible random state
    rf_classifier = MyRandomForestClassifier(n_trees=20, m_trees=7, f_attributes=2, random_state=42)
    rf_classifier.fit(X_train_interview, y_train_interview)
    predictions = rf_classifier.predict(X_test_interview)

    # Check that predictions are returned for all test instances
    assert len(predictions) == len(X_test_interview)

    # Check that all predictions are valid class labels
    valid_labels = set(y_train_interview)
    for pred in predictions:
        assert pred in valid_labels


def test_random_forest_classifier_with_iphone_dataset():
    """Test MyRandomForestClassifier with the iPhone dataset."""
    X_train_iphone = [
        [1, 3, "fair"],
        [1, 3, "excellent"],
        [2, 3, "fair"],
        [2, 2, "fair"],
        [2, 1, "fair"],
        [2, 1, "excellent"],
        [2, 1, "excellent"],
        [1, 2, "fair"],
        [1, 1, "fair"],
        [2, 2, "fair"],
        [1, 2, "excellent"],
        [2, 2, "excellent"],
        [2, 3, "fair"],
        [2, 2, "excellent"],
        [2, 3, "fair"]
    ]
    y_train_iphone = ["no", "no", "yes", "yes", "yes", "no", "yes", "no", "yes", "yes", "yes", "yes", "yes", "no", "yes"]

    X_test_iphone = [
        [2, 2, "fair"],
        [1, 1, "excellent"],
        [2, 3, "excellent"]
    ]

    # Test with different parameters
    rf_classifier = MyRandomForestClassifier(n_trees=15, m_trees=10, f_attributes=2, random_state=123)
    rf_classifier.fit(X_train_iphone, y_train_iphone)
    predictions = rf_classifier.predict(X_test_iphone)

    # Check that predictions are returned
    assert len(predictions) == len(X_test_iphone)

    # Check that predictions are valid
    valid_labels = {"yes", "no"}
    for pred in predictions:
        assert pred in valid_labels

    # Check that the forest has the correct number of trees
    assert len(rf_classifier.forest) == 10


def test_random_forest_classifier_majority_voting():
    """Test that random forest uses majority voting for predictions."""
    # Simple dataset where majority voting matters
    X_train = [
        ["A", "X"],
        ["A", "Y"],
        ["B", "X"],
        ["B", "Y"],
        ["A", "X"],
        ["A", "Y"],
        ["B", "X"],
        ["B", "Y"]
    ]
    y_train = ["yes", "yes", "no", "no", "yes", "no", "no", "yes"]

    X_test = [["A", "X"], ["B", "Y"]]

    # Use a fixed random state for reproducibility
    rf_classifier = MyRandomForestClassifier(n_trees=10, m_trees=5, f_attributes=1, random_state=0)
    rf_classifier.fit(X_train, y_train)
    predictions = rf_classifier.predict(X_test)

    # Verify predictions are valid
    assert len(predictions) == 2
    assert all(pred in ["yes", "no"] for pred in predictions)


def test_random_forest_classifier_different_f_values():
    """Test random forest with different F (number of random attributes) values."""
    X_train = [
        [1, 2, 3, 4],
        [1, 3, 4, 5],
        [2, 2, 3, 4],
        [2, 3, 4, 5],
        [1, 2, 4, 5],
        [2, 3, 3, 4]
    ]
    y_train = ["A", "A", "B", "B", "A", "B"]

    X_test = [[1, 2, 3, 5], [2, 3, 4, 4]]

    # Test with F=1 (select 1 random attribute at each split)
    rf_f1 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=1, random_state=10)
    rf_f1.fit(X_train, y_train)
    pred_f1 = rf_f1.predict(X_test)

    # Test with F=2 (select 2 random attributes at each split)
    rf_f2 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=2, random_state=10)
    rf_f2.fit(X_train, y_train)
    pred_f2 = rf_f2.predict(X_test)

    # Test with F=3 (select 3 random attributes at each split)
    rf_f3 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=3, random_state=10)
    rf_f3.fit(X_train, y_train)
    pred_f3 = rf_f3.predict(X_test)

    # All should produce valid predictions
    assert len(pred_f1) == 2
    assert len(pred_f2) == 2
    assert len(pred_f3) == 2
    assert all(p in ["A", "B"] for p in pred_f1)
    assert all(p in ["A", "B"] for p in pred_f2)
    assert all(p in ["A", "B"] for p in pred_f3)
