import numpy as np

from mysklearn.myclassifiers import MyDecisionTreeClassifier, MyRandomForestClassifier, MyNaiveBayesClassifier



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

    dt_classifier = MyDecisionTreeClassifier()
    dt_classifier.fit(X_train_interview, y_train_interview)
    assert dt_classifier.tree == tree_interview

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

    n_trees = 10
    m_trees = 5
    f_attributes = 2

    rf_classifier = MyRandomForestClassifier(n_trees=n_trees, m_trees=m_trees, f_attributes=f_attributes, random_state=42)
    rf_classifier.fit(X_train_interview, y_train_interview)

    assert rf_classifier.forest is not None
    assert len(rf_classifier.forest) == m_trees

    for tree in rf_classifier.forest:
        assert tree.tree is not None
        assert isinstance(tree.tree, list)


def test_random_forest_classifier_predict():
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

    rf_classifier = MyRandomForestClassifier(n_trees=20, m_trees=7, f_attributes=2, random_state=42)
    rf_classifier.fit(X_train_interview, y_train_interview)
    predictions = rf_classifier.predict(X_test_interview)

    assert len(predictions) == len(X_test_interview)

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

    rf_classifier = MyRandomForestClassifier(n_trees=15, m_trees=10, f_attributes=2, random_state=123)
    rf_classifier.fit(X_train_iphone, y_train_iphone)
    predictions = rf_classifier.predict(X_test_iphone)

    assert len(predictions) == len(X_test_iphone)

    valid_labels = {"yes", "no"}
    for pred in predictions:
        assert pred in valid_labels

    assert len(rf_classifier.forest) == 10


def test_random_forest_classifier_majority_voting():
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

    rf_classifier = MyRandomForestClassifier(n_trees=10, m_trees=5, f_attributes=1, random_state=0)
    rf_classifier.fit(X_train, y_train)
    predictions = rf_classifier.predict(X_test)

    assert len(predictions) == 2
    assert all(pred in ["yes", "no"] for pred in predictions)


def test_random_forest_classifier_different_f_values():
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

    rf_f1 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=1, random_state=10)
    rf_f1.fit(X_train, y_train)
    pred_f1 = rf_f1.predict(X_test)

    rf_f2 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=2, random_state=10)
    rf_f2.fit(X_train, y_train)
    pred_f2 = rf_f2.predict(X_test)

    rf_f3 = MyRandomForestClassifier(n_trees=5, m_trees=3, f_attributes=3, random_state=10)
    rf_f3.fit(X_train, y_train)
    pred_f3 = rf_f3.predict(X_test)

    assert len(pred_f1) == 2
    assert len(pred_f2) == 2
    assert len(pred_f3) == 2
    assert all(p in ["A", "B"] for p in pred_f1)
    assert all(p in ["A", "B"] for p in pred_f2)
    assert all(p in ["A", "B"] for p in pred_f3)

def test_naive_bayes_classifier_fit():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # iPhone purchases dataset
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

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]

    # Test Case 1: In-class example
    nb1 = MyNaiveBayesClassifier()
    nb1.fit(X_train_inclass_example, y_train_inclass_example)

    assert np.isclose(nb1.priors["yes"], 5/8)
    assert np.isclose(nb1.priors["no"], 3/8)

    assert np.isclose(nb1.conditionals["yes"][0][1], 4/5)  # P(att1=1|yes)
    assert np.isclose(nb1.conditionals["yes"][0][2], 1/5)  # P(att1=2|yes)
    assert np.isclose(nb1.conditionals["yes"][1][5], 2/5)  # P(att2=5|yes)
    assert np.isclose(nb1.conditionals["yes"][1][6], 3/5)  # P(att2=6|yes)

    assert np.isclose(nb1.conditionals["no"][0][1], 2/3)  # P(att1=1|no)
    assert np.isclose(nb1.conditionals["no"][0][2], 1/3)  # P(att1=2|no)
    assert np.isclose(nb1.conditionals["no"][1][5], 2/3)  # P(att2=5|no)
    assert np.isclose(nb1.conditionals["no"][1][6], 1/3)  # P(att2=6|no)

    # Test Case 2: iPhone dataset
    nb2 = MyNaiveBayesClassifier()
    nb2.fit(X_train_iphone, y_train_iphone)

    assert np.isclose(nb2.priors["yes"], 10/15)
    assert np.isclose(nb2.priors["no"], 5/15)

    assert np.isclose(nb2.conditionals["yes"][0][1], 2/10)  # P(standing=1|yes)
    assert np.isclose(nb2.conditionals["yes"][0][2], 8/10)  # P(standing=2|yes)
    assert np.isclose(nb2.conditionals["yes"][1][1], 3/10)  # P(job_status=1|yes)
    assert np.isclose(nb2.conditionals["yes"][1][2], 4/10)  # P(job_status=2|yes)
    assert np.isclose(nb2.conditionals["yes"][1][3], 3/10)  # P(job_status=3|yes)
    assert np.isclose(nb2.conditionals["yes"][2]["fair"], 7/10)  # P(credit_rating=fair|yes)
    assert np.isclose(nb2.conditionals["yes"][2]["excellent"], 3/10)  # P(credit_rating=excellent|yes)

    assert np.isclose(nb2.conditionals["no"][0][1], 3/5)  # P(standing=1|no)
    assert np.isclose(nb2.conditionals["no"][0][2], 2/5)  # P(standing=2|no)
    assert np.isclose(nb2.conditionals["no"][1][1], 1/5)  # P(job_status=1|no)
    assert np.isclose(nb2.conditionals["no"][1][2], 2/5)  # P(job_status=2|no)
    assert np.isclose(nb2.conditionals["no"][1][3], 2/5)  # P(job_status=3|no)
    assert np.isclose(nb2.conditionals["no"][2]["fair"], 2/5)  # P(credit_rating=fair|no)
    assert np.isclose(nb2.conditionals["no"][2]["excellent"], 3/5)  # P(credit_rating=excellent|no)

    # Test Case 3: Bramer 3.2 dataset
    nb3 = MyNaiveBayesClassifier()
    nb3.fit(X_train_train, y_train_train)

    assert np.isclose(nb3.priors["on time"], 14/20)
    assert np.isclose(nb3.priors["late"], 2/20)
    assert np.isclose(nb3.priors["very late"], 3/20)
    assert np.isclose(nb3.priors["cancelled"], 1/20)

    assert np.isclose(nb3.conditionals["on time"][0]["weekday"], 9/14)  # P(day=weekday|on time)
    assert np.isclose(nb3.conditionals["on time"][0]["saturday"], 2/14)  # P(day=saturday|on time)
    assert np.isclose(nb3.conditionals["on time"][0]["sunday"], 1/14)  # P(day=sunday|on time)
    assert np.isclose(nb3.conditionals["on time"][0]["holiday"], 2/14)  # P(day=holiday|on time)

    assert np.isclose(nb3.conditionals["on time"][1]["spring"], 4/14)  # P(season=spring|on time)
    assert np.isclose(nb3.conditionals["on time"][1]["summer"], 6/14)  # P(season=summer|on time)
    assert np.isclose(nb3.conditionals["on time"][1]["autumn"], 2/14)  # P(season=autumn|on time)
    assert np.isclose(nb3.conditionals["on time"][1]["winter"], 2/14)  # P(season=winter|on time)

    assert np.isclose(nb3.conditionals["on time"][2]["none"], 5/14)  # P(wind=none|on time)
    assert np.isclose(nb3.conditionals["on time"][2]["normal"], 5/14)  # P(wind=normal|on time)
    assert np.isclose(nb3.conditionals["on time"][2]["high"], 4/14)  # P(wind=high|on time)

    assert np.isclose(nb3.conditionals["on time"][3]["none"], 5/14)  # P(rain=none|on time)
    assert np.isclose(nb3.conditionals["on time"][3]["slight"], 8/14)  # P(rain=slight|on time)
    assert np.isclose(nb3.conditionals["on time"][3]["heavy"], 1/14)  # P(rain=heavy|on time)

def test_naive_bayes_classifier_predict():
    # in-class Naive Bayes example (lab task #1)
    header_inclass_example = ["att1", "att2"]
    X_train_inclass_example = [
        [1, 5], # yes
        [2, 6], # yes
        [1, 5], # no
        [1, 5], # no
        [1, 6], # yes
        [2, 6], # no
        [1, 5], # yes
        [1, 6] # yes
    ]
    y_train_inclass_example = ["yes", "yes", "no", "no", "yes", "no", "yes", "yes"]

    # iPhone purchases dataset
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

    # Bramer 3.2 train dataset
    header_train = ["day", "season", "wind", "rain"]
    X_train_train = [
        ["weekday", "spring", "none", "none"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "none", "slight"],
        ["weekday", "winter", "high", "heavy"],
        ["saturday", "summer", "normal", "none"],
        ["weekday", "autumn", "normal", "none"],
        ["holiday", "summer", "high", "slight"],
        ["sunday", "summer", "normal", "none"],
        ["weekday", "winter", "high", "heavy"],
        ["weekday", "summer", "none", "slight"],
        ["saturday", "spring", "high", "heavy"],
        ["weekday", "summer", "high", "slight"],
        ["saturday", "winter", "normal", "none"],
        ["weekday", "summer", "high", "none"],
        ["weekday", "winter", "normal", "heavy"],
        ["saturday", "autumn", "high", "slight"],
        ["weekday", "autumn", "none", "heavy"],
        ["holiday", "spring", "normal", "slight"],
        ["weekday", "spring", "normal", "none"],
        ["weekday", "spring", "normal", "slight"]
    ]
    y_train_train = ["on time", "on time", "on time", "late", "on time", "very late", "on time",
                    "on time", "very late", "on time", "cancelled", "on time", "late", "on time",
                    "very late", "on time", "on time", "on time", "on time", "on time"]

    # Test Case 1: In-class example - test with [2, 6]
    nb1 = MyNaiveBayesClassifier()
    nb1.fit(X_train_inclass_example, y_train_inclass_example)

    y_pred1 = nb1.predict([[2, 6]])
    assert y_pred1 == ["yes"]

    # Test Case 2: iPhone dataset
    nb2 = MyNaiveBayesClassifier()
    nb2.fit(X_train_iphone, y_train_iphone)

    y_pred2_test1 = nb2.predict([[2, 2, "fair"]])
    assert y_pred2_test1 == ["yes"]

    y_pred2_test2 = nb2.predict([[1, 1, "excellent"]])
    assert y_pred2_test2 == ["no"]

    nb3 = MyNaiveBayesClassifier()
    nb3.fit(X_train_train, y_train_train)

    y_pred3 = nb3.predict([["weekday", "winter", "high", "heavy"]])
    assert y_pred3 == ["very late"] 

    y_pred3_ex1 = nb3.predict([["weekday", "summer", "high", "heavy"]])
    assert y_pred3_ex1 == ["on time"] 

    y_pred3_ex2 = nb3.predict([["sunday", "summer", "normal", "slight"]])
    assert y_pred3_ex2 == ["on time"]  