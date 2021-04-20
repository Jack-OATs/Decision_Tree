class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function_info, class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes.
        decision_function_info = (decision_function, func_val, func_feature_index) where
          decision_function is the function to split on
          func_val is the threshold value for the function to split at
          func_feature_index is the index of the feature (or column) to split on
        """
        self.left = left
        self.right = right
        if type(decision_function_info) is tuple:
            self.decision_function, self.func_val, self.func_feature_index = decision_function_info
        else:
            self.decision_function, self.func_val, self.func_feature_index = decision_function_info, None, None
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        else:
            if self.func_val is None:
                if self.decision_function(feature):
                    return self.left.decide(feature)
                else:
                    return self.right.decide(feature)
            else:
                if self.decision_function(feature, self.func_val, self.func_feature_index):
                    return self.left.decide(feature)
                else:
                    return self.right.decide(feature)
