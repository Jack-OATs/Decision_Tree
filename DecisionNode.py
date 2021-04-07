class DecisionNode():
    """Class to represent a single node in
    a decision tree."""

    def __init__(self, left, right, decision_function, class_label=None):
        """Create a node with a left child, right child,
        decision function and optional class label
        for leaf nodes."""
        self.left = left
        self.right = right
        self.decision_function = decision_function
        self.class_label = class_label

    def decide(self, feature):
        """Return on a label if node is leaf,
        or pass the decision down to the node's
        left/right child (depending on decision
        function)."""
        if self.class_label is not None:
            return self.class_label
        elif self.decision_function(feature):
            return self.left.decide(feature)
        else:
            return self.right.decide(feature)
