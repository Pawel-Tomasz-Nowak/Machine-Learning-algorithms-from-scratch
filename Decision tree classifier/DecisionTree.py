import polars as pl
import numpy as np
from sklearn.preprocessing import KBinsDiscretizer as KBD
from sklearn.model_selection import train_test_split as array_split


class Node():
    """A class representing the nodes of a decision tree"""

    def __init__(self, dataset:pl.DataFrame, target_var: str, depth:int = 0) -> None:
        """A constructor of the node. 

        Parameters:
        ---------
        dataset : pl.DataFrame
        A dataset associated with the node.
        
        target_var : str 
        A string representing the name of the target variable we are willing to classify.    
        
        depth : int = 0
        A distance from root to the node.
        
    
        """
        
        self.dataset:pl.DataFrame = dataset #The dataset for the node.

        self.depth:int = depth #The depth of the node.
        self.is_leaf:bool = False #Indicator whether the node is a leaf.

        self.left_node: Node = None #Reference to left kid-node.
        self.right_node: Node = None #Reference to right kid-node.

        self.target_var: str = target_var #The name of the target variable.
        self.n: int = self.dataset.shape[0] #The total number of observations.


    def compute_gini_impurity_index(self,y:pl.Series) -> float:
        """The function computes the gini impurity index for sample y. 
        Parameters:
        ---------
        y:pl.Series:
            A polars series representing the sample of the categorical variable 'y'.
        Returns:
        ---------
        gini_impurity_index: float
            A floating-point number between 0 and 1 representing the impurity rate with 0 indicating perfect purity and 1 indicating perfect impurity.


        """
        probabilities:pl.Series = y.value_counts(sort = False)["count"]/y.shape[0] #Find the relatives frequencies of the levels of y variable.
        squared_probabilities:pl.Series = probabilities.pow(2) #Square these probabilities.


        gini_impurity_index:float = 1 - squared_probabilities.sum() #Finally, compute the gini index.

        return gini_impurity_index
    

    def compute_total_gini_impurity_index(self, split_attr:str, split_value:float) -> float:
        """The function computes the total gini impurity index for 'split_attr' feature and 'split_value' splitting value.
        Parameters:
        ---------
        split_attr: str
        Name of the feature which will be used as a splitting attribute.

        split_value:float
        A floating-point value of the 'split_attr' feature with respect to which splitting process will be performed.
        
            
        Returns:
        ---------
        total_gini_impurity_index:float
            A floating-point values between 0 and 1 indicating the total impurity of the splitting process with 1 being the maximum impurity and 0 being
            maximum purity.

        """
        #assert self.dataset[split_attr].min()<=split_value <= self.dataset[split_attr].max(), "A splitting value should lay in the range of the spliting attribute"

        # Create left and right sub-DataFrames.
        left_data = self.dataset.filter(pl.col(split_attr) <= split_value)
        right_data = self.dataset.filter(pl.col(split_attr) > split_value)

        n_left, n_right = left_data.shape[0], right_data.shape[0]  # Numbers of rows in each group.
        

        # Calculate Gini impurity for both groups.
        left_impurity_index = self.compute_gini_impurity_index(y=left_data[self.target_var])
        right_impurity_index = self.compute_gini_impurity_index(y=right_data[self.target_var])

        # Compute the total impurity index
        total_impurity_index = (n_left * left_impurity_index + n_right * right_impurity_index) / self.n


        return total_impurity_index
    

    def find_splitting_values(self, feature:str) -> np.ndarray:
        """ The function finds the quantiles of 0.1, 0.2, 0.3, .. 0.9 rank respectively

        Parameters:
        ---------
        feature: str
        Name of the feature whose means are computed.
        
            
        Returns: 
        ---------
        quantiles : np.ndarray
        an array of quantiles of 0.1, 0.2, ..., 0.9 rank

        """
        feature_vals: pl.Series = self.dataset[feature] #Find sample values of the feature
        
        splitting_values:np.ndarray = np.quantile(feature_vals, q = np.arange(0.1, 1, 0.1)) #Now compute the quantiles.

        return splitting_values


    def find_optimal_split(self) -> None:
        """The function finds the pair (attribute, attribute_value) for which the node-split has the minimum
        gini impurity index.

        Parameters :
        ---------

        Returns :
        ---------
        None

        """
        #List of candidates for optimal splitting attribute.
        possible_split_attr:list[str] = self.dataset.select((pl.selectors.numeric())
                                                  & 
                                                  ((~pl.selectors.matches(pattern = self.target_var)))
                                                  ).columns
        
        self.split_attr: str = "" #Declare the attribute of the Object: split_attr.
        self.split_value: float | None = None #By default, split_value is equal to None. This attribute will be overridden immediately in the loop.

        min_gini_index:float = 1.5 #The total gini impurity index is always less than 1. That's why 1.5 can be treated as 'infinity' in this context.  

        #For each feature, find it's minimal gini index and the splitting value for which the minimum is reached.
        for feature in possible_split_attr:
            feature_means:np.nmdarray = self.find_splitting_values(feature = feature)

            gini_impurity_indices:np.ndarray = np.vectorize(pyfunc = self.compute_total_gini_impurity_index,
                                                 otypes = [float], excluded = ["split_attr"]
                                                 )(split_attr = feature,
                                                   split_value = feature_means)
            
            min_idx: int = np.argmin(gini_impurity_indices) #Index of the smallest gini_impurity index for given feature.
            feat_min_gini_index: float = gini_impurity_indices[min_idx] #Find the value of the smallest gini impurity index for given feature.

            if feat_min_gini_index <= min_gini_index:
                self.split_attr = feature
                self.split_value = feature_means[min_idx]

                min_gini_index = feat_min_gini_index

        self.split_attr_id = possible_split_attr.index(self.split_attr) #Find the relative position of the splitting attribute.
    

    def split_the_node(self, n_min_samples:int, max_depth:int) -> tuple[bool, bool]:
        """The methods creates two child nodes possible (if such a split is possible).
        The split is possible only if (depth of node +1) <= max_depth and 
        there is a child node (left or right) whose number of observations is not less than n_min_samples.
        The split is executed with attribute and attribute value minimizing the Gini impurity index.

        Parameters:
        n_min_samples : int
        minimum number of samples a child node has to have to be created.
        
        max_depth : int
        the maximum depth any node can reach.
        
        Returns:
        ---------
        A tuple of two boolean indicating respectively whether we can create left and right node.

        Notes:
        ---------
        Naturally if there are only False values in the returned tuple, split is not possible.

        """
        self.find_optimal_split() #First things first, find the optimal split.
      

        left_data:pl.DataFrame = self.dataset.filter(pl.col(self.split_attr) <= self.split_value)
        right_data:pl.DataFrame = self.dataset.filter(pl.col(self.split_attr) > self.split_value)

        self.is_left_valid: bool = (left_data.shape[0] >= n_min_samples) and ((self.depth+1)<=max_depth)
        self.is_right_valid: bool = (right_data.shape[0] >= n_min_samples) and ((self.depth+1)<=max_depth)

        if self.is_left_valid:
            self.left_node = Node(left_data, self.target_var, self.depth + 1)
        if self.is_right_valid:
            self.right_node = Node(right_data, self.target_var, self.depth + 1)

      
        self.is_leaf = not (self.is_left_valid or self.is_right_valid) #If neither left_node nor right_node were born, the nod is a leaf.
        
        return(self.is_left_valid, self.is_right_valid)
    

    def predict(self, x:pl.DataFrame) -> int:
        """The methods classify one observation x if the node is a leaf. Otherwise, x is passed to
        left child node or right child node accordingly.

        Parameters:
        ---------
        x : pl.DataFrame
        One observation of size (n_features)

        Returns:
        class_id : int
        An identification of a class the observation is classified to.

        """
        if self.is_leaf:
            return self.dataset[self.target_var].mode()[0]
        else:
            if self.is_left_valid and self.is_right_valid:
                if x[self.split_attr_id] <= self.split_value:
                    return self.left_node.predict(x)
                else:
                    return self.right_node.predict(x)
            else:
                if self.is_left_valid:
                    return self.left_node.predict(x)
                else:
                    return self.right_node.predict(x)


        
class Decision_Tree():
    """A class representing decision_tree-based classifier"""

    def __init__(self, max_depth: int, n_min_samples:int):
        """
        Attempts to split the current node into two child nodes based on conditions defined by `n_min_samples` and `max_depth`. 
        The split uses the attribute and attribute value that minimizes the Gini impurity index.

        A split is possible if both conditions are met:
        - The current node depth + 1 does not exceed `max_depth`.
        - There is at least one child node (left or right) with a number of observations not less than `n_min_samples`.

        Parameters:
        -----------
        n_min_samples : int
        The minimum number of samples a child node must have to be created.

        max_depth : int
        The maximum depth allowed for any node in the tree.

        Returns:
        --------
        tuple of bool
        A tuple `(can_create_left, can_create_right)`, where each boolean indicates if the respective child node can be created.

        Notes:
        ------
        If the returned tuple contains only `False` values, a split is not possible.
        """

        
        self.max_depth: int = max_depth
        self.n_min_samples: int = n_min_samples

    
        
    def fit(self, dataset:pl.DataFrame, target_var:str) -> None:
        """The method for training the tree and growing all the nodes of the tree.

        Parameters:
        ---------
        dataset : pl.DataFrame 
            The dataset from which the decision tree will be learning.
        
        target_var : str
            the target variable (that is - variable we're classifying)

            
        Returns:
        ---------
        None

        """
        self.root: Node = Node(dataset = dataset, target_var = target_var, depth = 0)
        self.target_var : str = target_var

        self.__grow_tree(node = self.root)


    def __grow_tree(self, node: Node):
        """
        Recursively creates nodes for the decision tree.
        
        When first called, the function finds child nodes for the root of the tree.
        It is then recursively called for each child node that can be created.

        Parameters:
        ----------
        node : Node
            The root node to begin growing the tree from.
            
        Returns: 
        --------
        None

        Notes:
        ------
        Upon splitting the `node` into child nodes (left or right), each split subtree can be interpreted as an independent subtree rooted at either the left or right child.
        """
        # Attempt to split the node
        child_nodes: tuple[bool, bool] = node.split_the_node(self.n_min_samples, self.max_depth)
        
        # If the node is not a leaf, recursively grow the tree for each valid child
        if not node.is_leaf:
            # Recursive calls if child nodes are created
            if child_nodes[0]:
                self.__grow_tree(node.left_node)
            if child_nodes[1]:
                self.__grow_tree(node.right_node)


    def predict(self, X:pl.DataFrame) -> int:
        """The methods classify one observation x if the node is a leaf. Otherwise, x is passed to
        left child node or right child node accordingly.

        Parameters:
        ---------
        x : pl.DataFrame
        Dataframe of observation to classify.

        Returns:
        class_id : int
        An identification of a class the observation is classified to.

        """

        return pl.Series(self.target_var, 
        values = X.map_rows(function = self.root.predict))
      