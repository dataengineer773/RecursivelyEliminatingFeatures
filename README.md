We want to automatically select the best features to keep, Use scikit-learn’s RFECV to conduct recursive feature elimination (RFE) using cross-validation (CV).
That is, use the wrapper feature selection method and repeatedly train a model, each time removing a
feature until model performance (e.g., accuracy) becomes worse. The remaining features are the best, Once we have conducted RFE, we can see the number of features
we should keep, We can also see which of those features we should keep, We can even view the rankings of the features, This is likely the most advanced recipe in this book up to this point, combining a number of topics we
have yet to address in detail. However, the intuition is straightforward enough that we can address it
here rather than holding off until a later chapter. The idea behind RFE is to train a model repeatedly,
updating the weights or coefficients of that model each time. The first time we train the model, we
include all the features. Then, we find the feature with the smallest parameter (notice that this assumes
the features are either rescaled or standardized), meaning it is less important, and remove the feature
from the feature set.
The obvious question then is: how many features should we keep? We can (hypothetically) repeat this
loop until we only have one feature left. A better approach requires that we include a new concept
called cross-validation (CV). We will discuss cross-validation in detail in the next chapter, but here is
the general idea.
Given data containing 1) a target we want to predict and 2) a feature matrix, first we split the data into
two groups: a training set and a test set. Second, we train our model using the training set. Third, we
pretend that we do not know the target of the test set, and apply our model to the test set’s features in
order to predict the values of the test set. Finally, we compare our predicted target values with the true
target values to evaluate our model.
We can use CV to find the optimum number of features to keep during RFE. Specifically, in RFE with
CV after every iteration, we use cross-validation to evaluate our model. If CV shows that our model
improved after we eliminated a feature, then we continue on to the next loop. However, if CV shows
that our model got worse after we eliminated a feature, we put that feature back into the feature set and
select those features as the best.
In scikit-learn, RFE with CV is implemented using RFECV and contains a number of important
parameters. The estimator parameter determines the type of model we want to train (e.g., linear
regression). The step parameter sets the number or proportion of features to drop during each loop. The
scoring parameter sets the metric of quality we use to evaluate our model during cross-validation
