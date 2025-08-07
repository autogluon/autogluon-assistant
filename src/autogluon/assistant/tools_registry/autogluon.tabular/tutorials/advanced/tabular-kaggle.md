Summary: This tutorial demonstrates how to use AutoGluon for fraud detection in the IEEE Fraud detection competition. AutoGluon's TabularPredictor can be used to train a model for binary classification with minimal code. It shows how to use TabularPredictor for binary classification with TabularPredictor for binary classification with TabularPredictor for binary classification.

```
import pandas as pd
import numpy as np
from autogluon.tabular import TabularPredictor

directory = '~/IEEEfraud/'  # directory where you have downloaded the data CSV files from the competition
label = 'isFraud'  # name of target variable to predict in this competition
eval_metric = 'roc_auc'  # Optional: specify that competition evaluation metric is AUC
save_path = directory + 'AutoGluonModels/'  # where to store trained models

train_identity = pd.read_csv(directory+'train_identity.csv')
train_transaction = pd.read_csv(directory+'train_transaction.csv')
```


Since the training data for this competition is comprised of multiple CSV files, we just first join them into a single large table (with rows = examples, columns = features) before applying AutoGluon:

```
train_data = pd.merge(train_transaction, train_identity, on='TransactionID', how='left')
```


Note that a left-join on the `TransactionID` key happened to be most appropriate for this Kaggle competition, but for others involving multiple training data files, you will likely need to use a different join strategy (always consider this very carefully). Now that all our training data resides within a single table, we can apply AutoGluon. Below, we specify the `presets` argument to maximize AutoGluon's predictive accuracy which usually requires that you run `fit()` with longer time limits (3600s below should likely be increased in your run):

```
predictor = TabularPredictor(label=label, eval_metric=eval_metric, path=save_path, verbosity=3).fit(
    train_data, presets='best_quality', time_limit=3600
)

results = predictor.fit_summary()
```


Now, we use the trained AutoGluon Predictor to make predictions on the competition's test data. It is imperative that multiple test data files are joined together in the exact same manner as the training data. Because this competition is evaluated based on the AUC (Area under the ROC curve) metric, we ask AutoGluon for predicted class-probabilities rather than class predictions. In general, when to use `predict` vs `predict_proba` will depend on the particular competition.

```
test_identity = pd.read_csv(directory+'test_identity.csv')
test_transaction = pd.read_csv(directory+'test_transaction.csv')
test_data = pd.merge(test_transaction, test_identity, on='TransactionID', how='left')  # same join applied to training files

y_predproba = predictor.predict_proba(test_data)
y_predproba.head(5)  # some example predicted fraud-probabilities
```


When submitting predicted probabilities for classification competitions, it is imperative these correspond to the same class expected by Kaggle. For binary classification tasks, you can see which class AutoGluon's predicted probabilities correspond to via:

```
predictor.positive_class
```


For multiclass classification tasks, you can see which classes AutoGluon's predicted probabilities correspond to via:

```
predictor.class_labels  # classes in this list correspond to columns of predict_proba() output
```


Now, let's get prediction probabilities for the entire test data, while only getting the positive class predictions by specifying:

```
y_predproba = predictor.predict_proba(test_data, as_multiclass=False)
```


Now that we have made a prediction for each row in the test dataset, we can submit these predictions to Kaggle. Most Kaggle competitions provide a sample submission file, in which you can simply overwrite the sample predictions with your own as we do below:

```
submission = pd.read_csv(directory+'sample_submission.csv')
submission['isFraud'] = y_predproba
submission.head()
submission.to_csv(directory+'my_submission.csv', index=False)
```


We have now completed steps (4)-(6) from the top of this tutorial. To submit your predictions to Kaggle, you can run the following command in your terminal (from the appropriate directory):

`kaggle competitions submit -c ieee-fraud-detection -f sample_submission.csv -m "my first submission"`

You can now play with different `fit()` arguments and feature-engineering techniques to try and maximize the rank of your submissions in the Kaggle Leaderboard!


**Tips to maximize predictive performance:**

   - Be sure to specify the appropriate evaluation metric if one is specified on the competition website! If you are unsure which metric is best, then simply do not specify this argument when invoking `fit()`; AutoGluon should still produce high-quality models by automatically inferring which metric to use.

   - If the training examples are time-based and the competition test examples come from future data, we recommend you reserve the most recently-collected training examples as a separate validation dataset passed to `fit()`. Otherwise, you do not need to specify a validation set yourself and AutoGluon will automatically partition the competition training data into its own training/validation sets.

   - Beyond simply specifying `presets = 'best_quality'`, you may play with more advanced `fit()` arguments such as: `num_bag_folds`, `num_stack_levels`, `num_bag_sets`, `hyperparameter_tune_kwargs`, `hyperparameters`, `refit_full`. However we recommend spending most of your time on feature-engineering and just specify `presets = 'best_quality'` inside the call to `fit()`.


**Troubleshooting:**

- Check that you have the right user-permissions on your computer to access the data files downloaded from Kaggle.

- For issues downloading Kaggle data or submitting predictions, check your Kaggle account setup and the [Kaggle FAQ](https://www.kaggle.com/general/14438).
