import pandas as pd
import matplotlib.pyplot as plt

__all__ = [
    'graph_results',
    'graph_results_multiple',
]


def graph_results(prod_records: pd.DataFrame, predicted_bbls: pd.Series, label='Predicted'):
    """
    Graph the actual and predicted production for a well.

    :param prod_records: A dataframe of preprocessed monthly production
     records.
    :param predicted_bbls: A series of predicted BBLs (per day). Should
     represent the same number of days as in the actual records.
    :param label: A plot label for the prediction.
    """
    x_monthly = prod_records['calendar_days'].cumsum()
    y_actual = prod_records['bbls_per_calendar_day']
    x_daily = pd.Series(range(1, len(predicted_bbls) + 1))
    ax = plt.subplot()
    ax.set_title('Daily Production')
    ax.set_xlabel('Days')
    ax.set_ylabel('BBLs Produced')
    ax.plot(x_monthly, y_actual, label='Actual')
    ax.plot(x_daily, predicted_bbls, label=label)
    ax.legend(loc='upper right')
    return ax


def graph_results_multiple(
        prod_records: pd.DataFrame,
        all_predicted_bbls: list,
        prediction_labels: list = None):
    """
    Graph the actual production for a well, along with predictions by
    multiple models.

    :param prod_records: A dataframe of preprocessed monthly production
     records.
    :param all_predicted_bbls: A list of series of predicted BBLs (per
     day). Each should represent the same number of days as in the
     actual records.
    :param prediction_labels: (Optional) A list of the labels for the
     predictions. If passed, this list should be the same length as
     ``all_predicted_bbls``. If not passed, will be labeled:
      ``'Predicted (1)'``, ``'Predicted (2)'``, etc.
    """
    x_monthly = prod_records['calendar_days'].cumsum()
    y_actual = prod_records['bbls_per_calendar_day']
    x_daily = pd.Series(range(1, len(all_predicted_bbls[0]) + 1))
    ax = plt.subplot()
    ax.set_title('Daily Production')
    ax.set_xlabel('Days')
    ax.set_ylabel('BBLs Produced')
    ax.plot(x_monthly, y_actual, label='Actual')
    ax.legend(loc='upper right')
    n_predictions = len(all_predicted_bbls)
    if prediction_labels is None:
        if n_predictions == 1:
            prediction_labels = [f"Predicted"]
        else:
            prediction_labels = [f"Predicted ({i + 1})" for i in range(1, n_predictions + 1)]
    for predicted_bbls, label in zip(all_predicted_bbls, prediction_labels):
        ax.plot(x_daily, predicted_bbls, label=label)
    ax.legend(loc='upper right')
    plt.show()
