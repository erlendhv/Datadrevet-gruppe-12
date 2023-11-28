from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

def print_metrics(modelStr, Y_test=None, Y_pred=None, display_conf_matrix=False):

    print("\n---------")
    print(f"\nMetrics for {modelStr}:\n")

    # Define metrics
    class_report = classification_report(Y_test, Y_pred)
    acc_score = accuracy_score(Y_test, Y_pred)
    conf_matrix = confusion_matrix(list(Y_test), Y_pred)
    

    # Print metrics
    print(class_report)
    print(f"Accuracy for {modelStr}: \n{acc_score}")
    print(f"Confusion matrix for {modelStr}: \n{conf_matrix}\n")

    if display_conf_matrix:

        # Display metrics
        disp = ConfusionMatrixDisplay(conf_matrix)
        disp.plot()
        plt.show()

def print_avg_metrics(modelStr, num_runs: int, accs: list, f1_score:list):

    print("\n---------")
    print(f"\n Average Metrics for {modelStr}:\n")

    print(f"Average accuracy for MLP after {num_runs} run{'s'*min(num_runs-1,1)}: {sum(accs)/num_runs}")
    print(f"Average f1_score for MLP after {num_runs} run{'s'*min(num_runs-1,1)}: {sum(accs)/num_runs}")