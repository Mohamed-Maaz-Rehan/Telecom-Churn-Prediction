import Models
import seaborn as sns
import matplotlib.pyplot as plt


def metrics(X_tr, Y_tr, X_te, Y_tes, label=''):
    print('\n', label, '\n')
    results_df = Models.models(X_tr, Y_tr, X_te, Y_tes)
    print(results_df.to_string())
    return results_df


def visualize_metrics(results_df):
    sns.set(style="whitegrid")

    plt.figure(figsize=(15, 10))
    num_classifiers = len(results_df)

    bar_width = 0.35

    index = range(num_classifiers)

    # Accuracy
    plt.bar(index, results_df['Training Accuracy'], bar_width, color='orange', label='Training Accuracy')
    plt.bar([x + bar_width for x in index], results_df['Testing Accuracy'], bar_width, color='skyblue',
            label='Testing Accuracy')

    plt.xlabel('Classifier')
    plt.ylabel('Accuracy')
    plt.title('Accuracy Comparison')
    plt.xticks([x + bar_width / 2 for x in index], results_df['Classifier'], rotation=45)
    plt.legend()

    plt.tight_layout()
    plt.show()

    # Precision
    plt.figure(figsize=(15, 10))
    plt.bar(index, results_df['Training Precision'], bar_width, color='orange', label='Training Precision')
    plt.bar([x + bar_width for x in index], results_df['Testing Precision'], bar_width, color='skyblue',
            label='Testing Precision')
    plt.xlabel('Classifier')
    plt.ylabel('Precision')
    plt.title('Precision Comparison')
    plt.xticks([x + bar_width / 2 for x in index], results_df['Classifier'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Recall
    plt.figure(figsize=(15, 10))
    plt.bar(index, results_df['Training Recall'], bar_width, color='orange', label='Training Recall')
    plt.bar([x + bar_width for x in index], results_df['Testing Recall'], bar_width, color='skyblue',
            label='Testing Recall')
    plt.xlabel('Classifier')
    plt.ylabel('Recall')
    plt.title('Recall Comparison')
    plt.xticks([x + bar_width / 2 for x in index], results_df['Classifier'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

    # F1-score
    plt.figure(figsize=(15, 10))
    plt.bar(index, results_df['Training F1'], bar_width, color='orange', label='Training F1-score')
    plt.bar([x + bar_width for x in index], results_df['Testing F1'], bar_width, color='skyblue',
            label='Testing F1-score')
    plt.xlabel('Classifier')
    plt.ylabel('F1-score')
    plt.title('F1-score Comparison')
    plt.xticks([x + bar_width / 2 for x in index], results_df['Classifier'], rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_auroc(fprs, tprs, aucs, classifier_names):
    plt.figure(figsize=(8, 6))
    for fpr, tpr, auc_score, clf_name in zip(fprs, tprs, aucs, classifier_names):
        sns.lineplot(x=fpr, y=tpr, label='%s (AUC = %0.2f)' % (clf_name, auc_score))

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (AUROC)')
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.show()
