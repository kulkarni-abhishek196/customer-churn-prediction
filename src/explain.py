import shap
import matplotlib.pylab as plt

def explain_model(clf, X_test):

    model = clf.named_steps['xgboost']

    preprocessor = clf.named_steps['preprocessor']
    X_test_transformed = preprocessor.transform(X_test)

    feature_names = (
        clf.named_steps['preprocessor']
        .get_feature_names_out()
    )

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_transformed)

    #plot1
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        plot_type='bar',
        show=False
    )
    plt.title('Feature importance (SHAP)')
    plt.tight_layout()
    plt.savefig('outputs/shap_importance.png')
    plt.show()

    #plot2 - direction of impact
    plt.figure()
    shap.summary_plot(
        shap_values,
        X_test_transformed,
        feature_names=feature_names,
        show=False
    )
    plt.title('SHAP summary plot')
    plt.tight_layout()
    plt.savefig('outputs/shap_summary.png')
    plt.show()

