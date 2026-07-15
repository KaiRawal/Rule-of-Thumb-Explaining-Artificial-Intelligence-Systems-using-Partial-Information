import shap
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import gradio as gr
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from rule_of_thumb import RuleOfThumb
import torch
from sklearn.svm import SVC
import os
from sklearn.model_selection import GridSearchCV
import json

FEATS = 8

# Define global variables
bb_model = None
x_test = None
x_train = None
feature_names = None
scaler = None
rot_model = None

def set_seeds(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_train_model():
    """
    Load Pima Indians Diabetes dataset and train a logistic regression model
    """
    global bb_model, x_test, x_train, feature_names, scaler, rot_model
    
    # Load the Pima Indians Diabetes dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'Blood\nPressure', 'Skin\nThickness', 'Insulin', 'BMI', 'Diabetes\nPedigree\nFunction', 'Age', 'Outcome']
    diabetes_df = pd.read_csv(url, names=column_names)
    
    x_data = diabetes_df.iloc[:, :-1].values
    y_data = diabetes_df.iloc[:, -1].values
    
    scaler = StandardScaler()

    x_scaled = scaler.fit_transform(x_data)

    # print("Original x_data:")
    # for i, feature in enumerate(column_names[:FEATS]):
    #     print(f"{feature}: min={x_data[:, i].min()}, mean={x_data[:, i].mean()}, max={x_data[:, i].max()}")

    # print("\nScaled x_scaled:")
    # for i, feature in enumerate(column_names[:FEATS]):
    #     print(f"{feature}: min={x_scaled[:, i].min()}, mean={x_scaled[:, i].mean()}, max={x_scaled[:, i].max()}")
    
    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, 
        y_data, 
        test_size=0.2, 
        random_state=0
    )

    print(f'{len(y_train)=}')
    print(f'{len(y_test)=}')

    # Define the parameter grid for hyperparameter optimization
    param_grid = {
        'n_estimators': [100],
        'max_features': ['sqrt'],
        'max_depth': [10],
        'min_samples_split': [2],
        'min_samples_leaf': [4],
        'bootstrap': [True]
    }
    rf = RandomForestClassifier(random_state=0, oob_score=True)

    # Perform grid search with cross-validation
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring='accuracy')
    grid_search.fit(x_train, y_train)

    bb_model = grid_search.best_estimator_
    print(f"Best parameters found: {grid_search.best_params_}")
    print(f"Best OOB score: {bb_model.oob_score_}")

    y_pred_train = bb_model.predict(x_train)
    y_pred_test = bb_model.predict(x_test)
    
    train_accuracy = np.mean(y_pred_train == y_train)
    test_accuracy = np.mean(y_pred_test == y_test)
    print(f"Model Train Accuracy: {train_accuracy}")
    print(f"Model Test Accuracy: {test_accuracy}")


    rot = RuleOfThumb(y_pred_train, x_train)
    rot_preds_train = rot._explainer_model.predict(torch.from_numpy(x_train).to(torch.float)).detach().numpy()
    rot_preds_test = rot._explainer_model.predict(torch.from_numpy(x_test).to(torch.float)).detach().numpy()
    
    train_accuracy = np.mean(rot_preds_train == y_train)
    test_accuracy = np.mean(rot_preds_test == y_test)
    print(f"RoT Train Accuracy: {train_accuracy}")
    print(f"RoT Test Accuracy: {test_accuracy}")

    rot_model = rot

    feature_names = column_names[:FEATS]

    # rot_exps = rot.get_explanation(x_test)
    # explainer = shap.KernelExplainer(bb_model.predict_proba, x_test)
    # shap_exps = explainer.shap_values(x_test)
    # pd.DataFrame(rot_exps, columns=feature_names).to_csv('RoT_explanations.csv')
    # pd.DataFrame(shap_exps[:,:,1], columns=feature_names).to_csv('SHAP_explanations.csv')
    

def get_model_and_data():
    return bb_model, x_test, feature_names


def generate_datapoint_19_figures():
    """Generate and save explanations for datapoint 19 using existing plot functions."""
    model, background_data, names = get_model_and_data()

    if background_data is None or len(background_data) <= 19:
        print(f"[warning] datapoint 19 unavailable (x_test size={0 if background_data is None else len(background_data)})")
        return

    x_example_scaled = background_data[19, :].reshape(1, -1)
    print("[info] Generating SHAP and RoT figures for datapoint 19")

    generate_shap_plot(model, x_example_scaled, background_data, names)
    generate_rot_plot(model, x_example_scaled, background_data, names, 'identity')
    generate_rot_plot(model, x_example_scaled, background_data, names, 'logit')

def generate_examples():
    """
    Create a DataFrame with min, max, and mean values for each feature in the Pima Indians Diabetes dataset.
    """
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    diabetes_df = pd.read_csv(url, names=column_names)
    data, names = diabetes_df.iloc[:, :FEATS].values, column_names[:FEATS]
    diabetes_df = pd.DataFrame(data, columns=names)
    
    min_values = diabetes_df.min()
    mean_values = diabetes_df.mean()
    max_values = diabetes_df.max()
    
    examples_df = pd.DataFrame({
        'min': min_values,
        'mean': mean_values,
        'max': max_values
    })

    return examples_df.T

def generate_rot_plot(model, x_example, dataset, feature_names, link='identity'):
    """
    Generate RoT force plot using matplotlib
    """
    plt.clf()
    predicted_class = model.predict(x_example)[0]
    output_marker = float(model.predict_proba(x_example)[0,1])
    final_pred = float(np.round(model.predict_proba(x_example)[0,1] - 0.01, 2)) 
    print(f'{final_pred=}')

    rot_xv = rot_model._explainer_model.g.detach().numpy()[1]
    print(f"{rot_xv=}")
    rot_exps = rot_model.get_explanation(x_example)[0]
    rot_predictions = rot_model._explainer_model.score(torch.from_numpy(x_example).to(torch.float)).detach().numpy()
    # print(f"{rot_exps.shape=}")
    print(f"{rot_exps=}")
    print(f"{rot_exps.sum()+rot_xv=}")
    print(f"{rot_predictions=}")
    print(f'CORRECTION FACTOR: \t {rot_predictions=} \t {model.predict_proba(x_example)=}')

    outnames= ''
    if link == 'logit':
        print(f'LOGIT: {[1 / (1 + np.exp(-x)) for x in rot_exps.flatten()]}')
        outnames = '~p(x)'
        # outnames = 'RoT_prob_pred(inputs)'
    if link == 'identity':
        print(f'IDENTITY: {[x for x in rot_exps.flatten()]}')
        outnames = '~logit(p(x))'
        # outnames = 'RoT_pred(inputs)'
    
    x_example_inverted = scaler.inverse_transform(np.array(x_example).reshape(1, -1)).flatten().tolist()
    x_example_inverted = np.array(x_example_inverted, dtype=np.int64)
    # x_example_inverted = [int(f) for f in x_example_inverted]
    print(f'{x_example_inverted=}')
 
    
    shap.force_plot(
        rot_xv,
        rot_exps,
        # np.round(x_example_inverted, 0),
        # [str(i)[:-2] for i in x_example_inverted],
        # np.array(x_example_inverted, dtype=np.int),
        x_example_inverted,
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        link=link,
        out_names='model prediction',
        figsize=(10,3),
        plot_cmap='BrBG',
        text_rotation=45,
        contribution_threshold=0.1,
    )
    for item in plt.gca().get_children():
        if isinstance(item, plt.Text) and str(final_pred) in item.get_text():
            pass
            item.remove()
        if isinstance(item, Line2D) and item._color == '#F2F2F2':
            pass
            # xdata, ydata = item.get_xdata(), item.get_ydata()
            # y = [ for yy in y]
            # item.set_xdata([xxx if xxx > 0.3 else output_marker for xxx in item.get_xdata()])
            # absc, ordi = item.get_position()
            # item.set_position((absc, ordi - 0.075))
            # item.remove()
        if isinstance(item, plt.Text) and "base value" in item.get_text():
            item.set_text(item.get_text())
            item.set_text("RoT baseline")
            item.set_fontsize(item.get_fontsize() + 2)  # Increase font size
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.075))
            item.set_zorder(-1)
        if isinstance(item, plt.Text) and "model prediction" in item.get_text():
            item.set_text(item.get_text())
            item.set_fontsize(item.get_fontsize() + 2)  # Increase font size
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.075))
            item.set_zorder(-1)
        if isinstance(item, plt.Text) and "higher" in item.get_text():
            item.set_text("Positive Contribution    \n(Diabetic)    ")
            item.set_fontsize(item.get_fontsize() + 3)  # Increase font size
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.0375))
            item.set_zorder(10)  # Bring the text to the top
        if isinstance(item, plt.Text) and "lower" in item.get_text():
            item.set_text("    Negative Contribution\n    (Non-Diabetic)  ")
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.0375))
            item.set_fontsize(item.get_fontsize() + 3)  # Increase font size
            item.set_zorder(10)  # Bring the text to the top
        if isinstance(item, plt.Text) and '=' in item.get_text():
            item.set_text(item.get_text()[:-2])
            item.set_fontsize(15)
            absc, ordi = item.get_position()
            if 'BMI' in item.get_text():
                item.set_position((absc-0.005, ordi + 0.1))
            elif 'Glucose' in item.get_text():
                item.set_position((absc+0.055, ordi + 0.1))
            elif 'Age' in item.get_text():
                item.set_position((absc-0.015, ordi + 0.1))
            else:
                item.set_position((absc-0.01, ordi + 0.1))
    # plt.xlim((0.2,0.5))
    plt.tight_layout()
    fig=plt.gcf()
    fig.subplots_adjust(top=0.65)
    plt.savefig(f'PLOTS/RoT_force_plot_{link}.pdf', dpi=300)
    return plt, rot_exps, rot_xv, rot_predictions

def generate_shap_plot(model, x_example, dataset, feature_names, link='identity'):
    """
    Generate SHAP force plot using matplotlib
    """
    plt.clf()
    output_marker = float(model.predict_proba(x_example)[0,1])
    explainer = shap.KernelExplainer(model.predict_proba, dataset)
    shap_values = explainer.shap_values(x_example)[0]
    print(f"{model.predict_proba(dataset).mean(axis=0)=}")
    # print(model.predict_proba(dataset.mean(axis=0).reshape(1,-1)))
    print(f'{shap_values=}')
    
    predicted_class = model.predict(x_example)[0]
    x_example_inverted = scaler.inverse_transform(np.array(x_example).reshape(1, -1)).flatten().tolist()


    final_pred = float(np.round(model.predict_proba(x_example)[0,1],2))
    
    # for item in plt.gca().get_children():
        # if isinstance(item, plt.Text) and "base value" in item.get_text():
        #     item.set_text(" Feature Importances")
        #     item.set_fontsize(item.get_fontsize() + 6)  # Increase font size
        #     item.set_color('black')  # Set text color to black
        #     item.set_zorder(-1)  # Bring the text to the top
        # if isinstance(item, plt.Text) and '=' in item.get_text():
        #     item.set_fontsize(item.get_fontsize() + 3)
        #     # item.set_weight('bold')
        #     absc, ordi = item.get_position()
        #     if caller == 'grad' and 'BMI' in item.get_text():
        #         item.set_position((absc + 7.5, ordi - 0.2))
        #         pass
        #     elif caller == 'shap' and 'Glucose' in item.get_text():
        #         item.set_position((absc - 6.6, ordi - 0.2))
        #         pass
        #     elif caller == 'shap' and 'BMI' in item.get_text():
        #         item.set_position((absc - 5.5, ordi - 0.2))
        #         pass
        #     else:
        #         item.set_position((absc, ordi - 0.2))
    # plt.subplots_adjust(top=0.7, bottom=0.1)  # Adjust margins to reduce top and bottom space
 
    print(f'{explainer.expected_value.shape=}')
    print(f'{explainer.expected_value[1].shape=}')
    print(f'{shap_values.shape=}')
    print(f'{shap_values[:,1].shape=}')
    shap.force_plot(
        explainer.expected_value[1],
        
        # Depending on the shap + python version installed, we may need a different version from the 2 lines below:
        # shap_values.flatten(),
        shap_values[:,1],
        
        np.round(x_example_inverted, 0),
        feature_names=feature_names,
        matplotlib=True,
        show=False,
        link=link,
        # out_names='SHAP_prob_pred(inputs)=',
        out_names='model prediction',
        figsize=(10,3),
        plot_cmap='RdBu',
        text_rotation=45,
        contribution_threshold=0.1,
    )
    for item in plt.gca().get_children():
        # print()
        # print(f'{item=}')
        # ddict = {k: item.__dict__[k] for k in item.__dict__ if 'color' in str(k)}
        # print(f'{ddict=}')
        # print('==============================')
        if isinstance(item, plt.Text) and str(final_pred) in item.get_text():
            pass
            item.remove()
        if isinstance(item, Line2D) and item._color == '#F2F2F2':
            # item.set_xdata([xxx if xxx > 0.3 else output_marker for xxx in item.get_xdata()])
            pass
            # item.remove()
        if isinstance(item, plt.Text) and "base value" in item.get_text():
            # print(f'a. {item=}')
            # print(f'a. {item.__dict__=}')
            item.set_text(item.get_text())
            item.set_text("SHAP baseline")
            # item.set_color('#000000')
            item.set_fontsize(item.get_fontsize() + 2)  # Increase font size
            # item.set_zorder(5)  # Bring the text to the top
            # print(f'b. {item=}')
            # print(f'b. {item.__dict__=}')
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.075))
            item.set_zorder(-1)
        if isinstance(item, plt.Text) and "model prediction" in item.get_text():
            # print(f'1. {item=}')
            # print(f'1. {item.__dict__=}')
            item.set_text(item.get_text())
            # item.set_color("#000000")
            item.set_fontsize(item.get_fontsize() + 2)  # Increase font size
            # item.set_zorder(5)  # Bring the text to the top
            # print(f'2. {item=}')
            # print(f'2. {item.__dict__=}')
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.075))
            item.set_zorder(-1)
        if isinstance(item, plt.Text) and "higher" in item.get_text():
            item.set_text("Positive Contribution    \n(Diabetic)    ")
            item.set_fontsize(item.get_fontsize() + 3)  # Increase font size
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.0375))
            item.set_zorder(10)  # Bring the text to the top
            # absc, ordi = item.get_position()
            # item.set_position((absc, ordi + 0.9))  # Move the text a little higher up
        if isinstance(item, plt.Text) and "lower" in item.get_text():
            # print(f'3. {item=}')
            # print(f'3. {item.__dict__=}')
            item.set_text("    Negative Contribution\n    (Non-Diabetic)  ")
            absc, ordi = item.get_position()
            item.set_position((absc, ordi - 0.0375))
            # item.set_text("  Non-diabetic")
            item.set_fontsize(item.get_fontsize() + 3)  # Increase font size
            item.set_zorder(10)  # Bring the text to the top
            # absc, ordi = item.get_position()
            # item.set_position((absc, ordi + 0.2))
            # print(f'4. {item=}')
            # print(f'4. {item.__dict__=}')
        if isinstance(item, plt.Text) and '=' in item.get_text():
            item.set_text(item.get_text()[:-2])
            item.set_fontsize(15)
            # item.set_weight('bold')
            absc, ordi = item.get_position()
            if 'Glucose' in item.get_text():
                item.set_position((absc+0.045, ordi + 0.1))
            else:
                item.set_position((absc-0.01, ordi + 0.1))
            # item.set_position((absc-0.01, ordi + 0.1))
    # plt.xlim((0.1,0.5))
    plt.tight_layout()
    fig=plt.gcf()
    fig.subplots_adjust(top=0.65)
    plt.savefig(f'PLOTS/SHAP_force_plot.pdf', dpi=300)
    return plt, shap_values, explainer.expected_value



def resetter(_):
    return 4, 120, 70, 21, 80, 32, 0.47, 33

def loader(index):
    result = scaler.inverse_transform(x_test[index, :].reshape(1, -1)).flatten().tolist()
    return result

def gradio_interface(pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age):
    """
    Gradio interface function
    """
    x_example = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    x_example_scaled = scaler.transform(x_example)
    
    imp_per_feature = [0,0,0,0,0,0,0,0,0,0,0,0,0]
    
    model, background_data, feature_names = get_model_and_data()
    
    
    plt_figure, shaps, shap_xv = generate_shap_plot(model, x_example_scaled, background_data, feature_names)
    plt_figure.savefig('shap_force_plot.png')  # Save the plot to a local PNG file


    rot_figure_i, rots_i, rot_xv_i, rot_predictions_i = generate_rot_plot(model, x_example_scaled, background_data, feature_names, 'identity')
    rot_figure_i.savefig('rot_force_plot_i.png')  # Save the plot to a local PNG file
    
    rot_figure_l, rots_l, rot_xv_l, rot_predictions_l = generate_rot_plot(model, x_example_scaled, background_data, feature_names, 'logit')
    # rot_figure_l, rots_l, rot_predictions_l = generate_shap_plot(model, x_example_scaled, background_data, feature_names, 'logit')
    rot_figure_l.savefig('rot_force_plot_l.png')  # Save the plot to a local PNG file

    
    prediction = model.predict(x_example_scaled)[0]
    pred_proba = model.predict_proba(x_example_scaled)
    pred_class = prediction
    
    return format_outputs(imp_per_feature, rot_xv_i, shap_xv, x_example_scaled, pred_class, pred_proba, rot_predictions_i, shaps, rots_i)
    # return format_outputs(f'ROT expected value: {rot_xv}', f'SHAP expected value: {shap_xv}', f"Input features (scaled): {x_example_scaled}", 
    #         f"Predicted class: {pred_class} (Probability: {pred_proba})", f"RoT prediction: {rot_predictions}", 
    #         'shap_force_plot.png', 'rot_force_plot.png', shaps, rots, extended=True)

def format_outputs(imp_per_feature, rot_xv_i, shap_xv, x_example_scaled, pred_class, pred_proba, rot_predictions_i, shaps, rots_i, extended=True):
    rot_plot = 'rot_force_plot_i.png'
    

    # Check the file size of 'rot_force_plot_l.png'
    if os.path.getsize('rot_force_plot_l.png') > 8 * 1024:  # 8 KB
        rot_plot = 'rot_force_plot_l.png'

    if extended:
        return (
            imp_per_feature[0],
            imp_per_feature[1],
            imp_per_feature[2],
            imp_per_feature[3],
            imp_per_feature[4],
            imp_per_feature[5],
            imp_per_feature[6],
            imp_per_feature[7],
            f'ROT expected value: {rot_xv_i}',
            f'SHAP expected value: {shap_xv}',
            f"Input features (scaled): {x_example_scaled}", 
            f"Predicted class: {pred_class} (Probability: {pred_proba})",
            f"RoT prediction: {rot_predictions_i}", 
            'shap_force_plot.png',
            rot_plot,
            # 'rot_force_plot_i.png',
            # 'rot_force_plot_l.png',
            shaps,
            rots_i
        )
    else:
        return (
            'shap_force_plot.png',
            rot_plot
            # 'rot_force_plot_i.png',
            # 'rot_force_plot_l.png'
        )

# Add custom CSS to match the height of lindex to tindex
custom_css = """
#lindex_button {
    height: 5.7em !important; /* Adjust this value to match the height of the tindex input */
}
"""

with gr.Blocks(css=custom_css) as iface:
    gr.Markdown("# SHAP and RoT explanations for a RandomForest Classifier for Diabetes Prediction")
    gr.Markdown("""
        A random forest model has been trained to predict diabetes using the 'pima-indians' dataset. <br />
        Enter **your own numbers** for patient measurements (fiddle the sliders), or use one of the **154 real-world preset inputs** (click "Load Preset Input Values"). <br />
        Finally click "Predict Diabetes and Explain Model Output" to see visualisations for the SHAP and RoT explanations.
    """)
    
    with gr.Accordion("View and Change Model Inputs", open=True):
        with gr.Row():
            with gr.Column():
                pregnancies = gr.Slider(label="Pregnancies", minimum=0, maximum=17, step=1, value=4)
                glucose = gr.Slider(label="Glucose", minimum=0, maximum=200, step=1, value=120.0)
                blood_pressure = gr.Slider(label="Blood Pressure", minimum=0, maximum=122, step=1, value=70.0)
                skin_thickness = gr.Slider(label="Skin Thickness", minimum=0, maximum=99, step=1, value=21.0)
            with gr.Column():
                insulin = gr.Slider(label="Insulin", minimum=0, maximum=846, step=5, value=80.0)
                bmi = gr.Slider(label="BMI", minimum=0, maximum=67, step=1, value=32.0)
                diabetes_pedigree_function = gr.Slider(label="Diabetes Pedigree Function", minimum=0.08, maximum=2.42, step=0.01, value=0.47)
                age = gr.Slider(label="Age", minimum=21, maximum=81, step=1, value=33.0)
    inp=[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]

    with gr.Row():
        tindex = gr.Number(label="Preset Input Index (0-153)", value=0, minimum=0, maximum=153)
        lindex = gr.Button("Load Preset Input Values", elem_id="lindex_button")
    prex = gr.Examples(
        fn=loader,
        inputs=[tindex], 
        outputs=inp,
        # examples=[15, 19, 40, 49, 53, 55, 61, 68, 75, 90, 99, 105, 107, 113, 127, 141, 144], 
        examples=[10,15,19,40,49,53,55,61,68,75,82,90,99,105,107,113,127,133,136,137,141,144],
        cache_examples=False, 
        run_on_click=True, 
        examples_per_page=25, 
        label="some interesting real-world inputs where SHAP and RoT differ"
    )

    submit = gr.Button("Predict Diabetes and Explain Model Output", variant="primary")

    
    with gr.Row():
        shap_force_plot = gr.Image(label="SHAP Force Plot")
        rot_force_plot = gr.Image(label="RoT Force Plot")
        # rot_force_plot_logits = gr.Image(label="RoT Force Plot - logits")
    
    
    with gr.Accordion("Debug Information", open=False):
        with gr.Row():
            shap_xv = gr.Text(label="shap-xv")
            rot_xv = gr.Text(label="rot-xv")
            inputs = gr.Text(label="INPUTS")
            model_prediction = gr.Text(label="Model Prediction")
            rot_prediction = gr.Text(label="RoT Prediction")
        
        with gr.Row():
            shap_values = gr.Text(label="SHAP Values")
            rot_values = gr.Text(label="RoT Values")

        with gr.Column():
            with gr.Row():
                in_feats = ['Pregnancies', 'Glucose', 'Blood\nPressure', 'Skin\nThickness']
                gti1 = [gr.Number(label=fname) for fname in in_feats]
            with gr.Row():
                in_feats = ['Insulin', 'BMI', 'Diabetes\nPedigree\nFunction', 'Age']
                gti2 = [gr.Number(label=fname) for fname in in_feats]
    
    reset = gr.Button("Reset All Inputs to Original (Average) Values")

    outp=[shap_force_plot, rot_force_plot]
    outp=[shap_xv, rot_xv, inputs, model_prediction, rot_prediction, shap_force_plot, rot_force_plot, shap_values, rot_values]
    outp=gti1+gti2+outp

    submit.click(
        fn=gradio_interface,
        inputs=inp,
        outputs=outp,
    )

    reset.click(
        fn=resetter,
        inputs=[],
        outputs=inp,
    )
    
    gr.on(
        triggers=[tindex.submit, lindex.click],
        fn=loader,
        inputs=tindex,
        outputs=inp,
    )

    gr.Examples(inputs=inp, examples=generate_examples().values.tolist(), label="min, avg, and max inputs")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--non-interactive", action="store_true", help="Run setup only and skip launching Gradio")
    args = parser.parse_args()

    set_seeds()
    load_and_train_model()

    if args.non_interactive or os.getenv("ROT_NON_INTERACTIVE", "0") == "1":
        generate_datapoint_19_figures()
        print("[info] non-interactive mode: skipping Gradio launch.")
        raise SystemExit(0)

    iface.launch(share=True)