import os
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def get_processed_data_and_names(clf, df_input, categorical=None, numeric=None):
    """Return processed dense matrix and feature names from a fitted pipeline.

    Args:
        clf: sklearn Pipeline with a fitted 'preprocess' step (ColumnTransformer)
        df_input: pandas DataFrame to transform
        categorical: optional list of categorical column names used when constructing fallback names
        numeric: optional list of numeric column names used when constructing fallback names

    Returns:
        X_proc: numpy.ndarray (dense)
        feature_names: list[str]
    """
    pre = clf.named_steps["preprocess"]
    X_proc = pre.transform(df_input)
    # convert sparse to dense if needed
    if hasattr(X_proc, "toarray"):
        X_proc = X_proc.toarray()
    else:
        X_proc = np.asarray(X_proc)

    feature_names = []
    # Preferred: ColumnTransformer provides get_feature_names_out in recent sklearn
    try:
        # Some versions accept no args, some need input features; try both
        try:
            feature_names = list(pre.get_feature_names_out())
        except TypeError:
            # older signatures
            feature_names = list(pre.get_feature_names_out(df_input.columns))
    except Exception:
        # manual fallback: build names from one-hot encoder categories + numeric
        try:
            ohe = pre.named_transformers_.get("cat")
            cat_names = []
            if ohe is not None and hasattr(ohe, "categories_") and categorical is not None:
                for col_idx, col_name in enumerate(categorical):
                    cats = ohe.categories_[col_idx]
                    cat_names.extend([f"{col_name}_{cat}" for cat in cats])
                feature_names = cat_names + (numeric or [])
        except Exception:
            feature_names = []

    # If still mismatch or unknown, fallback to generic names matching width
    if len(feature_names) != X_proc.shape[1]:
        feature_names = [f"Feature_{i}" for i in range(X_proc.shape[1])]

    feature_names = [str(f) for f in feature_names]
    return X_proc, feature_names


def create_shap_artifacts(clf, X_train, new_df, categorical=None, numeric=None, out_dir="explainability_outputs", bg_sample_size=200, top_n=5):
    """Create SHAP explanations and save artifacts (waterfall or fallback bar, summary plot, and textual explanation).

    Usage from notebook:
        from explainability_helpers import create_shap_artifacts
        create_shap_artifacts(clf, X_train, new_df, categorical, numeric)

    Args:
        clf: fitted sklearn Pipeline with 'preprocess' and 'model' steps
        X_train: original (untransformed) training DataFrame used to build background
        new_df: single-row DataFrame with new user's raw features
        categorical, numeric: lists of column names used by the pipeline (optional but recommended)
        out_dir: output folder for saved artifacts
        bg_sample_size: how many rows to use for SHAP background sample
        top_n: how many top features to include in textual explanation

    Returns:
        dict with keys: 'waterfall_path','summary_path','text_explanation','shap_values'
    """
    os.makedirs(out_dir, exist_ok=True)

    # get processed training data and feature names
    X_train_proc, feat_names = get_processed_data_and_names(clf, X_train, categorical=categorical, numeric=numeric)
    # background sample
    n_bg = min(bg_sample_size, X_train_proc.shape[0])
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(X_train_proc.shape[0], n_bg, replace=False)
    bg_sample = X_train_proc[bg_idx]

    # Try to initialize TreeExplainer (fast for tree models). Fall back to generic Explainer.
    model = clf.named_steps.get("model")
    try:
        explainer = shap.TreeExplainer(model, data=bg_sample, feature_perturbation="interventional")
    except Exception:
        explainer = shap.Explainer(model, bg_sample)

    # process the new instance
    new_proc, _ = get_processed_data_and_names(clf, new_df, categorical=categorical, numeric=numeric)

    # compute shap values robustly across SHAP versions
    try:
        shap_values_all = explainer.shap_values(new_proc)
    except Exception:
        # some explainers are callable
        res = explainer(new_proc)
        # res may be an Explanation with .values or a numpy array
        try:
            shap_values_all = res.values
        except Exception:
            shap_values_all = np.asarray(res)

    # predicted class (if classifier) - try to fetch from model
    pred_class = None
    try:
        if hasattr(clf.named_steps.get("model"), "classes_"):
            pred_class = clf.predict(new_df)[0]
            classes = list(clf.named_steps.get("model").classes_)
            class_idx = classes.index(pred_class)
        else:
            class_idx = 0
    except Exception:
        class_idx = 0

    # Extract instance-level shap values for the predicted class (or first available)
    if isinstance(shap_values_all, list) or (isinstance(shap_values_all, np.ndarray) and shap_values_all.ndim == 3):
        try:
            shap_vals_instance = np.array(shap_values_all[class_idx]).flatten()
        except Exception:
            # Try first element
            shap_vals_instance = np.array(shap_values_all[0]).flatten()
    else:
        shap_vals_instance = np.array(shap_values_all).flatten()

    # Attempt to get base value(s)
    base_value = None
    try:
        ev = explainer.expected_value
        if isinstance(ev, (list, tuple, np.ndarray)):
            try:
                base_value = ev[class_idx]
            except Exception:
                base_value = ev[0]
        else:
            base_value = float(ev)
    except Exception:
        base_value = None

    # Create waterfall plot (if shap supports it) using shap.Explanation wrapper when possible
    wf_path = os.path.join(out_dir, "shap_waterfall.png")
    plt.figure(figsize=(8, 6))
    try:
        data_for_explanation = new_proc[0]
        if hasattr(data_for_explanation, "ndim") and data_for_explanation.ndim > 1:
            data_for_explanation = data_for_explanation.flatten()

        # Build a shap.Explanation if shap has the class
        try:
            explanation_object = shap.Explanation(values=shap_vals_instance,
                                                 base_values=base_value,
                                                 data=data_for_explanation,
                                                 feature_names=feat_names)
            shap.plots.waterfall(explanation_object, show=False)
        except Exception:
            # Fallback: shap.plots.waterfall accepts (base_value, shap_values, features)
            try:
                shap.plots.waterfall(base_value, shap_vals_instance, data_for_explanation, show=False)
            except Exception:
                raise

        plt.title(f"SHAP Waterfall - Predicted: {pred_class}")
        plt.tight_layout()
        plt.savefig(wf_path, dpi=150)
        plt.close()
    except Exception as e:
        # fallback: bar chart of top features
        top_idx = np.argsort(np.abs(shap_vals_instance))[-min(10, len(shap_vals_instance)):][::-1]
        top_feat = np.array(feat_names)[top_idx]
        top_vals = shap_vals_instance[top_idx]
        plt.figure(figsize=(8, 6))
        plt.barh(range(len(top_idx))[::-1], top_vals)
        plt.yticks(range(len(top_idx)), top_feat)
        plt.xlabel("SHAP value (impact on model output)")
        plt.title(f"Top SHAP features - Predicted: {pred_class}")
        plt.tight_layout()
        wf_path = os.path.join(out_dir, "shap_top_bar.png")
        plt.savefig(wf_path, dpi=150)
        plt.close()

    # textual explanation (best-effort mapping back to original features)
    def textual_explanation_from_shap(shap_vals, feature_names, raw_input_row, top_n=5):
        idx = np.argsort(np.abs(shap_vals))[-top_n:][::-1]
        lines = []
        row0 = raw_input_row.iloc[0] if isinstance(raw_input_row, pd.DataFrame) else raw_input_row
        for i in idx:
            i = int(i)
            name = feature_names[i]
            val = float(shap_vals[i])
            raw_val = None
            # if one-hot style like col_value and we have original col name in categorical, try to map
            if "_" in name and categorical is not None:
                for col in categorical:
                    prefix = f"{col}_"
                    if name.startswith(prefix):
                        cat = name[len(prefix):]
                        raw_val = row0.get(col, None) if hasattr(row0, 'get') or col in row0.index else None
                        # indicate if the raw value matches the category
                        match = (str(raw_val) == str(cat)) if raw_val is not None else False
                        pretty_name = f"{col}: {cat}"
                        if match:
                            pretty_name += " (user value)"
                        break
                else:
                    pretty_name = name
            else:
                pretty_name = name
                raw_val = row0.get(name, None) if hasattr(row0, 'get') or name in row0.index else None

            sign = "increased" if val > 0 else "decreased"
            if raw_val is not None:
                lines.append(f"- {pretty_name}: {sign} probability of being '{pred_class}' (impact {val:.3f}); value = {raw_val}")
            else:
                lines.append(f"- {pretty_name}: {sign} probability of being '{pred_class}' (impact {val:.3f})")
        return "\n".join(lines)

    shap_text = textual_explanation_from_shap(shap_vals_instance, feat_names, new_df, top_n=top_n)

    # SHAP summary plot (global) using a small sample
    try:
        sample_idx = np.random.choice(X_train_proc.shape[0], min(200, X_train_proc.shape[0]), replace=False)
        sample = X_train_proc[sample_idx]
        sample_shap = explainer.shap_values(sample)
        summary_path = os.path.join(out_dir, "shap_summary.png")
        plt.figure(figsize=(10, 6))
        if isinstance(sample_shap, list):
            shap.summary_plot(sample_shap[class_idx] if len(sample_shap) > class_idx else sample_shap[0], sample, feature_names=feat_names, show=False)
        else:
            shap.summary_plot(sample_shap, sample, feature_names=feat_names, show=False)
        plt.savefig(summary_path, dpi=150, bbox_inches="tight")
        plt.close()
    except Exception:
        summary_path = None

    # save textual explanation
    text_path = os.path.join(out_dir, "user_explanation.txt")
    with open(text_path, "w", encoding="utf-8") as f:
        f.write(shap_text)

    return {
        "waterfall_path": wf_path,
        "summary_path": summary_path,
        "text_explanation": shap_text,
        "shap_values": shap_vals_instance
    }
