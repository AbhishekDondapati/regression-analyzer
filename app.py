import os
import uuid
import io
import base64

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from flask import (Flask, render_template, request, redirect,
                   url_for, session, flash, send_file)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures, LabelEncoder
from sklearn.metrics import (r2_score, mean_squared_error, accuracy_score,
                              classification_report, confusion_matrix)
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def encode_categoricals(df, encoding_map):
    df = df.copy()
    for col, method in encoding_map.items():
        if col not in df.columns:
            continue
        if method == 'label':
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
        elif method == 'onehot':
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=True).astype(int)
            df = pd.concat([df.drop(columns=[col]), dummies], axis=1)
    return df


def get_descriptive_stats(df, cols):
    stats = df[cols].describe().round(4)
    return {
        'columns': list(stats.columns),
        'index': list(stats.index),
        'data': [[round(v, 4) if pd.notna(v) else '—' for v in row]
                 for row in stats.values.tolist()]
    }


def run_regression(df, x_cols, y_col, regression_type='linear',
                   degree=1, alpha=1.0, test_size=0.2):
    X = df[x_cols].values.astype(float)
    y = df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)

    is_logistic = regression_type == 'logistic'
    feature_names = list(x_cols)
    poly = None

    if degree > 1 and not is_logistic:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_t = poly.fit_transform(X_train)
        X_test_t  = poly.transform(X_test)
        X_all_t   = poly.transform(X)
        feature_names = list(poly.get_feature_names_out(x_cols))
    else:
        X_train_t = X_train
        X_test_t  = X_test
        X_all_t   = X

    models = {
        'linear':   LinearRegression(),
        'polynomial': LinearRegression(),
        'ridge':    Ridge(alpha=alpha),
        'lasso':    Lasso(alpha=alpha, max_iter=10000),
        'logistic': LogisticRegression(max_iter=1000, random_state=42),
    }
    model = models.get(regression_type, LinearRegression())
    model.fit(X_train_t, y_train)

    y_pred_train = model.predict(X_train_t)
    y_pred_test  = model.predict(X_test_t)
    y_pred_all   = model.predict(X_all_t)

    coef_arr = model.coef_[0] if is_logistic else model.coef_
    intercept_val = float(model.intercept_[0] if is_logistic else model.intercept_)
    coefficients = [
        {'feature': name, 'coefficient': round(float(c), 6)}
        for name, c in zip(feature_names, coef_arr)
    ]

    result = {
        'x_cols': list(x_cols),
        'y_col': y_col,
        'regression_type': regression_type,
        'degree': degree,
        'alpha': alpha,
        'n_rows': len(df),
        'n_train': len(X_train),
        'n_test':  len(X_test),
        'feature_names': feature_names,
        'intercept': round(intercept_val, 6),
        'coefficients': coefficients,
        'is_logistic': is_logistic,
        'plots': {},
    }

    # ── Metrics ───────────────────────────────────────────────────────────
    if is_logistic:
        result['accuracy_train'] = round(accuracy_score(y_train, y_pred_train), 4)
        result['accuracy_test']  = round(accuracy_score(y_test,  y_pred_test),  4)
        report = classification_report(y_test, y_pred_test, output_dict=True, zero_division=0)
        result['class_report'] = {
            k: {m: round(v, 3) for m, v in v_dict.items()}
            if isinstance(v_dict, dict) else round(v_dict, 3)
            for k, v_dict in report.items()
        }
    else:
        residuals = y - y_pred_all
        result['r2_train']   = round(r2_score(y_train, y_pred_train), 4)
        result['r2_test']    = round(r2_score(y_test,  y_pred_test),  4)
        result['rmse_train'] = round(float(np.sqrt(mean_squared_error(y_train, y_pred_train))), 4)
        result['rmse_test']  = round(float(np.sqrt(mean_squared_error(y_test,  y_pred_test))),  4)

    # ── Plots ─────────────────────────────────────────────────────────────

    # 1. Scatter + fit line (single predictor, regression only)
    if not is_logistic and len(x_cols) == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x_sorted = np.linspace(X[:, 0].min(), X[:, 0].max(), 300).reshape(-1, 1)
        x_sorted_t = poly.transform(x_sorted) if poly else x_sorted
        ax.scatter(X[:, 0], y, alpha=0.55, color='steelblue',
                   edgecolors='white', linewidths=0.5, s=55, label='Data')
        ax.plot(x_sorted[:, 0], model.predict(x_sorted_t),
                color='crimson', linewidth=2, label=f'Degree-{degree} fit')
        ax.set_xlabel(x_cols[0], fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{x_cols[0]}  vs  {y_col}', fontsize=13)
        ax.legend()
        sns.despine(fig=fig)
        result['plots']['scatter'] = fig_to_base64(fig)

    # 2. Actual vs Predicted (regression) / Confusion Matrix (logistic)
    if is_logistic:
        cm = confusion_matrix(y_test, y_pred_test)
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    linewidths=0.5, cbar=False)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_title('Confusion Matrix (Test Set)', fontsize=13)
        result['plots']['confusion_matrix'] = fig_to_base64(fig)
    else:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.scatter(y, y_pred_all, alpha=0.5, color='darkorange',
                   edgecolors='white', linewidths=0.4, s=50)
        lims = [min(y.min(), y_pred_all.min()), max(y.max(), y_pred_all.max())]
        ax.plot(lims, lims, 'k--', linewidth=1.4, label='Perfect fit')
        ax.set_xlabel('Actual', fontsize=12)
        ax.set_ylabel('Predicted', fontsize=12)
        ax.set_title('Actual vs Predicted', fontsize=13)
        ax.legend()
        sns.despine(fig=fig)
        result['plots']['actual_vs_pred'] = fig_to_base64(fig)

        # 3. Residuals
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].scatter(y_pred_all, residuals, alpha=0.5, color='mediumseagreen',
                        edgecolors='white', linewidths=0.4, s=50)
        axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
        axes[0].set_xlabel('Predicted', fontsize=11)
        axes[0].set_ylabel('Residual', fontsize=11)
        axes[0].set_title('Residuals vs Predicted', fontsize=12)
        axes[1].hist(residuals, bins=25, color='mediumpurple',
                     edgecolor='white', linewidth=0.6)
        axes[1].set_xlabel('Residual', fontsize=11)
        axes[1].set_ylabel('Count', fontsize=11)
        axes[1].set_title('Residual Distribution', fontsize=12)
        fig.tight_layout()
        sns.despine(fig=fig)
        result['plots']['residuals'] = fig_to_base64(fig)

    # 4. Feature Importance
    coefs_abs = np.abs([c['coefficient'] for c in coefficients])
    names     = [c['feature']           for c in coefficients]
    top_n     = min(15, len(names))
    sorted_idx = np.argsort(coefs_abs)[-top_n:]
    fig, ax = plt.subplots(figsize=(7, max(3, top_n * 0.45 + 1)))
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, top_n))
    ax.barh([names[i] for i in sorted_idx], coefs_abs[sorted_idx],
            color=colors, edgecolor='white', linewidth=0.5)
    ax.set_xlabel('|Coefficient|', fontsize=11)
    ax.set_title('Feature Importance', fontsize=13)
    sns.despine(fig=fig)
    fig.tight_layout()
    result['plots']['feature_importance'] = fig_to_base64(fig)

    # 5. Correlation heatmap
    if len(x_cols) > 1:
        corr_cols = list(x_cols) + [y_col]
        n = len(corr_cols)
        fig, ax = plt.subplots(figsize=(max(5, n), max(4, n - 1)))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f',
                    cmap='coolwarm', center=0, ax=ax, linewidths=0.5)
        ax.set_title('Correlation Matrix', fontsize=13)
        fig.tight_layout()
        result['plots']['correlation'] = fig_to_base64(fig)

    # 6. Pair Plot (max 6 cols)
    pair_cols = (list(x_cols) + [y_col])[:6]
    if len(pair_cols) >= 2:
        try:
            pair_grid = sns.pairplot(
                df[pair_cols], plot_kws={'alpha': 0.4, 's': 25}, diag_kind='kde')
            pair_grid.figure.suptitle('Pair Plot', y=1.02, fontsize=13)
            buf = io.BytesIO()
            pair_grid.figure.savefig(buf, format='png', dpi=90, bbox_inches='tight')
            buf.seek(0)
            result['plots']['pair_plot'] = base64.b64encode(buf.read()).decode('utf-8')
            plt.close('all')
        except Exception:
            plt.close('all')

    # Save predictions CSV for export
    pred_df = pd.DataFrame({'actual': y, 'predicted': y_pred_all.round(6)})
    pred_filename = f"{uuid.uuid4().hex}_predictions.csv"
    pred_df.to_csv(os.path.join(UPLOAD_FOLDER, pred_filename), index=False)
    result['predictions_file'] = pred_filename

    return result


def generate_pdf(result):
    buf = io.BytesIO()
    with PdfPages(buf) as pdf:

        # ── Page 1: Summary ───────────────────────────────────────────────
        fig = plt.figure(figsize=(8.5, 11))

        # Header bar
        ax_h = fig.add_axes([0, 0.88, 1, 0.12])
        ax_h.set_facecolor('#1a73e8')
        ax_h.text(0.5, 0.6, 'Regression Analysis Report',
                  ha='center', va='center', fontsize=20,
                  fontweight='bold', color='white', transform=ax_h.transAxes)
        ax_h.text(0.5, 0.2, f"Target: {result['y_col']}  |  Type: {result['regression_type'].capitalize()}",
                  ha='center', va='center', fontsize=11, color='#cce0ff',
                  transform=ax_h.transAxes)
        ax_h.axis('off')

        # Metrics table
        ax_m = fig.add_axes([0.05, 0.55, 0.9, 0.31])
        ax_m.axis('off')
        rows = [
            ['Predictors (X)', ', '.join(result['x_cols'])],
            ['Polynomial Degree', str(result['degree'])],
            ['Training Samples', str(result['n_train'])],
            ['Test Samples', str(result['n_test'])],
        ]
        if result['is_logistic']:
            rows += [['Accuracy (Train)', str(result.get('accuracy_train'))],
                     ['Accuracy (Test)',  str(result.get('accuracy_test'))]]
        else:
            rows += [['R² (Train)', str(result.get('r2_train'))],
                     ['R² (Test)',  str(result.get('r2_test'))],
                     ['RMSE (Train)', str(result.get('rmse_train'))],
                     ['RMSE (Test)',  str(result.get('rmse_test'))]]

        tbl = ax_m.table(cellText=rows, colLabels=['Metric', 'Value'],
                         cellLoc='left', loc='center', colWidths=[0.4, 0.6])
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(11)
        tbl.scale(1, 2.0)
        for (r, c), cell in tbl.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor('#1a73e8')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f0f4f8')

        # Coefficients table
        ax_c = fig.add_axes([0.05, 0.05, 0.9, 0.46])
        ax_c.axis('off')
        coef_rows = [['intercept', str(result['intercept'])]] + [
            [c['feature'], str(c['coefficient'])]
            for c in result['coefficients'][:18]
        ]
        tbl2 = ax_c.table(cellText=coef_rows,
                          colLabels=['Feature', 'Coefficient'],
                          cellLoc='left', loc='upper center',
                          colWidths=[0.65, 0.35])
        tbl2.auto_set_font_size(False)
        tbl2.set_fontsize(9)
        tbl2.scale(1, 1.6)
        for (r, c), cell in tbl2.get_celld().items():
            cell.set_edgecolor('#dddddd')
            if r == 0:
                cell.set_facecolor('#1a73e8')
                cell.set_text_props(color='white', fontweight='bold')
            elif r % 2 == 0:
                cell.set_facecolor('#f0f4f8')
        ax_c.set_title('Model Coefficients', fontsize=12,
                        fontweight='bold', pad=8, loc='left')

        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # ── Pages for each plot ───────────────────────────────────────────
        plot_titles = {
            'scatter':           'Scatter Plot & Regression Fit',
            'actual_vs_pred':    'Actual vs Predicted',
            'residuals':         'Residual Diagnostics',
            'feature_importance':'Feature Importance',
            'correlation':       'Correlation Matrix',
            'pair_plot':         'Pair Plot',
            'confusion_matrix':  'Confusion Matrix',
        }
        for key, title in plot_titles.items():
            if key not in result['plots']:
                continue
            img_bytes = base64.b64decode(result['plots'][key])
            img = plt.imread(io.BytesIO(img_bytes))
            fig, ax = plt.subplots(figsize=(10, 7))
            ax.imshow(img)
            ax.axis('off')
            ax.set_title(title, fontsize=14, fontweight='bold', pad=12)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close(fig)

    buf.seek(0)
    return buf


# ── Routes ───────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)
        file = request.files['csv_file']
        if not file.filename or not allowed_file(file.filename):
            flash('Only CSV files are accepted.', 'danger')
            return redirect(request.url)

        filename  = secure_filename(file.filename)
        filepath  = os.path.join(app.config['UPLOAD_FOLDER'],
                                 f"{uuid.uuid4().hex}_{filename}")
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                raise ValueError('The CSV file is empty.')
        except Exception as e:
            flash(f'Could not read CSV: {e}', 'danger')
            os.remove(filepath)
            return redirect(request.url)

        numeric_cols     = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(exclude=[np.number]).columns.tolist()

        if len(numeric_cols) < 1:
            flash('CSV must contain at least 1 numeric column.', 'danger')
            os.remove(filepath)
            return redirect(request.url)

        session['filepath']         = filepath
        session['numeric_cols']     = numeric_cols
        session['categorical_cols'] = categorical_cols
        session['all_cols']         = df.columns.tolist()
        session['shape']            = list(df.shape)
        session['preview']          = df.head(5).to_dict(orient='records')
        return redirect(url_for('configure'))

    return render_template('index.html')


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    if 'filepath' not in session:
        return redirect(url_for('index'))

    numeric_cols     = session['numeric_cols']
    categorical_cols = session['categorical_cols']
    shape            = session['shape']
    preview          = session['preview']

    df = pd.read_csv(session['filepath'])
    stats = get_descriptive_stats(df, numeric_cols) if numeric_cols else None

    if request.method == 'POST':
        x_cols          = request.form.getlist('x_cols')
        y_col           = request.form.get('y_col')
        regression_type = request.form.get('regression_type', 'linear')
        degree          = int(request.form.get('degree', 1))
        alpha           = float(request.form.get('alpha', 1.0))
        test_size       = float(request.form.get('test_size', 0.2))

        # Encoding choices for categorical cols
        encoding_map = {}
        for col in categorical_cols:
            method = request.form.get(f'enc_{col}', 'skip')
            if method != 'skip':
                encoding_map[col] = method

        if not x_cols:
            flash('Select at least one predictor (X).', 'danger')
            return redirect(request.url)
        if not y_col:
            flash('Select a target variable (Y).', 'danger')
            return redirect(request.url)
        if y_col in x_cols:
            flash('Target variable cannot also be a predictor.', 'danger')
            return redirect(request.url)
        if degree > 1 and len(x_cols) > 1:
            flash('Polynomial degree > 1 is only for a single predictor. Degree reset to 1.', 'warning')
            degree = 1

        # Apply encoding then build working df
        df_enc   = encode_categoricals(df, encoding_map)
        all_cols = df_enc.columns.tolist()

        # Expand one-hot encoded x cols
        x_cols_expanded = []
        for col in x_cols:
            if col in all_cols:
                x_cols_expanded.append(col)
            else:
                # onehot expanded
                x_cols_expanded += [c for c in all_cols if c.startswith(col + '_')]

        if y_col not in all_cols:
            flash(f'Target column "{y_col}" not found after encoding.', 'danger')
            return redirect(request.url)

        work_df = df_enc[x_cols_expanded + [y_col]].dropna()
        if len(work_df) < 10:
            flash('Not enough rows after dropping missing values (need ≥ 10).', 'danger')
            return redirect(request.url)

        # Logistic: check target
        if regression_type == 'logistic':
            n_unique = work_df[y_col].nunique()
            if n_unique > 30:
                flash(f'Target has {n_unique} unique values — logistic regression needs a categorical target. Consider a regression type instead.', 'warning')

        try:
            result = run_regression(work_df, x_cols_expanded, y_col,
                                    regression_type=regression_type,
                                    degree=degree, alpha=alpha,
                                    test_size=test_size)
        except Exception as e:
            flash(f'Regression failed: {e}', 'danger')
            return redirect(request.url)

        # Store params for export
        session['last_params'] = {
            'x_cols': x_cols_expanded, 'y_col': y_col,
            'regression_type': regression_type, 'degree': degree,
            'alpha': alpha, 'test_size': test_size,
            'encoding_map': encoding_map,
            'predictions_file': result['predictions_file'],
        }
        # Store result without plots for PDF re-generation
        result_no_plots = {k: v for k, v in result.items() if k != 'plots'}
        session['last_result'] = result_no_plots

        return render_template('results.html', result=result)

    return render_template('configure.html',
                           numeric_cols=numeric_cols,
                           categorical_cols=categorical_cols,
                           preview=preview,
                           shape=shape,
                           stats=stats)


@app.route('/export_csv')
def export_csv():
    params = session.get('last_params')
    if not params or not params.get('predictions_file'):
        flash('No analysis results to export.', 'danger')
        return redirect(url_for('configure'))

    pred_path = os.path.join(UPLOAD_FOLDER, params['predictions_file'])
    if not os.path.exists(pred_path):
        flash('Predictions file not found. Re-run the analysis.', 'danger')
        return redirect(url_for('configure'))

    return send_file(pred_path, mimetype='text/csv',
                     as_attachment=True, download_name='predictions.csv')


@app.route('/export_pdf')
def export_pdf():
    params = session.get('last_params')
    filepath = session.get('filepath')
    if not params or not filepath:
        flash('No analysis results to export.', 'danger')
        return redirect(url_for('configure'))

    try:
        df = pd.read_csv(filepath)
        df_enc = encode_categoricals(df, params.get('encoding_map', {}))
        work_df = df_enc[params['x_cols'] + [params['y_col']]].dropna()
        result = run_regression(
            work_df,
            params['x_cols'], params['y_col'],
            regression_type=params['regression_type'],
            degree=params['degree'],
            alpha=params['alpha'],
            test_size=params['test_size'],
        )
        pdf_buf = generate_pdf(result)
        return send_file(pdf_buf, mimetype='application/pdf',
                         as_attachment=True, download_name='regression_report.pdf')
    except Exception as e:
        flash(f'PDF generation failed: {e}', 'danger')
        return redirect(url_for('configure'))


@app.route('/reset')
def reset():
    filepath = session.pop('filepath', None)
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    # Clean up prediction files
    for key in ['last_params']:
        params = session.get(key, {})
        pf = params.get('predictions_file') if isinstance(params, dict) else None
        if pf:
            p = os.path.join(UPLOAD_FOLDER, pf)
            if os.path.exists(p):
                os.remove(p)
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
