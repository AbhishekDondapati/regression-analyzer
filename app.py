import os
import uuid
import json
import base64
import io

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, redirect, url_for, session, flash
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = os.urandom(24)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
ALLOWED_EXTENSIONS = {'csv'}
MAX_CONTENT_LENGTH = 10 * 1024 * 1024  # 10 MB

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_LENGTH


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=110, bbox_inches='tight')
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return encoded


def run_regression(df, x_cols, y_col, degree=1, test_size=0.2):
    X = df[x_cols].values
    y = df[y_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    if degree > 1:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        X_train_t = poly.fit_transform(X_train)
        X_test_t = poly.transform(X_test)
        X_all_t = poly.transform(X)
        feature_names = poly.get_feature_names_out(x_cols)
    else:
        X_train_t = X_train
        X_test_t = X_test
        X_all_t = X
        feature_names = x_cols

    model = LinearRegression()
    model.fit(X_train_t, y_train)

    y_pred_train = model.predict(X_train_t)
    y_pred_test = model.predict(X_test_t)
    y_pred_all = model.predict(X_all_t)

    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

    residuals = y - y_pred_all

    coefficients = [
        {'feature': name, 'coefficient': round(float(coef), 6)}
        for name, coef in zip(feature_names, model.coef_)
    ]
    intercept = round(float(model.intercept_), 6)

    plots = {}

    # --- Plot 1: Scatter + Regression Line (only for single numeric x) ---
    if len(x_cols) == 1:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x_sorted = np.linspace(X[:, 0].min(), X[:, 0].max(), 300).reshape(-1, 1)
        if degree > 1:
            x_sorted_t = poly.transform(x_sorted)
        else:
            x_sorted_t = x_sorted
        y_line = model.predict(x_sorted_t)

        ax.scatter(X[:, 0], y, alpha=0.55, color='steelblue', edgecolors='white',
                   linewidths=0.5, s=55, label='Data')
        ax.plot(x_sorted[:, 0], y_line, color='crimson', linewidth=2,
                label=f'Degree-{degree} fit')
        ax.set_xlabel(x_cols[0], fontsize=12)
        ax.set_ylabel(y_col, fontsize=12)
        ax.set_title(f'{x_cols[0]}  vs  {y_col}', fontsize=13)
        ax.legend()
        sns.despine(fig=fig)
        plots['scatter'] = fig_to_base64(fig)

    # --- Plot 2: Actual vs Predicted ---
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(y, y_pred_all, alpha=0.5, color='darkorange', edgecolors='white',
               linewidths=0.4, s=50)
    lims = [min(y.min(), y_pred_all.min()), max(y.max(), y_pred_all.max())]
    ax.plot(lims, lims, 'k--', linewidth=1.4, label='Perfect fit')
    ax.set_xlabel('Actual', fontsize=12)
    ax.set_ylabel('Predicted', fontsize=12)
    ax.set_title('Actual vs Predicted', fontsize=13)
    ax.legend()
    sns.despine(fig=fig)
    plots['actual_vs_pred'] = fig_to_base64(fig)

    # --- Plot 3: Residuals ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    axes[0].scatter(y_pred_all, residuals, alpha=0.5, color='mediumseagreen',
                    edgecolors='white', linewidths=0.4, s=50)
    axes[0].axhline(0, color='red', linewidth=1.5, linestyle='--')
    axes[0].set_xlabel('Predicted', fontsize=11)
    axes[0].set_ylabel('Residual', fontsize=11)
    axes[0].set_title('Residuals vs Predicted', fontsize=12)

    axes[1].hist(residuals, bins=25, color='mediumpurple', edgecolor='white', linewidth=0.6)
    axes[1].set_xlabel('Residual', fontsize=11)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].set_title('Residual Distribution', fontsize=12)
    fig.tight_layout()
    sns.despine(fig=fig)
    plots['residuals'] = fig_to_base64(fig)

    # --- Plot 4: Correlation heatmap (if multiple columns) ---
    if len(x_cols) > 1:
        corr_cols = x_cols + [y_col]
        fig, ax = plt.subplots(figsize=(max(5, len(corr_cols)), max(4, len(corr_cols) - 1)))
        sns.heatmap(df[corr_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                    center=0, ax=ax, linewidths=0.5)
        ax.set_title('Correlation Matrix', fontsize=13)
        fig.tight_layout()
        plots['correlation'] = fig_to_base64(fig)

    return {
        'r2_train': round(r2_train, 4),
        'r2_test': round(r2_test, 4),
        'rmse_train': round(rmse_train, 4),
        'rmse_test': round(rmse_test, 4),
        'intercept': intercept,
        'coefficients': coefficients,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'plots': plots,
    }


# ── Routes ──────────────────────────────────────────────────────────────────

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'csv_file' not in request.files:
            flash('No file selected.', 'danger')
            return redirect(request.url)

        file = request.files['csv_file']
        if file.filename == '':
            flash('No file selected.', 'danger')
            return redirect(request.url)

        if not allowed_file(file.filename):
            flash('Only CSV files are accepted.', 'danger')
            return redirect(request.url)

        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        file.save(filepath)

        try:
            df = pd.read_csv(filepath)
            if df.empty:
                raise ValueError('The CSV file is empty.')
        except Exception as e:
            flash(f'Could not read CSV: {e}', 'danger')
            os.remove(filepath)
            return redirect(request.url)

        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) < 2:
            flash('CSV must contain at least 2 numeric columns.', 'danger')
            os.remove(filepath)
            return redirect(request.url)

        session['filepath'] = filepath
        session['columns'] = df.columns.tolist()
        session['numeric_cols'] = numeric_cols
        session['shape'] = list(df.shape)
        session['preview'] = df.head(5).to_dict(orient='records')

        return redirect(url_for('configure'))

    return render_template('index.html')


@app.route('/configure', methods=['GET', 'POST'])
def configure():
    if 'filepath' not in session:
        return redirect(url_for('index'))

    numeric_cols = session['numeric_cols']
    preview = session['preview']
    shape = session['shape']

    if request.method == 'POST':
        x_cols = request.form.getlist('x_cols')
        y_col = request.form.get('y_col')
        degree = int(request.form.get('degree', 1))
        test_size = float(request.form.get('test_size', 0.2))

        # Validation
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
            flash('Polynomial regression is only supported for a single predictor.', 'warning')
            degree = 1

        df = pd.read_csv(session['filepath'])
        df = df[x_cols + [y_col]].dropna()

        if len(df) < 10:
            flash('Not enough rows after dropping missing values (need ≥ 10).', 'danger')
            return redirect(request.url)

        try:
            result = run_regression(df, x_cols, y_col, degree=degree, test_size=test_size)
        except Exception as e:
            flash(f'Regression failed: {e}', 'danger')
            return redirect(request.url)

        result['x_cols'] = x_cols
        result['y_col'] = y_col
        result['degree'] = degree
        result['n_rows'] = len(df)
        return render_template('results.html', result=result)

    return render_template('configure.html',
                           numeric_cols=numeric_cols,
                           preview=preview,
                           shape=shape)


@app.route('/reset')
def reset():
    filepath = session.pop('filepath', None)
    if filepath and os.path.exists(filepath):
        os.remove(filepath)
    session.clear()
    return redirect(url_for('index'))


if __name__ == '__main__':
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.run(debug=True)
