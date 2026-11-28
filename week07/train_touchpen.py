"""Progress-aligned stylus calibration. Run from any directory; see README."""
from pathlib import Path
import argparse
import copy
import csv
import hashlib
import json
import os
import platform
import sys

ROOT = Path(__file__).resolve().parent
LOCAL_DEPS = ROOT.parent / '.analysis-deps'
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))
os.environ.setdefault('MPLCONFIGDIR', str(ROOT.parent / '.mpl-cache'))
import numpy as np
import openpyxl
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge, SGDRegressor
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rmse(a, b):
    """Root mean squared Euclidean position error, not per-axis RMSE."""
    return float(np.sqrt(np.mean(np.sum((a - b) ** 2, axis=1))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, default=ROOT.parent.parent / 'AID_ML_Project' / '터치펜_데이터.xlsx')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    workbook = openpyxl.load_workbook(args.data, data_only=True, read_only=True)
    paths, audit = {}, []
    for name in ['펜 경로', '그림 경로']:
        sheet = workbook[name]
        values = list(sheet.values)
        assert values[1][:3] == ('t', 'x', 'y'), 'Unexpected schema'
        a = np.asarray([r[:3] for r in values[2:]], dtype=float)
        assert np.isfinite(a).all(), 'Missing/nonfinite measurement'
        assert np.all(np.diff(a[:, 0]) > 0), 'Time must increase strictly'
        paths[name] = a
        audit.append(dict(sheet=name, n=len(a), t_min=float(a[0, 0]), t_max=float(a[-1, 0]),
                          min_dt=float(np.diff(a[:, 0]).min()), max_dt=float(np.diff(a[:, 0]).max()),
                          duplicate_rows=int(len(a) - len(np.unique(a, axis=0)))))
    workbook.close()
    pen, drawing = paths['펜 경로'], paths['그림 경로']
    p = (pen[:, 0] - pen[0, 0]) / (pen[-1, 0] - pen[0, 0])
    q = (drawing[:, 0] - drawing[0, 0]) / (drawing[-1, 0] - drawing[0, 0])
    observed = np.column_stack([np.interp(p, q, drawing[:, j]) for j in (1, 2)])
    actual = pen[:, 1:]
    error = observed - actual
    n = len(p)
    a, b = int(n * .6), int(n * .8)
    # Two unused points at each boundary; preserve order, no random row split.
    train, val, test = np.arange(a), np.arange(a + 2, b), np.arange(b + 2, n)
    splits = {'train': train, 'validation': val, 'test': test}
    assert len(set(train) & set(test)) == 0
    # Check that interpolation source rows used by training do not touch validation/test.
    right = np.clip(np.searchsorted(q, p, side='right'), 1, len(q) - 1)
    source_sets = {k: set(right[idx]) | set(right[idx] - 1) for k, idx in splits.items()}
    assert source_sets['train'].isdisjoint(source_sets['validation'])
    assert source_sets['validation'].isdisjoint(source_sets['test'])
    aligned_rows = []
    for i in range(n):
        split = next((k for k, idx in splits.items() if i in idx), 'gap')
        aligned_rows.append(dict(pen_excel_row=i+3, progress=p[i], split=split,
                                 observed_x=observed[i, 0], observed_y=observed[i, 1],
                                 actual_x=actual[i, 0], actual_y=actual[i, 1],
                                 error_x=error[i, 0], error_y=error[i, 1]))
    write_csv(out / 'aligned_data.csv', aligned_rows)

    # Exploratory unsupervised analysis uses full recorded trajectories only.
    cluster_x = StandardScaler().fit_transform(drawing[:, 1:])
    clusters, cluster_models = [], {}
    for k in range(2, 7):
        model = KMeans(n_clusters=k, n_init=30, max_iter=300, random_state=42).fit(cluster_x)
        clusters.append(dict(k=k, inertia=float(model.inertia_),
                             silhouette=float(silhouette_score(cluster_x, model.labels_)),
                             iterations=int(model.n_iter_)))
        cluster_models[k] = model
    best_k = max(clusters, key=lambda row: row['silhouette'])['k']
    write_csv(out / 'clusters.csv', clusters)
    pca_rows = []
    for name, values in paths.items():
        xy = values[:, 1:]
        model = PCA(n_components=1).fit(xy)
        reconstruction = model.inverse_transform(model.transform(xy))
        pca_rows.append(dict(sheet=name, pc1_variance=float(model.explained_variance_ratio_[0]),
                             reconstruction_rmse=rmse(xy, reconstruction)))
    write_csv(out / 'pca.csv', pca_rows)

    # Targets are observed-minus-actual errors. All transforms fit TRAIN ONLY.
    target_scaler = StandardScaler().fit(error[train])
    y_train = target_scaler.transform(error[train])
    history, candidates, snapshots = [], [], {}
    for degree in [1, 2]:
        poly = PolynomialFeatures(degree=degree, include_bias=False)
        expanded = poly.fit_transform(observed)
        scaler = StandardScaler().fit(expanded[train])
        X = scaler.transform(expanded)
        for eta in [.001, .01, .03]:
            for alpha in [.001, .1]:
                cid = f'd{degree}_eta{eta}_alpha{alpha}'
                models = [SGDRegressor(loss='squared_error', penalty='l2', alpha=alpha,
                                      learning_rate='constant', eta0=eta, shuffle=False,
                                      random_state=42) for _ in range(2)]
                rng = np.random.default_rng(42)
                best = None
                for epoch in range(1, 201):
                    order = rng.permutation(len(train))
                    for j, model in enumerate(models):
                        model.partial_fit(X[train][order], y_train[order, j])
                    pred_error = target_scaler.inverse_transform(np.column_stack([m.predict(X) for m in models]))
                    corrected = observed - pred_error
                    tr, va = rmse(actual[train], corrected[train]), rmse(actual[val], corrected[val])
                    assert np.isfinite([tr, va]).all(), 'Training diverged'
                    history.append(dict(candidate=cid, epoch=epoch, train_rmse=tr, validation_rmse=va))
                    # Require >= 30 epochs for the chosen checkpoint.
                    if epoch >= 30 and (best is None or va < best['validation_rmse']):
                        best = dict(candidate=cid, degree=degree, eta=eta, alpha=alpha,
                                    epoch=epoch, train_rmse=tr, validation_rmse=va)
                        snapshots[cid] = (copy.deepcopy(models), X.copy())
                    if epoch in [1, 30, 60, 100, 200]:
                        snapshots[(cid, epoch)] = corrected.copy()
                candidates.append(best)
    chosen = min(candidates, key=lambda row: row['validation_rmse'])
    models, X = snapshots[chosen['candidate']]
    selected_error = target_scaler.inverse_transform(np.column_stack([m.predict(X) for m in models]))
    predictions = {
        'Uncorrected': observed,
        'Mean offset (train only)': observed - error[train].mean(axis=0),
    }
    affine_scaler = StandardScaler().fit(observed[train])
    affine_x = affine_scaler.transform(observed)
    ridge = Ridge(alpha=1.0).fit(affine_x[train], error[train])
    predictions['Ridge affine (alpha=1)'] = observed - ridge.predict(affine_x)
    predictions['SGD selected'] = observed - selected_error
    # Freeze selection using validation ONLY, then evaluate the held-out test.
    selected_model = min(predictions, key=lambda name: rmse(actual[val], predictions[name][val]))
    scores = []
    for name, pred in predictions.items():
        for split, idx in splits.items():
            dist = np.linalg.norm(pred[idx] - actual[idx], axis=1)
            scores.append(dict(model=name, split=split, n=len(idx), rmse=rmse(actual[idx], pred[idx]),
                               mean_distance=float(dist.mean()), max_distance=float(dist.max())))
    epoch_scores = []
    for epoch in [1, 30, 60, 100, 200]:
        pred = snapshots[(chosen['candidate'], epoch)]
        epoch_scores.append(dict(epoch=epoch, **{f'{s}_rmse': rmse(actual[i], pred[i]) for s, i in splits.items()}))
    # All selected candidate epoch comparisons are retrospective; no test-based tuning.
    write_csv(out / 'epoch_history.csv', history)
    write_csv(out / 'candidates.csv', candidates)
    write_csv(out / 'metrics.csv', scores)
    write_csv(out / 'epoch_comparison.csv', epoch_scores)
    rows = []
    for i in test:
        rows.append(dict(pen_excel_row=int(i+3), progress=p[i], actual_x=actual[i, 0], actual_y=actual[i, 1],
                         raw_x=observed[i, 0], raw_y=observed[i, 1],
                         sgd_x=predictions['SGD selected'][i, 0], sgd_y=predictions['SGD selected'][i, 1]))
    write_csv(out / 'test_predictions.csv', rows)
    result = dict(data_sha256=hashlib.sha256(args.data.read_bytes()).hexdigest(), audit=audit,
                  split_counts={s:len(i) for s,i in splits.items()}, selected_sgd=chosen,
                  selected_model=selected_model, best_k=best_k, pca=pca_rows,
                  versions=dict(python=platform.python_version(), numpy=np.__version__,
                                sklearn=sklearn.__version__, matplotlib=matplotlib.__version__, openpyxl=openpyxl.__version__))
    (out / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    plt.rcParams.update({'font.family':'Malgun Gothic', 'axes.unicode_minus':False, 'font.size':10})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for ax, (name, values) in zip(axes, paths.items()):
        ax.plot(values[:, 1], values[:, 2], '.-', markersize=3)
        ax.set(title=name, xlabel='x (원본 좌표 단위)', ylabel='y (원본 좌표 단위)')
        ax.set_aspect('equal', adjustable='datalim')
        ax.grid(alpha=.2)
    fig.savefig(out / 'raw_paths.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    axes[0].scatter(drawing[:,1], drawing[:,2], c=cluster_models[best_k].labels_, cmap='tab10', s=17)
    axes[0].set(title=f'그림 경로의 위치 군집 (k={best_k})', xlabel='x', ylabel='y')
    axes[1].plot([r['k'] for r in clusters], [r['silhouette'] for r in clusters], 'o-')
    axes[1].set(title='실루엣 점수 비교 (탐색용)', xlabel='k', ylabel='실루엣 점수', xticks=range(2,7))
    fig.savefig(out / 'clusters.png', dpi=160)
    plt.close(fig)
    curve = [r for r in history if r['candidate'] == chosen['candidate']]
    fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for field, label in [('train_rmse','훈련'), ('validation_rmse','검증')]:
        ax.plot([r['epoch'] for r in curve], [r[field] for r in curve], label=label)
    ax.axvline(chosen['epoch'], color='gray', linestyle='--', label=f"선택: {chosen['epoch']}에포크")
    ax.set(xlabel='에포크', ylabel='위치 RMSE (원본 좌표 단위)', title='SGD 보정 학습 곡선')
    ax.legend()
    ax.grid(alpha=.2)
    fig.savefig(out / 'learning_curve.png', dpi=160)
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), constrained_layout=True)
    for j, label in enumerate(['x','y']):
        axes[j].plot(p, actual[:,j], 'k-', label='실제 펜 위치')
        axes[j].plot(p, observed[:,j], '--', label='보정 전')
        axes[j].plot(p, predictions['SGD selected'][:,j], label='SGD 보정')
        axes[j].axvspan(p[test[0]], 1, color='gray', alpha=.15, label='테스트 구간')
        axes[j].set(xlabel='정규화 시간 진행률', ylabel=f'{label} (원본 좌표 단위)')
        axes[j].legend(fontsize=8)
    fig.savefig(out / 'correction.png', dpi=160)
    plt.close(fig)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print(json.dumps(scores, ensure_ascii=False, indent=2))
    print(json.dumps(epoch_scores, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
