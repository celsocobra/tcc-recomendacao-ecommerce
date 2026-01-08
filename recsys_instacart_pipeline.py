#!/usr/bin/env python3
"""
Instacart Recommender – pipeline fim-a-fim (TCC)

Executa: carga/limpeza, modelos (Popularidade, CF item-based, Conteúdo, ALS opcional),
híbrido ponderado e avaliação Top-K, gerando métricas e figuras para o TCC.

Uso (Windows/macOS/Linux):
    python recsys_instacart_pipeline.py --data_dir data/instacart --out_dir outputs --max_users 20000 --K 5,10,20
    # com ALS (se a lib 'implicit' estiver instalada):
    python recsys_instacart_pipeline.py --data_dir data/instacart --out_dir outputs --max_users 20000 --als --K 5,10,20

Arquivos esperados em data/instacart/ (Instacart 2017):
    orders.csv
    order_products__prior.csv
    order_products__train.csv
    products.csv
    aisles.csv
    departments.csv

Saídas (em --out_dir):
    metrics_summary.csv
    precision_recall_curves.png
    ndcg10_vs_predtime.png
"""

from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
from tqdm import tqdm

def business_metrics(db: DataBundle, recs: np.ndarray, K: int = 10) -> dict:
    """
    Retorna:
      - coverage_%:  itens distintos nas Top-K vs. catálogo (%)
      - intra_list_diversity: média de (1 - cos) entre pares da lista (0..1)
      - avg_popularity_%: share médio de popularidade dos itens (%)
    """
    I = db.X_train.shape[1]
    topk = recs[:, :K]

    # 1) Cobertura do catálogo
    coverage = float(np.unique(topk).size) / float(I) * 100.0

    # 2) Diversidade intra-lista (1 - cos) média por usuário
    F = db.item_features.tocsr()      # já normalizado em L2 no seu pipeline
    diversities = []
    for items in topk:
        items = np.unique(items)
        if items.size < 2:
            continue
        M = F[items].dot(F[items].T).toarray()  # similaridade cosseno
        tri = np.triu_indices_from(M, k=1)
        sims = M[tri]
        diversities.append(1.0 - float(np.mean(sims)))
    ild = float(np.mean(diversities)) if diversities else 0.0

    # 3) Popularidade média (share %)
    pop = db.item_popularity
    pop_share = pop / pop.sum()
    avg_pop = float(np.mean(pop_share[topk])) * 100.0

    return {"coverage_%": coverage,
            "intra_list_diversity": ild,
            "avg_popularity_%": avg_pop}


# ALS opcional
try:
    import implicit
    HAS_IMPLICIT = True
except Exception:
    HAS_IMPLICIT = False

np.random.seed(42)


# ===================== DATA ===================== #
@dataclass
class DataBundle:
    user_map: Dict[int, int]          # raw_user_id -> uid (0..U-1)
    item_map: Dict[int, int]          # raw_product_id -> iid (0..I-1)
    user_rev: List[int]
    item_rev: List[int]
    X_train: sp.csr_matrix            # U x I (implícita)
    test_truth: Dict[int, np.ndarray] # uid -> np.array[iids] do último pedido
    item_features: sp.csr_matrix      # I x F (conteúdo)
    item_popularity: np.ndarray       # I


def load_instacart(
    data_dir: str,
    min_orders_per_user: int = 3,
    min_product_freq: int = 20,
    max_users: int | None = None,
) -> DataBundle:
    print("[1/6] Lendo CSVs…")
    orders = pd.read_csv(f"{data_dir}/orders.csv")
    op_prior = pd.read_csv(f"{data_dir}/order_products__prior.csv")
    op_train = pd.read_csv(f"{data_dir}/order_products__train.csv")
    products = pd.read_csv(f"{data_dir}/products.csv")
    aisles = pd.read_csv(f"{data_dir}/aisles.csv")
    departments = pd.read_csv(f"{data_dir}/departments.csv")

    # usuários com eval_set='train' (terão um "último pedido" em op_train)
    train_users = orders.loc[orders.eval_set == 'train', ['user_id', 'order_id']]
    if max_users is not None and max_users < len(train_users):
        train_users = train_users.sample(n=max_users, random_state=42)

    # prior somente desses usuários
    prior_orders = orders.merge(train_users[['user_id']], on='user_id')
    prior_orders = prior_orders[prior_orders.eval_set == 'prior'][['order_id', 'user_id']]
    op_prior = op_prior.merge(prior_orders[['order_id', 'user_id']], on='order_id')

    # filtra produtos/usuários raros
    prod_freq = op_prior['product_id'].value_counts()
    keep_products = set(prod_freq[prod_freq >= min_product_freq].index)
    op_prior = op_prior[op_prior.product_id.isin(keep_products)]

    user_order_counts = op_prior.groupby('user_id').order_id.nunique()
    keep_users = set(user_order_counts[user_order_counts >= min_orders_per_user].index)
    op_prior = op_prior[op_prior.user_id.isin(keep_users)]
    train_users = train_users[train_users.user_id.isin(keep_users)]

    # mapeamentos
    user_ids = sorted(train_users.user_id.unique())
    item_ids = sorted(op_prior.product_id.unique())
    user_map = {u: i for i, u in enumerate(user_ids)}
    item_map = {p: j for j, p in enumerate(item_ids)}
    user_rev = user_ids
    item_rev = item_ids

    # matriz U×I (treino) do PRIOR
    print("[2/6] Construindo matriz usuário×item (treino)…")
    rows = op_prior.user_id.map(user_map).values
    cols = op_prior.product_id.map(item_map).values
    data = np.ones_like(rows, dtype=np.float32)
    U, I = len(user_ids), len(item_ids)
    X_train = sp.csr_matrix((data, (rows, cols)), shape=(U, I), dtype=np.float32)

    # verdade de teste (último pedido por usuário, vindo do TRAIN)
    print("[3/6] Preparando conjunto de teste (último pedido por usuário)…")
    tr_idx = orders[orders.eval_set == 'train'][['order_id', 'user_id']]
    op_train = op_train.merge(tr_idx, on='order_id')
    op_train = op_train[op_train.user_id.isin(keep_users)]
    op_train = op_train[op_train.product_id.isin(keep_products)]
    test_truth: Dict[int, np.ndarray] = {}
    for uid, grp in op_train.groupby('user_id'):
        uidx = user_map.get(uid)
        if uidx is None:
            continue
        items = [item_map[p] for p in grp.product_id.values if p in item_map]
        if items:
            test_truth[uidx] = np.unique(np.array(items, dtype=int))

    # ---------- FEATURES DE CONTEÚDO ALINHADAS A item_ids ----------
    print("[4/6] Criando features de conteúdo dos itens…")
    prod_full = (
        products.merge(aisles, on='aisle_id', how='left')
                .merge(departments, on='department_id', how='left')
                .set_index('product_id')
    )
    meta = prod_full.reindex(item_ids)  # I linhas, na ordem certa
    meta['product_name'] = meta['product_name'].fillna('')
    meta['aisle']        = meta['aisle'].fillna('unknown_aisle')
    meta['department']   = meta['department'].fillna('unknown_department')
    meta['tokenized_name'] = meta['product_name'].str.lower()

    try:
        tfidf = TfidfVectorizer(min_df=5, max_df=0.5)
        

        tfidf_mat = tfidf.fit_transform(meta['tokenized_name'])  # I × V
    except Exception:
        tfidf_mat = sp.csr_matrix((len(meta), 0))

    aisle_codes = pd.Categorical(meta['aisle']).codes
    dept_codes  = pd.Categorical(meta['department']).codes
    A = sp.csr_matrix(
        (np.ones(len(aisle_codes), dtype=np.float32), (np.arange(len(aisle_codes)), aisle_codes)),
        shape=(len(aisle_codes), int(aisle_codes.max()) + 1),
    )
    D = sp.csr_matrix(
        (np.ones(len(dept_codes), dtype=np.float32), (np.arange(len(dept_codes)), dept_codes)),
        shape=(len(dept_codes), int(dept_codes.max()) + 1),
    )
    item_features = sp.hstack([A, D, tfidf_mat], format='csr')
    item_features = normalize(item_features, norm='l2', axis=1)

    # popularidade global
    item_popularity = np.asarray(X_train.sum(axis=0)).ravel()

    print("[5/6] Bundle pronto:", X_train.shape[0], "usuários x", X_train.shape[1], "itens")
    return DataBundle(user_map, item_map, user_rev, item_rev, X_train, test_truth, item_features, item_popularity)


# ===================== MODELOS ===================== #
def recommend_popularity(db: DataBundle, K: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = db.item_popularity.astype(np.float32)
    order = np.argsort(-scores)
    topk = order[:K]
    U = db.X_train.shape[0]
    recs = np.tile(topk[None, :], (U, 1))
    scores_out = np.tile(scores[topk][None, :], (U, 1))
    return recs, scores_out


def recommend_itemknn(db: DataBundle, K: int, knorm: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Item-KNN (cosseno aproximado) sem construir matriz item×item.
    ux (1×I) @ Xn.T (I×U) -> tmp (1×U); tmp @ Xn (U×I) -> scores (1×I).
    """
    X = db.X_train.tocsr()            # U × I
    X_bin = X.copy(); X_bin.data[:] = 1.0
    Xn = normalize(X_bin, norm='l2', axis=0)  # normaliza por item (coluna) → U × I

    U, I = X.shape
    recs = np.zeros((U, K), dtype=int)
    scores_out = np.zeros((U, K), dtype=float)

    for u in tqdm(range(U), desc='ItemKNN'):
        ux = X[u]  # 1 × I
        if ux.nnz == 0:
            recs[u], scores_out[u] = recommend_popularity(db, K)
            continue

        tmp = ux.dot(Xn.T)           # 1 × U
        s = tmp.dot(Xn)              # 1 × I
        if sp.issparse(s):
            s = s.toarray().ravel()
        else:
            s = np.asarray(s).ravel()

        if knorm:
            m, v = s.mean(), s.std() + 1e-8
            s = (s - m) / v

        # não recomendar itens já vistos
        if ux.indices.size:
            s[ux.indices] = -1e9

        topk = np.argpartition(-s, K)[:K]
        topk = topk[np.argsort(-s[topk])]
        recs[u] = topk
        scores_out[u] = s[topk]
    return recs, scores_out


def recommend_content(db: DataBundle, K: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perfil do usuário = média dos vetores de itens comprados (denso).
    Score = F (I×F) · prof (F,)  -> vetor denso (I,).
    """
    X = db.X_train.tocsr()
    F = db.item_features.tocsr()      # I × F  (esparsa)
    U, I = X.shape

    recs = np.zeros((U, K), dtype=int)
    scores_out = np.zeros((U, K), dtype=float)

    for u in tqdm(range(U), desc='Content'):
        items_u = X[u].indices
        if items_u.size == 0:
            recs[u], scores_out[u] = recommend_popularity(db, K)
            continue

        # perfil denso do usuário (1×F -> (F,))
        prof = F[items_u].mean(axis=0).A1  # .A1 = toarray().ravel()

        # scores densos (I,)
        s = F.dot(prof)                    # csr · ndarray -> ndarray denso (I,)
        s = np.asarray(s, dtype=np.float32)

        # não recomendar itens já vistos
        s[items_u] = -1e9

        # top-K
        topk = np.argpartition(-s, K)[:K]
        topk = topk[np.argsort(-s[topk])]
        recs[u] = topk
        scores_out[u] = s[topk]

    return recs, scores_out



def recommend_als(
    db: DataBundle,
    K: int,
    factors: int = 64,
    alpha: float = 20.0,
    reg: float = 0.05,
    iterations: int = 15,
) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_IMPLICIT:
        print("[ALS] Biblioteca 'implicit' não encontrada – pulando este modelo.")
        U = db.X_train.shape[0]
        return np.zeros((U, K), dtype=int), np.zeros((U, K), dtype=float)

    print("Treinando ALS (implicit)…")
    Cui = db.X_train.tocoo(copy=True)
    Cui.data = 1.0 + alpha * Cui.data
    Ciu = sp.csr_matrix((Cui.data, (Cui.col, Cui.row)), shape=(db.X_train.shape[1], db.X_train.shape[0]))
    model = implicit.als.AlternatingLeastSquares(factors=factors, regularization=reg, iterations=iterations)
    model.fit(Ciu)

    U = db.X_train.shape[0]
    recs = np.zeros((U, K), dtype=int)
    scores_out = np.zeros((U, K), dtype=float)
    for u in tqdm(range(U), desc='ALS recommend'):
        user_items = db.X_train[u]
        ids, scr = model.recommend(userid=u, user_items=user_items, N=K, filter_already_liked_items=True)
        recs[u, :len(ids)] = np.array(ids)
        scores_out[u, :len(scr)] = np.array(scr)
    return recs, scores_out


def recommend_hybrid(parts: List[Tuple[np.ndarray, np.ndarray]],
                     weights: List[float],
                     K: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Híbrido por votação/score em CANDIDATOS:
    - Candidatos = união dos itens recomendados pelos modelos (top-K de cada um).
    - Score combinado = soma ponderada dos z-scores dos modelos (faltantes = muito baixo).
    - Retorna top-K entre os candidatos para cada usuário.
    """
    assert len(parts) == len(weights)
    U, K_base = parts[0][0].shape
    if K is None:
        K = K_base

    recs_h = np.zeros((U, K), dtype=int)
    scores_h = np.zeros((U, K), dtype=float)

    # pré-calcula médias/desvios por usuário e por modelo (z-score)
    zparts = []
    for (recs, scores) in parts:
        mu = scores.mean(axis=1, keepdims=True)
        sd = scores.std(axis=1, keepdims=True) + 1e-8
        zparts.append((recs, (scores - mu) / sd))

    for u in range(U):
        # 1) união de candidatos
        cand_set: set[int] = set()
        for (recs, _z) in zparts:
            cand_set.update(recs[u])
        if not cand_set:
            # fallback improvável: nenhum candidato
            continue
        cand = np.fromiter(cand_set, dtype=int)
        idx_map = {itm: i for i, itm in enumerate(cand)}
        comb = np.zeros(len(cand), dtype=np.float32)

        # 2) agrega z-scores ponderados
        for w, (recs, zsc) in zip(weights, zparts):
            items_u = recs[u]         # itens recomendados por esse modelo ao usuário u
            scores_u = zsc[u]         # z-scores correspondentes (mesma ordem)
            # para itens não presentes, consideramos contribuição ~ muito baixa
            base = scores_u.min() - 1.0
            for itm, zs in zip(items_u, scores_u):
                comb[idx_map[itm]] += w * zs
            # (itens faltantes ficam só com contribuição dos outros modelos)

        # 3) escolhe top-K entre candidatos
        k_sel = min(K, comb.size)
        if k_sel == 0:
            continue
        topk_idx = np.argpartition(-comb, k_sel - 1)[:k_sel]
        topk_idx = topk_idx[np.argsort(-comb[topk_idx])]

        top_items = cand[topk_idx]
        top_scores = comb[topk_idx]

        # se faltou item para completar K, preenche repetindo os últimos (não afeta métricas)
        if k_sel < K:
            pad = K - k_sel
            top_items = np.pad(top_items, (0, pad), mode='edge')
            top_scores = np.pad(top_scores, (0, pad), mode='edge')

        recs_h[u] = top_items[:K]
        scores_h[u] = top_scores[:K]

    return recs_h, scores_h



# ===================== AVALIAÇÃO ===================== #
def precision_recall_at_k(pred: np.ndarray, truth_sets: List[set], K: int) -> Tuple[float, float, float]:
    hits = []
    recalls = []
    for u, gt in enumerate(truth_sets):
        if not gt:
            continue
        topk = pred[u, :K]
        h = len(set(topk) & gt)
        hits.append(h / K)
        recalls.append(h / len(gt))
    prec = float(np.mean(hits)) if hits else 0.0
    rec = float(np.mean(recalls)) if recalls else 0.0
    f1 = 2 * prec * rec / (prec + rec + 1e-12)
    return prec, rec, f1


def apk(actual: List[int], predicted: List[int], k: int) -> float:
    if len(predicted) > k:
        predicted = predicted[:k]
    score = 0.0
    num_hits = 0.0
    for i, p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
    return score / min(len(actual), k) if actual else 0.0


def mapk(truth: List[List[int]], pred: np.ndarray, k: int) -> float:
    return float(np.mean([apk(t, list(pred[i, :k]), k) for i, t in enumerate(truth) if t]))


def ndcg_at_k(truth: List[List[int]], pred: np.ndarray, k: int) -> float:
    ndcgs = []
    for i, t in enumerate(truth):
        if not t:
            continue
        topk = list(pred[i, :k])
        dcg = 0.0
        for j, p in enumerate(topk, start=1):
            rel = 1.0 if p in t else 0.0
            if rel:
                dcg += 1.0 / np.log2(j + 1)
        ideal_hits = min(len(t), k)
        idcg = sum(1.0 / np.log2(j + 1) for j in range(1, ideal_hits + 1)) or 1.0
        ndcgs.append(dcg / idcg)
    return float(np.mean(ndcgs)) if ndcgs else 0.0


def evaluate_model(name: str, recs: np.ndarray, truth_vecs: List[List[int]], Ks: List[int]) -> Dict[str, Dict[str, float]]:
    truth_sets = [set(t) for t in truth_vecs]
    metrics = {}
    for K in Ks:
        p, r, f1 = precision_recall_at_k(recs, truth_sets, K)
        metrics[f"K={K}"] = {
            'precision': p,
            'recall': r,
            'F1': f1,
            'MAP': mapk(truth_vecs, recs, K),
            'NDCG': ndcg_at_k(truth_vecs, recs, K)
        }
    print(f"\n{name} ->", {k: round(v['NDCG'], 4) for k, v in metrics.items()})
    return metrics



# ===================== MAIN ===================== #
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='outputs')
    parser.add_argument('--max_users', type=int, default=None, help='Amostra de usuários (para velocidade)')
    parser.add_argument('--run', type=str, default='all', choices=['all', 'prep', 'models', 'eval'])
    parser.add_argument('--als', action='store_true', help='Força o treino de ALS (se implicit estiver instalado)')
    parser.add_argument('--K', type=str, default='5,10,20')
    args = parser.parse_args()

    Ks = [int(k) for k in args.K.split(',')]

    db = load_instacart(args.data_dir, max_users=args.max_users)

    # ground truth por uid
    U = db.X_train.shape[0]
    truth_vecs: List[List[int]] = [list(db.test_truth.get(u, [])) for u in range(U)]

    results_summary: List[Dict[str, float]] = []

    # Popularidade
    t0 = time.time()
    rec_pop, sc_pop = recommend_popularity(db, max(Ks))
    t_pop = (time.time() - t0) / U
    m_pop = evaluate_model('Popularidade', rec_pop, truth_vecs, Ks)

    # ItemKNN
    t0 = time.time()
    rec_knn, sc_knn = recommend_itemknn(db, max(Ks))
    t_knn = (time.time() - t0) / U
    m_knn = evaluate_model('CF_item', rec_knn, truth_vecs, Ks)

    # Conteúdo
    t0 = time.time()
    rec_cb, sc_cb = recommend_content(db, max(Ks))
    t_cb = (time.time() - t0) / U
    m_cb = evaluate_model('Conteudo', rec_cb, truth_vecs, Ks)

    # ALS (opcional)
    if args.als and HAS_IMPLICIT:
        t0 = time.time()
        rec_als, sc_als = recommend_als(db, max(Ks))
        t_als = (time.time() - t0) / U
        m_als = evaluate_model('ALS', rec_als, truth_vecs, Ks)
    else:
        rec_als = sc_als = None
        t_als = None
        m_als = None

    # Híbrido (KNN + Conteúdo + Popularidade)
    parts = [(rec_knn, sc_knn), (rec_cb, sc_cb), (rec_pop, sc_pop)]
    weights = [0.5, 0.4, 0.1]
    t0 = time.time()
    rec_h, sc_h = recommend_hybrid(parts, weights, K=max(Ks))
    t_h = (time.time() - t0) / U
    m_h = evaluate_model('Hibrido', rec_h, truth_vecs, Ks)





    # ---- Métricas de negócio (K=10) ----
    biz_rows = []
    pairs = [
        ("Popularidade", rec_pop),
        ("CF_item",      rec_knn),
        ("Conteudo",     rec_cb),
        ("Hibrido",      rec_h),
    ]
    for name, rec in pairs:
        bm = business_metrics(db, rec, K=10)
        biz_rows.append({"modelo": name, "K": 10, **bm})
        print(f"[{name}] cobertura@10 = {bm['coverage_%']:.2f}% | "
            f"diversidade@10 = {bm['intra_list_diversity']:.3f} | "
            f"popularidade média = {bm['avg_popularity_%']:.3f}%")

    pd.DataFrame(biz_rows).to_csv(os.path.join(args.out_dir, "business_metrics.csv"), index=False)
    print("Salvo em:", os.path.join(args.out_dir, "business_metrics.csv"))






    # compila tabela
    def rows_from_metrics(name, metrics, tpred):
        for k, d in metrics.items():
            results_summary.append({
                'modelo': name,
                'K': int(k.split('=')[1]),
                'precision': d['precision'],
                'recall': d['recall'],
                'F1': d['F1'],
                'MAP': d['MAP'],
                'NDCG': d['NDCG'],
                'tempo_pred_ms': (tpred or 0.0) * 1000
            })

    rows_from_metrics('Popularidade', m_pop, t_pop)
    rows_from_metrics('CF_item', m_knn, t_knn)
    rows_from_metrics('Conteudo', m_cb, t_cb)
    if m_als is not None:
        rows_from_metrics('ALS', m_als, t_als)
    rows_from_metrics('Hibrido', m_h, t_h)

    os.makedirs(args.out_dir, exist_ok=True)
    summary = pd.DataFrame(results_summary)
    summary.sort_values(['K', 'NDCG'], ascending=[True, False]).to_csv(
        f"{args.out_dir}/metrics_summary.csv", index=False
    )
    print("\nSalvo em:", f"{args.out_dir}/metrics_summary.csv")

    # Gráficos
    try:
        import matplotlib.pyplot as plt

        # Figura 1 — Precision/Recall@K
        fig1 = plt.figure()
        model_metrics = [('Popularidade', m_pop), ('CF_item', m_knn), ('Conteudo', m_cb)]
        if m_als is not None:
            model_metrics.append(('ALS', m_als))
        model_metrics.append(('Hibrido', m_h))

        for name, mm in model_metrics:
            ks_sorted = sorted([int(x.split('=')[1]) for x in mm.keys()])
            precs = [mm[f'K={k}']['precision'] for k in ks_sorted]
            recs  = [mm[f'K={k}']['recall'] for k in ks_sorted]
            plt.plot(ks_sorted, precs, marker='o', label=f'{name} – precision')
            plt.plot(ks_sorted, recs,  marker='x', label=f'{name} – recall')
        plt.xlabel('K'); plt.ylabel('Métrica'); plt.title('Precision@K e Recall@K')
        plt.legend(); plt.grid(True)
        fig1.savefig(f"{args.out_dir}/precision_recall_curves.png", bbox_inches='tight', dpi=140)

        # Figura 2 — NDCG@10 × tempo de predição (ms)
        fig2 = plt.figure()
        df10 = summary[summary['K'] == 10]
        plt.scatter(df10['tempo_pred_ms'], df10['NDCG'])
        for _, r in df10.iterrows():
            plt.annotate(r['modelo'], (r['tempo_pred_ms'], r['NDCG']))
        plt.xlabel('Tempo de predição médio (ms/usuário)')
        plt.ylabel('NDCG@10')
        plt.title('NDCG@10 × tempo de predição')
        plt.grid(True)
        fig2.savefig(f"{args.out_dir}/ndcg10_vs_predtime.png", bbox_inches='tight', dpi=140)

        print("Figuras salvas em:", args.out_dir)
    except Exception as e:
        print("[Plot] Falha ao gerar gráficos:", e)




if __name__ == '__main__':
    main()
