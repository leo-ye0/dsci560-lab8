import re
import glob
import difflib
import numpy as np
import pandas as pd
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score

POSTS = "data/posts.csv"
CONFIGS = [1,2,3]
KS = [3,4,5,6]

def norm(s: str) -> str:
    s = (s or "").strip().strip('"').strip("'")
    s = s.replace("’", "'").replace("“","\"").replace("”","\"").replace("–","-").replace("—","-")
    s = re.sub(r"\s+", " ", s)
    return s.lower()

def extract_title(line: str) -> str:
    line = line.strip()
    m = re.match(r"\d+\.\s*(.*)", line)
    if m: line = m.group(1)
    line = re.split(r"\|\s*keywords\s*:|[\s|：:;；-]+\s*keywords\s*:", line, flags=re.I)[0]
    return line.strip().strip('"').strip("'")

def load_cluster_titles(files):
    by_cluster = []
    for cid, path in enumerate(files):
        titles = []
        with open(path, encoding="utf-8-sig") as f:
            for raw in f:
                t = extract_title(raw)
                if t:
                    titles.append(t)
        by_cluster.append((cid, titles))
    return by_cluster

def match_labels(titles_posts_norm, by_cluster, verbose_tag=""):
    exact = {}
    cluster_titles_norm = []
    for cid, titles in by_cluster:
        lst = [norm(t) for t in titles]
        cluster_titles_norm.append((cid, lst))
        for t in lst:
            exact[t] = cid
    all_norm_titles = [t for _, lst in cluster_titles_norm for t in lst]

    labels = np.full(len(titles_posts_norm), -1, dtype=int)
    hits_exact = hits_fuzzy = 0
    for i, tp in enumerate(titles_posts_norm):
        cid = exact.get(tp)
        if cid is not None:
            labels[i] = cid
            hits_exact += 1
            continue
        found = None
        for cid2, lst in cluster_titles_norm:
            for ct in lst:
                if tp in ct or ct in tp:
                    found = cid2; break
            if found is not None: break
        if found is not None:
            labels[i] = found
            hits_fuzzy += 1
            continue
        cand = difflib.get_close_matches(tp, all_norm_titles, n=1, cutoff=0.85)
        if cand:
            ct = cand[0]
            for cid3, lst in cluster_titles_norm:
                if ct in lst:
                    labels[i] = cid3
                    hits_fuzzy += 1
                    break

    mapped = int((labels != -1).sum())
    print(f"[map] {verbose_tag} exact={hits_exact} fuzzy={hits_fuzzy} total={mapped}/{len(labels)}")
    return labels

def purity_score(y_true, y_pred):
    from collections import Counter
    corr = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        if len(idx)==0: continue
        corr += Counter(y_true[idx]).most_common(1)[0][1]
    return corr / len(y_true)

def find_files(pattern):
    return sorted(glob.glob(f"**/{pattern}", recursive=True))

# main
posts = pd.read_csv(POSTS)
titles_raw = posts["title"].astype(str).tolist()
titles_norm = [norm(t) for t in titles_raw]
y_true = posts["topics"].astype("category").cat.codes.to_numpy()

records = []
any_pair = False

for config in CONFIGS:
    for k in KS:
        w2v_files = find_files(f"config{config}_k{k}_cluster_*.txt")
        d2v_files = find_files(f"model{config}_k{k}_cluster_*.txt")
        print(f"[scan] c{config} k{k}: W2V={len(w2v_files)} D2V={len(d2v_files)}")
        if not w2v_files or not d2v_files:
            continue

        w2v_byc = load_cluster_titles(w2v_files)
        d2v_byc = load_cluster_titles(d2v_files)
        w2v_labels = match_labels(titles_norm, w2v_byc, verbose_tag=f"W2V c{config}k{k}")
        d2v_labels = match_labels(titles_norm, d2v_byc, verbose_tag=f"D2V c{config}k{k}")

        mask_w = (w2v_labels != -1)
        mask_d = (d2v_labels != -1)
        mask_both = mask_w & mask_d
        if mask_both.sum() == 0:
            print(f"[skip] c{config}k{k} no overlap\n")
            continue

        any_pair = True
        p_w = purity_score(y_true[mask_w], w2v_labels[mask_w]) if mask_w.sum() else np.nan
        p_d = purity_score(y_true[mask_d], d2v_labels[mask_d]) if mask_d.sum() else np.nan
        nmi = normalized_mutual_info_score(w2v_labels[mask_both], d2v_labels[mask_both])
        ari = adjusted_rand_score(w2v_labels[mask_both], d2v_labels[mask_both])

        print(f"[OK] c{config}k{k}: Purity W2V={p_w:.4f} D2V={p_d:.4f}  NMI={nmi:.4f} ARI={ari:.4f} (n={mask_both.sum()})\n")
        records.append([config, k, int(mask_both.sum()), float(p_w), float(p_d), float(nmi), float(ari)])

if any_pair:
    out = "cluster_summary.csv"
    pd.DataFrame(records, columns=["Config","k","n_overlap","Purity_W2V","Purity_D2V","NMI","ARI"]).to_csv(out, index=False)
    print(f"Saved summary to {out}")
else:
    print("No valid results found.")
