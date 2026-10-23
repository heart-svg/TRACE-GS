from __future__ import annotations

import csv
import math
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd


ROOT = Path(r"/path/to/trace_gs_workspace")
OUT = Path(r"/path/to/article_workspace\结果")
OUT.mkdir(parents=True, exist_ok=True)
OBO = ROOT / "go-basic.obo"

TRAITS = {
    "FT10": ROOT / "paper_completion_20261015/go_enrichment/FT10",
    "FT16": ROOT / "paper_completion_20261015/go_enrichment/FT16",
    "Diameter": ROOT / "paper_completion_20261015/go_enrichment/Diameter",
    "100-grain weight": ROOT / "paper_completion_20261015/go_enrichment/maize_100grainweight",
    "Heading date": ROOT / "paper_completion_20261015/go_enrichment/maize_Headingdate",
    "Fly starvation": ROOT / "paper_completion_20261015/animal_interpretability/drosophila",
    "Stickleback immunity": ROOT / "paper_completion_20261015/animal_interpretability/stickleback",
}

TRAIT_ORDER = list(TRAITS)


def parse_obo(path: Path):
    names, namespace, parents, obsolete = {}, {}, defaultdict(list), set()
    current = None
    with path.open(encoding="utf-8") as handle:
        for raw in handle:
            line = raw.rstrip("\n")
            if line == "[Term]":
                current = {}
            elif line.startswith("["):
                current = None
            elif current is not None and ": " in line:
                key, value = line.split(": ", 1)
                if key == "id":
                    current["id"] = value
                elif key == "name":
                    current["name"] = value
                elif key == "namespace":
                    current["namespace"] = value
                elif key == "is_a":
                    current.setdefault("parents", []).append((value.split()[0], 0.8))
                elif key == "relationship" and value.startswith("part_of "):
                    current.setdefault("parents", []).append((value.split()[1], 0.6))
                elif key == "is_obsolete" and value == "true":
                    current["obsolete"] = True
            elif current is not None and line == "" and "id" in current:
                go = current["id"]
                names[go] = current.get("name", go)
                namespace[go] = current.get("namespace", "")
                parents[go].extend(current.get("parents", []))
                if current.get("obsolete"):
                    obsolete.add(go)
                current = None
    return names, namespace, parents, obsolete


NAMES, NAMESPACE, PARENTS, OBSOLETE = parse_obo(OBO)


def wang_contributions(term: str):
    scores = {term: 1.0}
    queue = deque([term])
    while queue:
        child = queue.popleft()
        child_score = scores[child]
        for parent, weight in PARENTS.get(child, []):
            candidate = child_score * weight
            if candidate > scores.get(parent, 0.0) + 1e-12:
                scores[parent] = candidate
                queue.append(parent)
    return scores


_WANG_CACHE = {}


def wang_similarity(a: str, b: str) -> float:
    if a == b:
        return 1.0
    sa = _WANG_CACHE.setdefault(a, wang_contributions(a))
    sb = _WANG_CACHE.setdefault(b, wang_contributions(b))
    shared = set(sa).intersection(sb)
    denom = sum(sa.values()) + sum(sb.values())
    return 0.0 if denom == 0 else sum(sa[x] + sb[x] for x in shared) / denom


def semantically_reduce(df: pd.DataFrame, fdr_col: str, score_col: str | None, threshold=0.70):
    if df.empty:
        return df.assign(cluster_id=pd.Series(dtype=int), representative=pd.Series(dtype=bool), representative_go="", semantic_similarity_to_rep=float())
    work = df.copy()
    work["_fdr"] = pd.to_numeric(work[fdr_col], errors="coerce").fillna(1.0)
    work["_score"] = 0.0 if score_col is None else pd.to_numeric(work[score_col], errors="coerce").abs().fillna(0.0)
    work["_size"] = pd.to_numeric(work.get("gene_count", 0), errors="coerce").fillna(0)
    work = work.sort_values(["_fdr", "_score", "_size", "go_id"], ascending=[True, False, False, True]).reset_index(drop=True)
    cluster, rep_flag, rep_go, sim_to_rep = [None] * len(work), [False] * len(work), [None] * len(work), [None] * len(work)
    cid = 0
    for i, row in work.iterrows():
        if cluster[i] is not None:
            continue
        cid += 1
        representative = row["go_id"]
        cluster[i], rep_flag[i], rep_go[i], sim_to_rep[i] = cid, True, representative, 1.0
        for j in range(i + 1, len(work)):
            if cluster[j] is not None:
                continue
            sim = wang_similarity(representative, work.loc[j, "go_id"])
            if sim >= threshold:
                cluster[j], rep_go[j], sim_to_rep[j] = cid, representative, sim
    work["cluster_id"] = cluster
    work["representative"] = rep_flag
    work["representative_go"] = rep_go
    work["semantic_similarity_to_rep"] = sim_to_rep
    return work.drop(columns=["_fdr", "_score", "_size"])


def read_ora(trait: str, folder: Path):
    path = folder / "ORA_results.csv"
    if not path.exists():
        return pd.DataFrame(columns=["trait", "analysis", "go_id", "term_name", "namespace", "gene_ratio", "gene_count", "set_size", "p_value", "fdr", "genes", "source_file"])
    x = pd.read_csv(path)
    if "term_id" in x.columns:
        out = pd.DataFrame({
            "trait": trait, "analysis": "ORA", "go_id": x["term_id"], "term_name": x["term_name"],
            "namespace": x.get("namespace", ""), "gene_ratio": x["stable_overlap"] / x["stable_total"],
            "gene_count": x["stable_overlap"], "set_size": x["set_size"], "p_value": x["P"],
            "fdr": x["FDR_BH"], "genes": x["genes"], "source_file": str(path),
        })
    else:
        out = pd.DataFrame({
            "trait": trait, "analysis": "ORA", "go_id": x["go_id"], "term_name": x["term_name"],
            "namespace": x["go_id"].map(NAMESPACE).fillna(""), "gene_ratio": x["overlap_size"] / x["query_size"],
            "gene_count": x["overlap_size"], "set_size": x["term_size"], "p_value": x["p_value"],
            "fdr": x["fdr_bh"], "genes": x["overlap_genes"], "source_file": str(path),
        })
    out["obsolete"] = out["go_id"].isin(OBSOLETE)
    return out


def read_gsea(trait: str, folder: Path):
    path = folder / "GSEA_results.csv"
    x = pd.read_csv(path)
    names = x["term_name"] if "term_name" in x.columns else x["Term"].map(NAMES)
    tag_num = x["Tag %"].astype(str).str.split("/").str[0]
    out = pd.DataFrame({
        "trait": trait, "analysis": "GSEA", "go_id": x["Term"], "term_name": names.fillna(x["Term"]),
        "namespace": x["Term"].map(NAMESPACE).fillna(""), "ES": x["ES"], "NES": x["NES"],
        "p_value": x["NOM p-val"], "fdr": x["FDR q-val"], "fwer": x["FWER p-val"],
        "gene_count": pd.to_numeric(tag_num, errors="coerce"), "tag_fraction": x["Tag %"],
        "gene_fraction": x["Gene %"], "leading_edge_genes": x["Lead_genes"], "source_file": str(path),
    })
    out["obsolete"] = out["go_id"].isin(OBSOLETE)
    return out


ora_all = pd.concat([read_ora(t, p) for t, p in TRAITS.items()], ignore_index=True)
gsea_all = pd.concat([read_gsea(t, p) for t, p in TRAITS.items()], ignore_index=True)
ora_all["significant_fdr_0.05"] = pd.to_numeric(ora_all["fdr"], errors="coerce") < 0.05
gsea_all["significant_fdr_0.05"] = pd.to_numeric(gsea_all["fdr"], errors="coerce") < 0.05

ora_reduced, gsea_reduced = [], []
for trait in TRAIT_ORDER:
    o = ora_all[(ora_all.trait == trait) & ora_all["significant_fdr_0.05"] & ~ora_all["obsolete"]]
    g = gsea_all[(gsea_all.trait == trait) & gsea_all["significant_fdr_0.05"] & ~gsea_all["obsolete"]]
    ora_reduced.append(semantically_reduce(o, "fdr", None))
    gsea_reduced.append(semantically_reduce(g, "fdr", "NES"))
ora_reduced = pd.concat(ora_reduced, ignore_index=True) if ora_reduced else pd.DataFrame()
gsea_reduced = pd.concat(gsea_reduced, ignore_index=True) if gsea_reduced else pd.DataFrame()

# Fixed plotting rule: after semantic reduction, retain at most two representative
# terms per trait. ORA ranks by FDR then gene ratio; GSEA ranks by |NES| then FDR.
ora_plot_parts, gsea_plot_parts = [], []
for trait in TRAIT_ORDER:
    o = ora_reduced[(ora_reduced.trait == trait) & ora_reduced.representative].copy()
    if not o.empty:
        o = o.sort_values(["fdr", "gene_ratio", "gene_count"], ascending=[True, False, False]).head(2)
        ora_plot_parts.append(o)
    g = gsea_reduced[(gsea_reduced.trait == trait) & gsea_reduced.representative].copy()
    if not g.empty:
        g["abs_NES"] = pd.to_numeric(g["NES"], errors="coerce").abs()
        g = g.sort_values(["abs_NES", "fdr"], ascending=[False, True]).head(2)
        gsea_plot_parts.append(g)
ora_plot = pd.concat(ora_plot_parts, ignore_index=True) if ora_plot_parts else pd.DataFrame()
gsea_plot = pd.concat(gsea_plot_parts, ignore_index=True) if gsea_plot_parts else pd.DataFrame()

# Human-readable labels remain deterministic and retain GO IDs in the table.
for data in (ora_plot, gsea_plot):
    if not data.empty:
        data["display_term"] = data["term_name"].astype(str).str.replace(r"^obsolete\s+", "", regex=True)
        data["minus_log10_fdr"] = -data["fdr"].clip(lower=1e-300).map(math.log10)
        data["trait_order"] = data["trait"].map({v: i + 1 for i, v in enumerate(TRAIT_ORDER)})

with pd.ExcelWriter(OUT / "补充表_GO_ORA_GSEA完整条目.xlsx", engine="openpyxl") as writer:
    ora_all.to_excel(writer, sheet_name="ORA_all_reported", index=False)
    gsea_all.to_excel(writer, sheet_name="GSEA_all_reported", index=False)
    ora_reduced.to_excel(writer, sheet_name="ORA_semantic_clusters", index=False)
    gsea_reduced.to_excel(writer, sheet_name="GSEA_semantic_clusters", index=False)

ora_all.to_csv(OUT / "补充表_ORA全部报告条目.csv", index=False, encoding="utf-8-sig")
gsea_all.to_csv(OUT / "补充表_GSEA全部报告条目.csv", index=False, encoding="utf-8-sig")
ora_plot.to_csv(OUT / "图4b_ORA气泡图数据.csv", index=False, encoding="utf-8-sig")
gsea_plot.to_csv(OUT / "图4c_GSEA去冗余NES数据.csv", index=False, encoding="utf-8-sig")
gsea_reduced[gsea_reduced.trait == "Diameter"].to_csv(OUT / "Diameter_GSEA语义去冗余模块.csv", index=False, encoding="utf-8-sig")

# Standard GSEA running-score source data for FT10 GO:0009911.
rank_path = TRAITS["FT10"] / "gsea_work/prerank_data.rnk"
gmt_path = TRAITS["FT10"] / "gsea_work/gene_sets.gmt"
ranks = pd.read_csv(rank_path, sep="\t", header=None, names=["gene_id", "rank_metric"])
ranks = ranks.sort_values("rank_metric", ascending=False).reset_index(drop=True)
target = "GO:0009911"
members = None
with gmt_path.open(encoding="utf-8") as handle:
    for line in handle:
        fields = line.rstrip("\n").split("\t")
        if fields[0] == target:
            members = set(fields[2:])
            break
if members is None:
    raise RuntimeError(f"{target} absent from {gmt_path}")
ranks["hit"] = ranks["gene_id"].isin(members)
hit_weight = ranks.loc[ranks.hit, "rank_metric"].abs()
norm_hit = hit_weight.sum()
n_miss = (~ranks.hit).sum()
ranks["increment"] = -1.0 / n_miss
ranks.loc[ranks.hit, "increment"] = ranks.loc[ranks.hit, "rank_metric"].abs() / norm_hit
ranks["running_ES"] = ranks["increment"].cumsum()
ranks["rank"] = range(1, len(ranks) + 1)
symbol_map = {"AT2G45660": "SOC1", "AT1G65480": "FT", "AT4G20370": "TSF", "AT5G60910": "FUL"}
ranks["label"] = ranks["gene_id"].map(symbol_map).fillna("")
reported = gsea_all[(gsea_all.trait == "FT10") & (gsea_all.go_id == target)].iloc[0]
ranks["term_id"] = target
ranks["term_name"] = NAMES[target]
ranks["reported_NES"] = reported["NES"]
ranks["reported_FDR"] = reported["fdr"]
ranks.to_csv(OUT / "补充图_FT10_GSEA_flower_development_源数据.csv", index=False, encoding="utf-8-sig")

summary = []
for trait in TRAIT_ORDER:
    raw = gsea_all[(gsea_all.trait == trait) & gsea_all["significant_fdr_0.05"] & ~gsea_all.obsolete]
    red = gsea_reduced[(gsea_reduced.trait == trait) & gsea_reduced.representative]
    summary.append({"trait": trait, "significant_nonobsolete_GSEA_terms": len(raw), "semantic_representative_modules": len(red)})
pd.DataFrame(summary).to_csv(OUT / "GSEA语义去冗余汇总.csv", index=False, encoding="utf-8-sig")
print(pd.DataFrame(summary).to_string(index=False))

