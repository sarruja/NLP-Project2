"""
NLP Project 2 – Helper Functions
Whisper v1 / v2 / v3 auf STT4SG-350 (Schweizerdeutsch)
"""

import warnings
warnings.filterwarnings('ignore')

import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from collections import Counter
from jiwer import process_words, process_characters
import Levenshtein


matplotlib.rcParams['figure.dpi'] = 150
os.makedirs('figures', exist_ok=True)

# ── Konstanten ────────────────────────────────────────────────────────────────

COL_REFERENCE  = 'sentence'
COL_HYPOTHESIS = 'hypothesis'
COL_CANTON     = 'canton'
COL_VERSION    = 'version'

VERSIONS = ['v1', 'v2', 'v3']

RESULTS_DIR = 'results/'
CSV_FILES = {
    'v1': os.path.join(RESULTS_DIR, 'checkpoint_openai_whisper-large.csv'),
    'v2': os.path.join(RESULTS_DIR, 'checkpoint_openai_whisper-large-v2.csv'),
    'v3': os.path.join(RESULTS_DIR, 'checkpoint_openai_whisper-large-v3.csv'),
}

V_COLORS = {'v1': "#3A88D5", 'v2': "#35A13B", 'v3': "#A653C9"}
V_LABELS = {'v1': 'Whisper large-v1', 'v2': 'Whisper large-v2', 'v3': 'Whisper large-v3'}
V_SHORT  = {'v1': 'large-v1', 'v2': 'large-v2', 'v3': 'large-v3'}

# ── Daten als ein DF laden ───────────────────────────────────────────────────────

def load_data():
    """Lädt alle 3 Modell-CSVs und gibt einen kombinierten DataFrame zurück."""
    return pd.concat([
        pd.read_csv(CSV_FILES[v]).assign(version=v)
        for v in VERSIONS
    ])

# ── Text-Normalisierung ───────────────────────────────────────────────────────

def safe_str(text):
    """Standard — mit Satzzeichen (Original)."""
    if pd.isna(text):
        return ''
    return str(text).strip().lower()


def safe_str_clean(text):
    """Ohne Satzzeichen (für Vergleich)."""
    if pd.isna(text):
        return ''
    text = re.sub(r'[^\w\s]', '', str(text))
    return text.strip().lower()


# ── Metriken ─────────────────────────────────────────────────────────────────

def compute_wer(df):
    """WER inkl. Fehlertypen (mit Satzzeichen). Gibt pd.Series zurück → für groupby.apply()."""
    refs  = [safe_str(t) for t in df[COL_REFERENCE]]
    hyps  = [safe_str(t) for t in df[COL_HYPOTHESIS]]
    wer_r = process_words(refs, hyps)
    total_errors = wer_r.substitutions + wer_r.deletions + wer_r.insertions
    return pd.Series({
        'wer':           wer_r.wer,
        'substitutions': wer_r.substitutions,
        'deletions':     wer_r.deletions,
        'insertions':    wer_r.insertions,
        'total_errors':  total_errors,
        'n_samples':     len(df),
    })


def compute_cer(df):
    """CER (mit Satzzeichen). Gibt pd.Series zurück → für groupby.apply()."""
    refs  = [safe_str(t) for t in df[COL_REFERENCE]]
    hyps  = [safe_str(t) for t in df[COL_HYPOTHESIS]]
    cer_r = process_characters(refs, hyps)
    return pd.Series({
        'cer':       cer_r.cer,
        'n_samples': len(df),
    })


def compute_wer_clean(df):
    """WER ohne Satzzeichen. Gibt pd.Series zurück → für groupby.apply()."""
    refs = [safe_str_clean(t) for t in df[COL_REFERENCE]]
    hyps = [safe_str_clean(t) for t in df[COL_HYPOTHESIS]]
    return pd.Series({'wer_clean': process_words(refs, hyps).wer})


def compute_cer_clean(df):
    """CER ohne Satzzeichen. Gibt pd.Series zurück → für groupby.apply()."""
    refs = [safe_str_clean(t) for t in df[COL_REFERENCE]]
    hyps = [safe_str_clean(t) for t in df[COL_HYPOTHESIS]]
    return pd.Series({'cer_clean': process_characters(refs, hyps).cer})


# ── Fehleranalyse ─────────────────────────────────────────────────────────────

def get_top_substitutions(df, top_n=20):
    """Gibt die häufigsten Substitutions-Paare (ref → hyp) zurück."""
    counter = Counter()
    for _, row in df.iterrows():
        ref = safe_str(row[COL_REFERENCE])
        hyp = safe_str(row[COL_HYPOTHESIS])
        if not ref or not hyp:
            continue
        result    = process_words([ref], [hyp])
        ref_words = ref.split()
        hyp_words = hyp.split()
        for chunk in result.alignments[0]:
            if chunk.type == 'substitute':
                rw = ' '.join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                hw = ' '.join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                counter[(rw, hw)] += 1
    return counter.most_common(top_n)


def get_punctuation_errors(df, top_n=50):
    """Findet Substitutionen, Deletions und Insertions wo Satzzeichen beteiligt sind."""
    sub_counter = Counter()
    del_counter = Counter()
    ins_counter = Counter()

    for _, row in df.iterrows():
        ref = safe_str(row[COL_REFERENCE])
        hyp = safe_str(row[COL_HYPOTHESIS])
        if not ref or not hyp:
            continue
        result    = process_words([ref], [hyp])
        ref_words = ref.split()
        hyp_words = hyp.split()

        for chunk in result.alignments[0]:
            if chunk.type == 'substitute':
                rw = ' '.join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                hw = ' '.join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                if re.search(r'[^\w\s]', rw) or re.search(r'[^\w\s]', hw):
                    sub_counter[(rw, hw)] += 1
            elif chunk.type == 'delete':
                rw = ' '.join(ref_words[chunk.ref_start_idx:chunk.ref_end_idx])
                if re.search(r'[^\w\s]', rw):
                    del_counter[rw] += 1
            elif chunk.type == 'insert':
                hw = ' '.join(hyp_words[chunk.hyp_start_idx:chunk.hyp_end_idx])
                if re.search(r'[^\w\s]', hw):
                    ins_counter[hw] += 1

    return sub_counter.most_common(top_n), del_counter.most_common(top_n), ins_counter.most_common(top_n)


def categorize_substitutions(df, top_n=200):
    """Kategorisiert Top-N Substitutionen in linguistische Fehlerkategorien."""
    pairs = get_top_substitutions(df, top_n=top_n)
    categories = {
        'Zahlen':      [],   # "zwanzig" → "20"
        'Eigennamen':  [],   # "Zürich"  → "Zurich"
        'Zerstückelt': [],   # "Postulat" → "Post und Art"
        'Sonstige':    [],   # alles andere (Code-switching, Synonyme etc.)
    }
    for (ref_w, hyp_w), cnt in pairs:
        if re.search(r'\d', hyp_w) and not re.search(r'\d', ref_w):
            categories['Zahlen'].append((ref_w, hyp_w, cnt))
        elif len(hyp_w.split()) > len(ref_w.split()) + 1:
            categories['Zerstückelt'].append((ref_w, hyp_w, cnt))
        elif ref_w and ref_w[0].isupper():
            categories['Eigennamen'].append((ref_w, hyp_w, cnt))
        else:
            categories['Sonstige'].append((ref_w, hyp_w, cnt))
    return categories


def find_punctuation_in_data(df):
    """Zählt alle Satzzeichen im Dataset (REF + HYP)."""
    counter = Counter()
    for text in list(df[COL_REFERENCE]) + list(df[COL_HYPOTHESIS]):
        if pd.isna(text):
            continue
        counter.update(re.findall(r'[^\w\s]', str(text)))
    return counter


def count_punctuation_in_col(series):
    """Zählt Satzzeichen in einer einzelnen Spalte."""
    counter = Counter()
    for text in series:
        if pd.isna(text):
            continue
        counter.update(re.findall(r'[^\w\s]', str(text)))
    return counter

# ── Visualisierungen ──────────────────────────────────────────────────────────

def fig_wer_overall(wer_overall):
    """Fig: Overall WER pro Modell. Erwartet DataFrame mit index=version."""
    fig, ax = plt.subplots(figsize=(6, 4))
    x    = np.arange(len(VERSIONS))
    wers = [wer_overall.loc[v, 'wer'] * 100 for v in VERSIONS]
    bars = ax.bar(x, wers, 0.5, color=[V_COLORS[v] for v in VERSIONS], edgecolor='white')
    ax.bar_label(bars, fmt='%.1f%%', padding=4, fontsize=11, fontweight='bold')
    ax.set_ylabel('WER (%)')
    ax.set_title('Overall WER – Whisper v1 / v2 / v3')
    ax.set_xticks(x)
    ax.set_xticklabels([V_SHORT[v] for v in VERSIONS])
    ax.set_ylim(0, max(wers) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig_wer_overall.png', dpi=200)
    plt.show()
    print('→ figures/fig_wer_overall.png')


def fig_wer_by_canton(wer_by_canton, cantons):
    """Fig: WER pro Kanton. Erwartet DataFrame mit MultiIndex (version, canton)."""
    fig, ax = plt.subplots(figsize=(14, 5))
    x     = np.arange(len(cantons))
    width = 0.25
    for i, v in enumerate(VERSIONS):
        vals = [wer_by_canton.loc[(v, c), 'wer'] * 100 for c in cantons]
        bars = ax.bar(x + (i-1)*width, vals, width,
                      label=V_LABELS[v], color=V_COLORS[v], edgecolor='white', alpha=0.9)
        ax.bar_label(bars, fmt='%.0f%%', padding=2, fontsize=7)
    ax.set_ylabel('WER (%)')
    ax.set_title('WER pro Kanton – Whisper v1 / v2 / v3')
    ax.set_xticks(x)
    ax.set_xticklabels(cantons)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig_wer_by_canton.png', dpi=200)
    plt.show()
    print('→ figures/fig_wer_by_canton.png')


def fig_error_types(wer_overall):
    """Fig: Fehlertypen Stacked Bar (Sub / Del / Ins)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    C_SUB = "#5DADE2"   
    C_DEL = "#52BE80"   
    C_INS = "#BC70DB"  
    x    = np.arange(len(VERSIONS))
    subs = [wer_overall.loc[v, 'substitutions'] / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    dels = [wer_overall.loc[v, 'deletions']     / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    ins  = [wer_overall.loc[v, 'insertions']    / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    ax.bar(x, subs, label='Substitutions', color=C_SUB, edgecolor='white')
    ax.bar(x, dels, bottom=subs, label='Deletions', color=C_DEL, edgecolor='white')
    ax.bar(x, ins,  bottom=[s+d for s,d in zip(subs,dels)], label='Insertions', color=C_INS, edgecolor='white')
    for i in range(len(VERSIONS)):
        if subs[i] > 4: ax.text(i, subs[i]/2,                f'{subs[i]:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if dels[i] > 4: ax.text(i, subs[i]+dels[i]/2,        f'{dels[i]:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if ins[i]  > 4: ax.text(i, subs[i]+dels[i]+ins[i]/2, f'{ins[i]:.1f}%',  ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('Error Type-Distribution – Whisper v1 / v2 / v3')
    ax.set_xticks(x)
    ax.set_xticklabels([V_SHORT[v] for v in VERSIONS])
    ax.set_ylim(0, 115)
    ax.legend()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig_error_types.png', dpi=200)
    plt.show()
    print('→ figures/fig_error_types.png')


def fig_wer_vs_cer(wer_cer_overall):
    """Fig: WER vs CER Overall. Erwartet DataFrame mit columns wer, cer, index=version."""
    fig, ax = plt.subplots(figsize=(7, 4))
    x    = np.arange(len(VERSIONS))
    w    = 0.35
    wers = [wer_cer_overall.loc[v, 'wer'] * 100 for v in VERSIONS]
    cers = [wer_cer_overall.loc[v, 'cer'] * 100 for v in VERSIONS]
    b1 = ax.bar(x - w/2, wers, w, label='WER', color='#54A7EA', edgecolor='white')
    b2 = ax.bar(x + w/2, cers, w, label='CER', color='#19A871', edgecolor='white')
    ax.bar_label(b1, fmt='%.1f%%', padding=3, fontsize=9)
    ax.bar_label(b2, fmt='%.1f%%', padding=3, fontsize=9)
    ax.set_ylabel('Error Rate (%)')
    ax.set_title('WER vs CER – Whisper v1 / v2 / v3')
    ax.set_xticks(x)
    ax.set_xticklabels([V_SHORT[v] for v in VERSIONS])
    ax.legend()
    ax.set_ylim(0, max(wers + cers) * 1.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig('figures/fig_wer_vs_cer.png', dpi=200)
    plt.show()
    print('→ figures/fig_wer_vs_cer.png')


def fig_error_top10(df, top_n=10):
    """Fig: Top-N Error pro Type (Sub/Del/Ins) for all 3 Model."""
    versions_present = [v for v in VERSIONS if v in df[COL_VERSION].unique()]

    C_SUB = "#5DADE2"   
    C_DEL = "#52BE80"   
    C_INS = "#BC70DB"
    
    err_types = [
        ("sub", "Top Substitutions",  C_SUB),
        ("del", "Top Deletions",      C_DEL),
        ("ins", "Top Insertions",     C_INS),
    ]

    # Fehler-Token extrahieren
    error_data = {}
    for v in versions_present:
        sub_c, del_c, ins_c = Counter(), Counter(), Counter()
        df_v = df[df[COL_VERSION] == v]
        print(f"  Extrahiere {v} ({len(df_v):,} rows) ...", end=" ", flush=True)
        for _, row in df_v.iterrows():
            ref = safe_str(row[COL_REFERENCE])
            hyp = safe_str(row[COL_HYPOTHESIS])
            if not ref or not hyp:
                continue
            try:
                out = process_words([ref], [hyp])
            except Exception:
                continue
            rw_all = ref.split()
            hw_all = hyp.split()
            for chunk in out.alignments[0]:
                rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]
                if chunk.type == "substitute":
                    for r, h in zip(rw, hw):
                        sub_c[f"{r} → {h}"] += 1
                elif chunk.type == "delete":
                    for w in rw:
                        del_c[w] += 1
                elif chunk.type == "insert":
                    for w in hw:
                        ins_c[w] += 1
        error_data[v] = {"sub": sub_c, "del": del_c, "ins": ins_c}
        print("done")

    # Plot
    fig, axes = plt.subplots(
        nrows=3, ncols=len(versions_present),
        figsize=(7 * len(versions_present), 18),
        constrained_layout=True
    )
    if len(versions_present) == 1:
        axes = axes.reshape(3, 1)

    for col_i, v in enumerate(versions_present):
        data = error_data[v]
        total_all = sum(data["sub"].values()) + sum(data["del"].values()) + sum(data["ins"].values())

        for row_i, (etype, etype_label, color) in enumerate(err_types):
            ax = axes[row_i][col_i]
            top = data[etype].most_common(top_n)
            if not top:
                ax.axis("off")
                continue

            labels = [t[0] for t in top]
            counts = [t[1] for t in top]
            type_total = sum(data[etype].values())

            bars = ax.barh(range(len(labels)), counts,
                           color=color, alpha=0.82, edgecolor="white", linewidth=0.5)

            for bar, count in zip(bars, counts):
                pct = count / type_total * 100 if type_total > 0 else 0
                ax.text(bar.get_width() + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        f"{count:,}  ({pct:.1f}%)",
                        va="center", ha="left", fontsize=8.5, color="#333333")

            ax.set_yticks(range(len(labels)))
            ax.set_yticklabels(labels, fontsize=9.5)
            ax.invert_yaxis()
            ax.set_xlabel("Count", fontsize=9)
            ax.set_xlim(0, max(counts) * 1.38)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.tick_params(axis="x", labelsize=8)
            ax.grid(axis="x", linestyle="--", alpha=0.3, linewidth=0.6)

            if col_i == 0:
                ax.set_ylabel(etype_label, fontsize=11, fontweight="bold",
                              color=color, labelpad=10)

            ax.set_title(
                f"{etype_label}  —  {type_total:,} total  ({type_total/total_all*100:.1f}% of all errors)",
                fontsize=9.5, color=color, pad=6
            )

        # Spaltenheader
        axes[0][col_i].annotate(
            f"Whisper {V_SHORT[v]}",
            xy=(0.5, 1.22), xycoords="axes fraction",
            ha="center", va="bottom", fontsize=14, fontweight="bold",
            color=V_COLORS[v],
            bbox=dict(boxstyle="round,pad=0.3", facecolor=V_COLORS[v],
                      edgecolor="none", alpha=0.15)
        )

    fig.suptitle(
        f"Top {top_n} Error-Token pro Type — Whisper v1/v2/v3 in Swissgerman (STT4SG-350)",
        fontsize=14, fontweight="bold", color="#1B3A6B", y=1.01
    )
    plt.savefig("figures/fig_error_top10.png", dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
    print("→ figures/fig_error_top10.png")


def fig_error_top10_grouped(df, top_n=10):
    """
    Fig: Top-N Fehler-Token pro Typ (Sub/Del/Ins), gruppierte Balken v1/v2/v3.
    Ranking basiert auf Summe aller 3 Modelle.
    Speichert → figures/fig_error_top10_grouped.png
    """
    C_SUB = "#5DADE2"   
    C_DEL = "#52BE80"   
    C_INS = "#BC70DB"
    err_types = [
        ("sub", "Substitutions", C_SUB),
        ("del", "Deletions",     C_DEL),
        ("ins", "Insertions",    C_INS),
    ]
    versions_present = [v for v in VERSIONS if v in df[COL_VERSION].unique()]

    # ── 1. Token-Fehler extrahieren ──────────────────────────────────────────
    error_data = {}   # error_data[v][etype] = Counter
    for v in versions_present:
        sub_c, del_c, ins_c = Counter(), Counter(), Counter()
        df_v = df[df[COL_VERSION] == v]
        print(f"  Extrahiere {v} ({len(df_v):,} rows) ...", end=" ", flush=True)
        for _, row in df_v.iterrows():
            ref = safe_str(row[COL_REFERENCE])
            hyp = safe_str(row[COL_HYPOTHESIS])
            if not ref or not hyp:
                continue
            try:
                out = process_words([ref], [hyp])
            except Exception:
                continue
            rw_all = ref.split()
            hw_all = hyp.split()
            for chunk in out.alignments[0]:
                rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]
                if chunk.type == "substitute":
                    for r, h in zip(rw, hw):
                        sub_c[f"{r} → {h}"] += 1
                elif chunk.type == "delete":
                    for w in rw:
                        del_c[w] += 1
                elif chunk.type == "insert":
                    for w in hw:
                        ins_c[w] += 1
        error_data[v] = {"sub": sub_c, "del": del_c, "ins": ins_c}
        print("done")

    # ── 2. Top-N bestimmen (Summe aller Modelle) ─────────────────────────────
    def top_tokens(etype, n):
        """Gibt die Top-N Token/Paare nach Summe über alle Modelle zurück."""
        total = Counter()
        for v in versions_present:
            total += error_data[v][etype]
        return [token for token, _ in total.most_common(n)]

    top_sub = top_tokens("sub", top_n)
    top_del = top_tokens("del", top_n)
    top_ins = top_tokens("ins", top_n)

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(7 * 3, 0.55 * top_n * 3 + 2),   # Höhe skaliert mit top_n
        constrained_layout=True
    )

    bar_h     = 0.25                          # Höhe eines einzelnen Balkens
    offsets   = [-bar_h, 0, bar_h]           # v1 oben, v2 mitte, v3 unten

    for ax, (etype, etype_label, bar_color), top_tokens_list in zip(
        axes, err_types, [top_sub, top_del, top_ins]
    ):
        # Y-Positionen (0 = unten im Plot → invert_yaxis dreht es um)
        y_pos = np.arange(len(top_tokens_list), dtype=float)

        max_count = 0
        for v_i, v in enumerate(versions_present):
            counts = [error_data[v][etype].get(tok, 0) for tok in top_tokens_list]
            max_count = max(max_count, max(counts) if counts else 1)

            # Helligkeit abstufen: v1 dunkel → v3 hell
            alpha = 1.0 - v_i * 0.22

            bars = ax.barh(
                y_pos + offsets[v_i], counts,
                height=bar_h,
                color=bar_color,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.4,
                label=V_SHORT[v]
            )

            # Count + % Label
            type_total = sum(error_data[v][etype].values())
            for bar, count in zip(bars, counts):
                if count == 0:
                    continue
                pct = count / type_total * 100 if type_total > 0 else 0
                ax.text(
                    bar.get_width() + max_count * 0.005,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,} ({pct:.1f}%)",
                    va="center", ha="left",
                    fontsize=7.5, color="#444444"
                )

        # Achsen
        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_tokens_list, fontsize=9.5)
        ax.invert_yaxis()                    # Rang 1 oben
        ax.set_xlabel("Count", fontsize=10)
        ax.set_xlim(0, max_count * 1.42)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.25, linewidth=0.6)

        # Titel mit Farbe
        ax.set_title(
            f"Top {top_n} {etype_label}",
            fontsize=13, fontweight="bold", color=bar_color, pad=10
        )

        ax.legend(
                title="Modell", fontsize=9, title_fontsize=9,
                loc="lower right", framealpha=0.7 
                )

        """
        # Legende nur im ersten Subplot
        if ax == axes[0]:
            ax.legend(
                title="Modell", fontsize=9, title_fontsize=9,
                loc="lower right", framealpha=0.7
            )
        """    

    fig.suptitle(
        f"Top {top_n} Error Tokens by Type",
        #f"Ranking nach Summe aller 3 Modelle  ·  Helligkeit: v1 (dunkel) → v3 (hell)",
        fontsize=13, fontweight="bold", color="#1B3A6B", y=1.02
    )

    out = "figures/fig_error_top10_grouped.png"
    plt.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"→ {out}")





def analyze_semantic_substitutions_fast(df, max_examples=10):
    """
    Schnelle Version ohne ML – nur String-Ähnlichkeit
    """

    total = 0
    similar = 0
    examples = []

    for _, row in df.iterrows():
        ref = safe_str(row[COL_REFERENCE])
        hyp = safe_str(row[COL_HYPOTHESIS])

        if not ref or not hyp:
            continue

        try:
            out = process_words([ref], [hyp])
        except:
            continue

        rw_all = ref.split()
        hw_all = hyp.split()

        for chunk in out.alignments[0]:
            if chunk.type != "substitute":
                continue

            rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
            hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]

            for r, h in zip(rw, hw):
                total += 1

                # 👉 nur einfache Ähnlichkeit
                if Levenshtein.distance(r, h) <= 2:
                    similar += 1

                    if len(examples) < max_examples:
                        examples.append((r, h))

    pct = similar / total * 100 if total > 0 else 0

    print(f"\nSimilar substitutions: {similar}/{total} ({pct:.2f}%)")

    print("\nExamples:")
    for r, h in examples:
        print(f"{r} → {h}")

    return pct, examples




def get_error_examples(df, n=10):
    examples = []

    for _, row in df.iterrows():
        ref = safe_str(row[COL_REFERENCE])
        hyp = safe_str(row[COL_HYPOTHESIS])

        if not ref or not hyp:
            continue

        # nur echte Fehler
        if ref != hyp:
            examples.append((ref, hyp))

        if len(examples) >= n:
            break

    return examples




def fig_error_top3_grouped_poster(df):
    """
    Poster-Version:
    - nur Top 3 pro Typ
    - keine Prozentlabels
    - größere Schrift
    - nur eine Legende
    - kompakter Titel
    Speichert → figures/fig_error_top3_grouped_poster.png
    """
    C_SUB = "#5DADE2"
    C_DEL = "#52BE80"
    C_INS = "#BC70DB"

    err_types = [
        ("sub", "Substitutions", C_SUB),
        ("del", "Deletions",     C_DEL),
        ("ins", "Insertions",    C_INS),
    ]

    versions_present = [v for v in VERSIONS if v in df[COL_VERSION].unique()]
    top_n = 3

    # ── 1. Token-Fehler extrahieren ──────────────────────────────────────────
    error_data = {}
    for v in versions_present:
        sub_c, del_c, ins_c = Counter(), Counter(), Counter()
        df_v = df[df[COL_VERSION] == v]
        print(f"  Extrahiere {v} ({len(df_v):,} rows) ...", end=" ", flush=True)

        for _, row in df_v.iterrows():
            ref = safe_str(row[COL_REFERENCE])
            hyp = safe_str(row[COL_HYPOTHESIS])
            if not ref or not hyp:
                continue

            try:
                out = process_words([ref], [hyp])
            except Exception:
                continue

            rw_all = ref.split()
            hw_all = hyp.split()

            for chunk in out.alignments[0]:
                rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == "substitute":
                    for r, h in zip(rw, hw):
                        sub_c[f"{r} → {h}"] += 1
                elif chunk.type == "delete":
                    for w in rw:
                        del_c[w] += 1
                elif chunk.type == "insert":
                    for w in hw:
                        ins_c[w] += 1

        error_data[v] = {"sub": sub_c, "del": del_c, "ins": ins_c}
        print("done")

    # ── 2. Top 3 nach Summe aller Modelle ───────────────────────────────────
    def top_tokens(etype, n):
        total = Counter()
        for v in versions_present:
            total += error_data[v][etype]
        return [token for token, _ in total.most_common(n)]

    top_sub = top_tokens("sub", top_n)
    top_del = top_tokens("del", top_n)
    top_ins = top_tokens("ins", top_n)

    # ── 3. Plot ───────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(
        nrows=1, ncols=3,
        figsize=(18, 6.5),
        constrained_layout=True
    )

    bar_h = 0.22
    offsets = [-bar_h, 0, bar_h]

    for ax, (etype, etype_label, bar_color), top_tokens_list in zip(
        axes, err_types, [top_sub, top_del, top_ins]
    ):
        y_pos = np.arange(len(top_tokens_list), dtype=float)
        max_count = 0

        for v_i, v in enumerate(versions_present):
            counts = [error_data[v][etype].get(tok, 0) for tok in top_tokens_list]
            max_count = max(max_count, max(counts) if counts else 1)

            alpha = 1.0 - v_i * 0.18  # etwas weniger extrem

            bars = ax.barh(
                y_pos + offsets[v_i],
                counts,
                height=bar_h,
                color=bar_color,
                alpha=alpha,
                edgecolor="white",
                linewidth=0.5,
                label=V_SHORT[v]
            )

            # nur Count, keine Prozentangabe
            for bar, count in zip(bars, counts):
                if count == 0:
                    continue
                ax.text(
                    bar.get_width() + max_count * 0.015,
                    bar.get_y() + bar.get_height() / 2,
                    f"{count:,}",
                    va="center", ha="left",
                    fontsize=10, color="#444444"
                )

        ax.set_yticks(y_pos)
        ax.set_yticklabels(top_tokens_list, fontsize=12)
        ax.invert_yaxis()
        ax.set_xlabel("Count", fontsize=11)
        ax.set_xlim(0, max_count * 1.28)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)
        ax.grid(axis="x", linestyle="--", alpha=0.2, linewidth=0.6)

        ax.set_title(
            etype_label,
            fontsize=15, fontweight="bold", color=bar_color, pad=10
        )

    # nur eine Legende, links
    axes[0].legend(
        title="Model",
        fontsize=10,
        title_fontsize=10,
        loc="lower right",
        framealpha=0.75
    )

    fig.suptitle(
        "Top Error Tokens by Type",
        fontsize=16, fontweight="bold", color="#1B3A6B", y=1.02
    )

    out = "figures/fig_error_top3_grouped_poster.png"
    plt.savefig(out, dpi=220, bbox_inches="tight", facecolor="white")
    plt.show()
    print(f"→ {out}")


def shorten_token_label(token, etype=None):
    if etype == "sub":
        return token.replace(" → ", "→").replace(" ", "")
    return token


def fig_error_top3_vertical(df):
    """
    Poster-Version (VERTIKAL):
    - Top 3 pro Typ
    - untereinander statt nebeneinander
    - gut für schmale Bereiche im Poster
    """

    C_SUB = "#5DADE2"
    C_DEL = "#52BE80"
    C_INS = "#BC70DB"

    err_types = [
        ("sub", "Substitutions", C_SUB),
        ("del", "Deletions",     C_DEL),
        ("ins", "Insertions",    C_INS),
    ]

    versions_present = [v for v in VERSIONS if v in df[COL_VERSION].unique()]
    top_n = 3

    # ── Daten sammeln ─────────────────────────
    error_data = {}
    for v in versions_present:
        sub_c, del_c, ins_c = Counter(), Counter(), Counter()
        df_v = df[df[COL_VERSION] == v]

        for _, row in df_v.iterrows():
            ref = safe_str(row[COL_REFERENCE])
            hyp = safe_str(row[COL_HYPOTHESIS])
            if not ref or not hyp:
                continue

            try:
                out = process_words([ref], [hyp])
            except:
                continue

            rw_all = ref.split()
            hw_all = hyp.split()

            for chunk in out.alignments[0]:
                rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == "substitute":
                    for r, h in zip(rw, hw):
                        sub_c[f"{r} → {h}"] += 1
                elif chunk.type == "delete":
                    for w in rw:
                        del_c[w] += 1
                elif chunk.type == "insert":
                    for w in hw:
                        ins_c[w] += 1

        error_data[v] = {"sub": sub_c, "del": del_c, "ins": ins_c}

    def top_tokens(etype):
        total = Counter()
        for v in versions_present:
            total += error_data[v][etype]
        return [t for t, _ in total.most_common(top_n)]

    top_sub = top_tokens("sub")
    top_del = top_tokens("del")
    top_ins = top_tokens("ins")

    # ── Plot ─────────────────────────
    fig, axes = plt.subplots(
        nrows=3, ncols=1,
        figsize=(6, 10),   # 👈 HOCH statt breit!
        constrained_layout=True
    )

    bar_h = 0.22
    offsets = [-bar_h, 0, bar_h]

    for ax, (etype, label, color), tokens in zip(
        axes, err_types, [top_sub, top_del, top_ins]
    ):
        y = np.arange(len(tokens))
        max_count = 0

        for i, v in enumerate(versions_present):
            counts = [error_data[v][etype].get(t, 0) for t in tokens]
            max_count = max(max_count, max(counts))

            bars = ax.barh(
                y + offsets[i],
                counts,
                height=bar_h,
                color=color,
                alpha=1 - i * 0.18,
                label=V_SHORT[v]
            )

            for bar, c in zip(bars, counts):
                if c > 0:
                    ax.text(
                        bar.get_width() + max_count * 0.02,
                        bar.get_y() + bar.get_height()/2,
                        f"{c}",
                        va="center",
                        fontsize=10
                    )

        ax.set_yticks(y)
        ax.set_yticklabels(tokens, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, max_count * 1.3)
        ax.set_title(label, fontsize=14, fontweight="bold", color=color)

        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].legend(title="Model", fontsize=9)

    fig.suptitle("Top Error Tokens by Type", fontsize=15, fontweight="bold")

    plt.savefig("figures/fig_error_top3_vertical.png", dpi=220, bbox_inches="tight")
    plt.show()


def fig_error_top3_focus_sub(df):
    """
    Poster-Version:
    - Substitutions gross oben
    - Deletions & Insertions unten nebeneinander
    - Farben = Modelle
    - kompaktere Labels bei Substitutions
    """

    from collections import Counter
    import numpy as np
    import matplotlib.pyplot as plt

    def shorten_token_label(token, etype=None):
        if etype == "sub":
            return token.replace(" → ", "→").replace(" ", "")
        return token

    # Farben pro Modellname / Kurzname
    MODEL_COLORS = {
        "v1": "#5DADE2",
        "v2": "#52BE80",
        "v3": "#BC70DB",
        "large-v1": "#5DADE2",
        "large-v2": "#52BE80",
        "large-v3": "#BC70DB",
    }

    versions_present = [v for v in VERSIONS if v in df[COL_VERSION].unique()]
    top_n = 3

    # ── Daten sammeln ─────────────────────────
    error_data = {}
    for v in versions_present:
        sub_c, del_c, ins_c = Counter(), Counter(), Counter()
        df_v = df[df[COL_VERSION] == v]

        for _, row in df_v.iterrows():
            ref = safe_str(row[COL_REFERENCE])
            hyp = safe_str(row[COL_HYPOTHESIS])
            if not ref or not hyp:
                continue

            try:
                out = process_words([ref], [hyp])
            except Exception:
                continue

            rw_all = ref.split()
            hw_all = hyp.split()

            for chunk in out.alignments[0]:
                rw = rw_all[chunk.ref_start_idx:chunk.ref_end_idx]
                hw = hw_all[chunk.hyp_start_idx:chunk.hyp_end_idx]

                if chunk.type == "substitute":
                    for r, h in zip(rw, hw):
                        sub_c[f"{r} → {h}"] += 1
                elif chunk.type == "delete":
                    for w in rw:
                        del_c[w] += 1
                elif chunk.type == "insert":
                    for w in hw:
                        ins_c[w] += 1

        error_data[v] = {"sub": sub_c, "del": del_c, "ins": ins_c}

    def top_tokens(etype):
        total = Counter()
        for v in versions_present:
            total += error_data[v][etype]
        return [t for t, _ in total.most_common(top_n)]

    top_sub = top_tokens("sub")
    top_del = top_tokens("del")
    top_ins = top_tokens("ins")

    # ── Layout ─────────────────────────
    fig = plt.figure(figsize=(8, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    ax_sub = fig.add_subplot(gs[0, :])   # oben full width
    ax_del = fig.add_subplot(gs[1, 0])
    ax_ins = fig.add_subplot(gs[1, 1])

    axes = [
        (ax_sub, "sub", "Substitutions", top_sub),
        (ax_del, "del", "Deletions",     top_del),
        (ax_ins, "ins", "Insertions",    top_ins),
    ]

    bar_h = 0.25
    offsets = [-bar_h, 0, bar_h]

    for ax, etype, label, tokens in axes:
        y = np.arange(len(tokens))
        max_count = 0

        for i, v in enumerate(versions_present):
            counts = [error_data[v][etype].get(t, 0) for t in tokens]
            max_count = max(max_count, max(counts) if counts else 0)

            # Kurzname nur für Legende
            label_name = V_SHORT[v] if v in V_SHORT else str(v)

            # Farbe robust bestimmen
            color = MODEL_COLORS.get(label_name, MODEL_COLORS.get(str(v), "#888888"))

            bars = ax.barh(
                y + offsets[i],
                counts,
                height=bar_h,
                color=color,
                alpha=1,
                label=label_name
            )

            for bar, c in zip(bars, counts):
                if c > 0:
                    ax.text(
                        bar.get_width() + max_count * 0.02,
                        bar.get_y() + bar.get_height() / 2,
                        f"{c}",
                        va="center",
                        fontsize=10
                    )

        short_tokens = [shorten_token_label(t, etype) for t in tokens]
        ax.set_yticks(y)
        ax.set_yticklabels(short_tokens, fontsize=11)
        ax.invert_yaxis()
        ax.set_xlim(0, max_count * 1.25 if max_count > 0 else 1)
        ax.set_title(label, fontsize=13, fontweight="bold")

        ax.grid(axis="x", linestyle="--", alpha=0.2)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Legende nur oben
    ax_sub.legend(title="Model", fontsize=9)

    fig.suptitle("Top Error Tokens by Type", fontsize=14, fontweight="bold")

    plt.savefig("figures/fig_error_top3_focus_sub.png", dpi=220, bbox_inches="tight")
    plt.show()