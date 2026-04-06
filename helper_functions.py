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
    x    = np.arange(len(VERSIONS))
    subs = [wer_overall.loc[v, 'substitutions'] / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    dels = [wer_overall.loc[v, 'deletions']     / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    ins  = [wer_overall.loc[v, 'insertions']    / wer_overall.loc[v, 'total_errors'] * 100 for v in VERSIONS]
    ax.bar(x, subs, label='Substitutionen', color='#E53935', edgecolor='white')
    ax.bar(x, dels, bottom=subs, label='Deletions', color='#FB8C00', edgecolor='white')
    ax.bar(x, ins,  bottom=[s+d for s,d in zip(subs,dels)], label='Insertions', color='#43A047', edgecolor='white')
    for i in range(len(VERSIONS)):
        if subs[i] > 4: ax.text(i, subs[i]/2,                f'{subs[i]:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if dels[i] > 4: ax.text(i, subs[i]+dels[i]/2,        f'{dels[i]:.1f}%', ha='center', va='center', color='white', fontsize=9, fontweight='bold')
        if ins[i]  > 4: ax.text(i, subs[i]+dels[i]+ins[i]/2, f'{ins[i]:.1f}%',  ha='center', va='center', color='white', fontsize=9, fontweight='bold')
    ax.set_ylabel('Anteil an Gesamtfehlern (%)')
    ax.set_title('Fehlertypen-Verteilung – Whisper v1 / v2 / v3')
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
    ax.set_ylabel('Fehlerrate (%)')
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
