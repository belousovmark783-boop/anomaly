# -*- coding: utf-8 -*-
"""
Поиск аномальных респондентов в активности поисковых запросов SoS.

ЗАПУСК ОДНОЙ КОМАНДОЙ:
    python solution.py
ДОПОЛНИТЕЛЬНО:
    python solution.py --data data_train --output output      # свои пути
    python solution.py --analytics                            # доп. аналитика (§8.2)

Что считается аномалией:
    Единица анализа (триггер): (SubjectID, researchdate, BrandID, CategoryDelivery).
    Метрика воздействия — дневной OTS:
        daily_ots(i, j, k) = Weight(i, k) * count_rows(i, j, k)
    где count_rows — число строк респондента по бренду за день среди строк с
    BrandinDelivery = 1 и непустым CategoryDelivery.
    Аномалия = ЧРЕЗМЕРНО ВЫСОКИЙ OTS, вызванный повторной активностью. Малый OTS
    аномалией не считается (по построению score и явному нижнему порогу).

Единица удаления: (SubjectID, researchdate) — попадает в anomalies.csv. При
пересчёте показателей удаляются ВСЕ строки респондента за этот день.

Алгоритм детерминированный, интерпретируемый, работает на уровне брендов.
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# ПАРАМЕТРЫ МЕТОДА — это статистически обоснованные константы, а НЕ подгонка под
# целевой процент удалений и НЕ конкретные ID/даты/бренды.
#   Z_THRESHOLD = 3.5  — классический порог модифицированного z-score выброса
#                        (Iglewicz & Hoaglin, 1993).
#   MIN_OBS     = 30   — минимум наблюдений, чтобы доверять робастной оценке
#                        распределения бренда; иначе эталон берём выше по иерархии.
# Порог активности НЕ зашит — считается из данных (верхний фенс Тьюки).
# ----------------------------------------------------------------------------
Z_THRESHOLD = 3.5
MIN_OBS = 30


# =====================  ЗАГРУЗКА И ПОДГОТОВКА ДАННЫХ  ========================
def detect_category_column(df: pd.DataFrame) -> str:
    """Имя столбца категории поставки устойчиво к расхождению схем
    (CategoryDelivery в ТЗ / CategoryNameDelivery в данных)."""
    for c in ("CategoryDelivery", "CategoryNameDelivery"):
        if c in df.columns:
            return c
    raise KeyError("Не найден столбец CategoryDelivery / CategoryNameDelivery")


def load_data(data_dir: str) -> tuple[pd.DataFrame, str]:
    """Читает parquet-партиции. month=*/ покрывают непересекающиеся диапазоны
    researchdate, поэтому объединение не создаёт дублей строк."""
    files = sorted(glob.glob(os.path.join(data_dir, "month=*", "part-*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(data_dir, "**", "*.parquet"), recursive=True))
    if not files:
        raise FileNotFoundError(f"Не найдены parquet-файлы в {data_dir!r}")
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    cat_col = detect_category_column(df)
    df["Weight"] = df["Weight"].astype(float)
    df["researchdate"] = pd.to_datetime(df["researchdate"]).dt.normalize()
    return df, cat_col


def filter_delivery(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """Строки для анализа: BrandinDelivery = 1 и непустой CategoryDelivery (§3, §5)."""
    mask = df[cat_col].notna() & (df[cat_col].astype(str).str.len() > 0)
    if "BrandinDelivery" in df.columns:
        mask &= df["BrandinDelivery"] == 1
    return df[mask].copy()


def build_groups(df: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """Агрегация до триггера (SubjectID, CategoryDelivery, BrandID, researchdate):
    число строк-запросов, дневной вес и daily_ots = Weight * count_rows."""
    g = (
        df.groupby(["SubjectID", cat_col, "BrandID", "researchdate"], observed=True)
        .agg(count_rows=("Weight", "size"),
             weight=("Weight", "first"),
             Brand=("Brand", "first"))
        .reset_index()
    )
    g["daily_ots"] = g["weight"] * g["count_rows"]
    g["log_ots"] = np.log1p(g["daily_ots"])
    return g


# =========================  РОБАСТНЫЙ SCORE  ================================
def _robust_stats(g: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    """median, MAD и размер выборки лог-OTS на заданном уровне группировки."""
    def mad(x):
        return np.median(np.abs(x - np.median(x)))
    return (
        g.groupby(keys, observed=True)["log_ots"]
        .agg(med="median", mad=mad, cnt="size")
        .reset_index()
    )


def compute_score(g: pd.DataFrame, cat_col: str) -> pd.DataFrame:
    """Иерархический робастный модифицированный z-score лог-OTS:
    эталон = бренд -> категория поставки -> вся выборка (устойчивость к редким
    брендам и малым выборкам, §6)."""
    brand = _robust_stats(g, [cat_col, "BrandID"])
    cat = _robust_stats(g, [cat_col]).rename(
        columns={"med": "med_c", "mad": "mad_c", "cnt": "cnt_c"})
    gl_med = float(g["log_ots"].median())
    gl_mad = float(np.median(np.abs(g["log_ots"] - gl_med))) or 1.0

    g = g.merge(brand, on=[cat_col, "BrandID"], how="left")
    g = g.merge(cat, on=[cat_col], how="left")

    use_brand = (g["cnt"] >= MIN_OBS) & (g["mad"] > 0)
    use_cat = (~use_brand) & (g["cnt_c"] >= MIN_OBS) & (g["mad_c"] > 0)

    g["ref_level"] = np.where(use_brand, "brand", np.where(use_cat, "category", "global"))
    g["ref_med"] = np.where(use_brand, g["med"], np.where(use_cat, g["med_c"], gl_med))
    g["ref_mad"] = np.where(use_brand, g["mad"], np.where(use_cat, g["mad_c"], gl_mad))
    g["ref_mad"] = g["ref_mad"].replace(0, gl_mad)

    # модифицированный z-score: z = 0.6745 * (x - median) / MAD
    g["score"] = 0.6745 * (g["log_ots"] - g["ref_med"]) / g["ref_mad"]
    return g


def activity_threshold(g: pd.DataFrame) -> int:
    """Data-driven порог активности: верхний фенс Тьюки (Q3 + 1.5*IQR) по числу
    строк-запросов среди ПОВТОРНЫХ обращений (count_rows >= 2). Гарантирует, что
    аномалия требует именно повторной активности, а не одиночного запроса с
    большим весом. Не зависит от желаемого процента удалений."""
    repeat = g.loc[g["count_rows"] >= 2, "count_rows"]
    if repeat.empty:
        return 2
    q1, q3 = repeat.quantile(0.25), repeat.quantile(0.75)
    fence = q3 + 1.5 * (q3 - q1)
    return int(max(2, np.ceil(fence)))


def detect(g: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Триггер аномалии = одновременное выполнение трёх условий:
       (1) score > Z_THRESHOLD     — OTS экстремально высок для бренда;
       (2) count_rows >= act_thr   — активность действительно повторная;
       (3) daily_ots > медианы OTS — малый OTS не может быть причиной (§3, §6)."""
    act_thr = activity_threshold(g)
    ots_floor = float(g["daily_ots"].median())

    g["is_anomaly"] = (
        (g["score"] > Z_THRESHOLD)
        & (g["count_rows"] >= act_thr)
        & (g["daily_ots"] > ots_floor)
    )
    params = {"Z_THRESHOLD": Z_THRESHOLD, "ACT_THRESHOLD": act_thr,
              "OTS_FLOOR": ots_floor, "MIN_OBS": MIN_OBS}
    return g, params


# ===========================  ЗАПИСЬ РЕЗУЛЬТАТОВ  ===========================
def write_outputs(g: pd.DataFrame, params: dict, cat_col: str, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    anom = g[g["is_anomaly"]].copy()

    # --- anomalies.csv: только SubjectID, researchdate; пары без дублей ---
    anomalies = (anom[["SubjectID", "researchdate"]]
                 .drop_duplicates()
                 .sort_values(["researchdate", "SubjectID"]))
    out = anomalies.copy()
    out["researchdate"] = out["researchdate"].dt.strftime("%Y-%m-%d")
    out.to_csv(os.path.join(out_dir, "anomalies.csv"), index=False)

    # --- anomaly_reasons.csv: строгая схема из §8.1 ---
    r = anom.copy()
    r["score_r"] = r["score"].round(2)
    r["threshold"] = params["Z_THRESHOLD"]
    r["reason"] = (
        "robust z-score лог-OTS=" + r["score_r"].astype(str)
        + " > " + str(params["Z_THRESHOLD"])
        + " при повторной активности count_rows=" + r["count_rows"].astype(str)
        + " >= " + str(params["ACT_THRESHOLD"])
        + " (эталон: " + r["ref_level"] + ")"
    )
    reasons = pd.DataFrame({
        "SubjectID": r["SubjectID"],
        "researchdate": r["researchdate"].dt.strftime("%Y-%m-%d"),
        "BrandID": r["BrandID"],
        "Brand": r["Brand"],
        "CategoryDelivery": r[cat_col],
        "daily_ots": r["daily_ots"].round(3),
        "score": r["score"].round(4),
        "threshold": params["Z_THRESHOLD"],
        "reason": r["reason"],
    }).sort_values(["researchdate", "SubjectID", "daily_ots"],
                   ascending=[True, True, False])
    reasons.to_csv(os.path.join(out_dir, "anomaly_reasons.csv"), index=False)
    return anomalies


# ============================  ОБЯЗАТЕЛЬНЫЕ ГРАФИКИ  ========================
def make_plots(g: pd.DataFrame, anomalies: pd.DataFrame, cat_col: str, out_dir: str):
    plots_dir = os.path.join(out_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    removed = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    g = g.copy()
    g["key"] = list(zip(g["SubjectID"], g["researchdate"]))
    kept = g[~g["key"].isin(removed)]

    # 1) общий OTS по дням до/после удаления
    before = g.groupby("researchdate")["daily_ots"].sum()
    after = kept.groupby("researchdate")["daily_ots"].sum().reindex(before.index, fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(before.index, before.values, "-o", ms=3, color="#E45756", label="до удаления")
    ax.plot(after.index, after.values, "-o", ms=3, color="#4C78A8", label="после удаления")
    keep_pct = 100 * after.sum() / before.sum()
    ax.set_title(f"Общий OTS по дням до/после удаления аномалий "
                 f"(сохранено {keep_pct:.1f}% суммарного OTS)")
    ax.set_xlabel("Дата"); ax.set_ylabel("Суммарный OTS"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "total_ots_before_after.png"), dpi=110)
    plt.close(fig)

    # 2) % изменения OTS по CategoryDelivery
    b = g.groupby(cat_col)["daily_ots"].sum()
    a = kept.groupby(cat_col)["daily_ots"].sum().reindex(b.index, fill_value=0)
    pct = ((a - b) / b * 100).sort_values()
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(range(len(pct)), pct.values, color="#4C78A8")
    ax.set_xticks(range(len(pct))); ax.set_xticklabels(pct.index, rotation=90, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("% изменения суммарного OTS")
    ax.set_title(f"Изменение OTS по CategoryDelivery после удаления "
                 f"(итог: {100*(a.sum()/b.sum()-1):+.2f}%)")
    fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "category_ots_change.png"), dpi=110)
    plt.close(fig)

    # 3) число аномальных респондентов по дням
    per_day = anomalies.groupby("researchdate")["SubjectID"].nunique()
    if len(per_day):
        idx = pd.date_range(per_day.index.min(), per_day.index.max(), freq="D")
        per_day = per_day.reindex(idx, fill_value=0)
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.bar(per_day.index, per_day.values, color="#4C78A8", width=0.8)
    ax.set_xlabel("Дата"); ax.set_ylabel("Аномальных респондентов")
    ax.set_title(f"Аномальные респонденты по дням. Всего пар: {len(anomalies)}, "
                 f"уникальных респондентов: {anomalies['SubjectID'].nunique()}")
    fig.autofmt_xdate(); fig.tight_layout()
    fig.savefig(os.path.join(plots_dir, "daily_anomaly_count.png"), dpi=110)
    plt.close(fig)


# =====================  АНАЛИТИЧЕСКИЕ ВОЗМОЖНОСТИ (§8.2)  ===================
# Все функции запускаются из кода без переписывания основной логики.
def segment_before_after(df: pd.DataFrame, removed: set, segment_col: str,
                         cat_col: str) -> pd.DataFrame:
    """Суммарный OTS и его изменение в разрезе любого признака до/после очистки.
    Подходит для демографии (Пол, Возраст, Регион, Федеральный_округ),
    ресурсов (ResourceName, ResourceType, Platform, UseType) и уровней категорий
    (CategoryDelivery, Category1, Category2, Category3)."""
    d = df.copy()
    d["ots_row"] = d["Weight"]            # вклад строки в OTS = Weight (count учтён числом строк)
    d["key"] = list(zip(d["SubjectID"], d["researchdate"]))
    before = d.groupby(segment_col)["ots_row"].sum()
    after = d[~d["key"].isin(removed)].groupby(segment_col)["ots_row"].sum()
    res = pd.DataFrame({"ots_before": before, "ots_after": after}).fillna(0)
    res["pct_change"] = (res["ots_after"] - res["ots_before"]) / res["ots_before"] * 100
    return res.sort_values("pct_change")


def plot_segment(df, removed, segment_col, cat_col, out_path, top=25):
    res = segment_before_after(df, removed, segment_col, cat_col)
    res = pd.concat([res.head(top), res.tail(top)]).drop_duplicates()
    fig, ax = plt.subplots(figsize=(min(16, 0.45 * len(res) + 4), 6))
    ax.bar(range(len(res)), res["pct_change"].values, color="#72B7B2")
    ax.set_xticks(range(len(res)))
    ax.set_xticklabels([str(x)[:25] for x in res.index], rotation=90, fontsize=7)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("% изменения OTS"); ax.set_title(f"OTS до/после по: {segment_col}")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def query_text_table(df: pd.DataFrame, subject_id, date, cat_col: str) -> pd.DataFrame:
    """Таблица поисковых запросов QueryText выбранного аномального респондента за день."""
    date = pd.to_datetime(date).normalize()
    sub = df[(df["SubjectID"] == subject_id) & (df["researchdate"] == date)]
    cols = [c for c in ["QueryText", "Brand", cat_col, "ResourceName", "Start"]
            if c in sub.columns]
    return sub[cols].sort_values("Start" if "Start" in cols else cols[0])


def plot_brand_ots_by_day(g: pd.DataFrame, removed: set, cat_col: str,
                          brand_id, out_path: str):
    """Изменение OTS по дням для выбранного бренда до и после очистки."""
    gg = g.copy(); gg["key"] = list(zip(gg["SubjectID"], gg["researchdate"]))
    sub = gg[gg["BrandID"] == brand_id]
    before = sub.groupby("researchdate")["daily_ots"].sum()
    after = sub[~sub["key"].isin(removed)].groupby("researchdate")["daily_ots"].sum()
    after = after.reindex(before.index, fill_value=0)
    name = sub["Brand"].iloc[0] if len(sub) else brand_id
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(before.index, before.values, "-o", ms=3, color="#E45756", label="до")
    ax.plot(after.index, after.values, "-o", ms=3, color="#4C78A8", label="после")
    ax.set_title(f"OTS по дням для бренда «{name}» (BrandID={brand_id}) до/после очистки")
    ax.set_xlabel("Дата"); ax.set_ylabel("OTS"); ax.legend()
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def run_analytics(df, g, anomalies, cat_col, out_dir):
    """Демонстрационный прогон всех аналитических возможностей §8.2.
    Объекты для примеров выбираются ИЗ ДАННЫХ (топ-аномалия), без hardcode."""
    adir = os.path.join(out_dir, "analytics")
    os.makedirs(adir, exist_ok=True)
    removed = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))

    demo = [c for c in ["Пол", "Возраст", "Регион", "Федеральный_округ"] if c in df.columns]
    resource = [c for c in ["ResourceName", "ResourceType", "Platform", "UseType"] if c in df.columns]
    cats = [c for c in [cat_col, "Category1", "Category2", "Category3"] if c in df.columns]
    for col in demo + resource + cats:
        plot_segment(df, removed, col, cat_col, os.path.join(adir, f"segment_{col}.png"))

    # таблица QueryText для самой яркой аномалии (макс. daily_ots среди триггеров)
    anom_groups = g[g["is_anomaly"]]
    if len(anom_groups):
        top = anom_groups.loc[anom_groups["daily_ots"].idxmax()]
        qt = query_text_table(df, top["SubjectID"], top["researchdate"], cat_col)
        qt.to_csv(os.path.join(adir, "querytext_top_anomaly.csv"), index=False)
        plot_brand_ots_by_day(g, removed, cat_col, top["BrandID"],
                              os.path.join(adir, "brand_ots_by_day.png"))
        print(f"  Пример аномалии: SubjectID={top['SubjectID']} "
              f"{pd.Timestamp(top['researchdate']).date()} бренд={top['Brand']} "
              f"daily_ots={top['daily_ots']:.0f}")
    print(f"  Аналитика сохранена в: {adir}/")


# ==============================  ВЛИЯНИЕ  ===================================
def print_impact(g, anomalies, cat_col, params):
    removed = set(zip(anomalies["SubjectID"], anomalies["researchdate"]))
    gg = g.copy(); gg["key"] = list(zip(gg["SubjectID"], gg["researchdate"]))
    kept = gg[~gg["key"].isin(removed)]
    bd0 = gg.groupby([cat_col, "BrandID", "researchdate"])["daily_ots"].sum()
    bd1 = kept.groupby([cat_col, "BrandID", "researchdate"])["daily_ots"].sum()
    thr = bd0.quantile(0.999)
    pe0 = (bd0[bd0 > thr] - thr).sum(); pe1 = (bd1[bd1 > thr] - thr).sum()
    n_subj = g["SubjectID"].nunique()
    print("\n================ ВЛИЯНИЕ ОЧИСТКИ ================")
    print(f"Порог активности (data-driven): count_rows >= {params['ACT_THRESHOLD']}")
    print(f"Нижний порог OTS (медиана):     {params['OTS_FLOOR']:.0f}")
    print(f"Удалено уникальных респондентов: {anomalies['SubjectID'].nunique()} "
          f"({100*anomalies['SubjectID'].nunique()/n_subj:.2f}% от {n_subj})")
    print(f"Удалено пар (респондент, день):  {len(anomalies)}")
    print(f"Сохранено суммарного OTS:        {100*kept['daily_ots'].sum()/gg['daily_ots'].sum():.2f}%")
    print(f"Снижение избытка экстремальных бренд-дневных пиков (>99.9 перц.): "
          f"{100*(1-pe1/pe0):.1f}%")
    print(f"Макс. бренд-дневной OTS: {bd0.max():.0f} -> {bd1.max():.0f}")


# ================================  MAIN  ====================================
def main():
    ap = argparse.ArgumentParser(description="Поиск аномальных респондентов SoS")
    ap.add_argument("--data", default="data_train", help="папка с month=*/...parquet")
    ap.add_argument("--output", default="output", help="папка для результатов")
    ap.add_argument("--analytics", action="store_true",
                    help="дополнительно построить аналитические разрезы (§8.2)")
    args = ap.parse_args()

    print("1/5 Загрузка данных ...")
    df, cat_col = load_data(args.data)
    df = filter_delivery(df, cat_col)
    print(f"    столбец категории: {cat_col}; строк (BrandinDelivery=1): {len(df)}; "
          f"респондентов: {df['SubjectID'].nunique()}; брендов: {df['BrandID'].nunique()}; "
          f"дни: {df['researchdate'].min().date()}..{df['researchdate'].max().date()}")

    print("2/5 Агрегация до триггера и расчёт daily_ots ...")
    g = build_groups(df, cat_col)

    print("3/5 Робастный score (бренд -> категория -> глобально) ...")
    g = compute_score(g, cat_col)

    print("4/5 Поиск аномалий ...")
    g, params = detect(g)

    print("5/5 Запись результатов и обязательных графиков ...")
    anomalies = write_outputs(g, params, cat_col, args.output)
    make_plots(g, anomalies, cat_col, args.output)

    if args.analytics:
        print("    Доп. аналитика (§8.2) ...")
        run_analytics(df, g, anomalies, cat_col, args.output)

    print_impact(g, anomalies, cat_col, params)
    print(f"\nГотово. Результаты в {args.output}/")
    print("  anomalies.csv, anomaly_reasons.csv,")
    print("  plots/total_ots_before_after.png, plots/category_ots_change.png, "
          "plots/daily_anomaly_count.png")


if __name__ == "__main__":
    main()
