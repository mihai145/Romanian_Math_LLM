import json
from pathlib import Path
import matplotlib.pyplot as plt


PARENT_DIR = Path("lm_eval_out")
OUT_PNG = Path("gather.png")
BENCHES = ["ro_gsm8k", "ro_mathqa"]


def is_chat(d):
    return d.get("chat_template") is not None


def score(d, bench):
    r = d["results"][bench]
    return (r["exact_match,final_number"] if bench == "ro_gsm8k" else r["acc,none"]) * 100.0


def parse_folder(name):
    if "finetuned_" not in name:
        return name.replace("__", "_"), "base"
    tail = name.split("finetuned_", 1)[1]
    steps, rest = tail.split("_", 1)
    ds = next(b for b in BENCHES if f"_{b}_" in rest)
    base = rest.split(f"_{ds}_", 1)[0].replace("__", "_")
    return base, f"ft {ds} {steps}"


def main():
    raw = {}
    for mdir in sorted(p for p in PARENT_DIR.iterdir() if p.is_dir()):
        for bench in BENCHES:
            bdir = mdir / bench
            if not bdir.is_dir():
                continue
            for rdir in sorted(p for p in bdir.iterdir() if p.is_dir()):
                f = sorted(rdir.glob("results_*.json"))[0]
                d = json.loads(f.read_text())
                kind = "chat" if is_chat(d) else "nochat"
                raw.setdefault(mdir.name, {}).setdefault(bench, {})[kind] = score(d, bench)

    grouped = {}
    for folder, m in raw.items():
        base, tag = parse_folder(folder)
        tup = (m["ro_gsm8k"]["chat"], m["ro_gsm8k"]["nochat"], m["ro_mathqa"]["chat"], m["ro_mathqa"]["nochat"])
        grouped.setdefault(base, {})[tag] = tup

    col_labels = ["Model", "gsm8k chat", "gsm8k nochat", "mathqa chat", "mathqa nochat"]
    rows, colors, is_base_row = [], [], []

    for base in sorted(grouped):
        b = grouped[base]["base"]
        rows.append([f"{base} (base)"] + [f"{x:.2f}" for x in b])
        colors.append(["black"] * 5)
        is_base_row.append(True)

        for tag in sorted(t for t in grouped[base] if t != "base"):
            ft = grouped[base][tag]
            deltas = [ft[i] - b[i] for i in range(4)]
            rows.append([f"     {base} ({tag})"] + [f"{d:+.2f}" for d in deltas])
            colors.append(["black"] + [("green" if d > 0 else ("red" if d < 0 else "black")) for d in deltas])
            is_base_row.append(False)

    fig_h = max(2.0, 0.42 * (len(rows) + 1))
    fig, ax = plt.subplots(figsize=(16, fig_h))
    ax.axis("off")

    tbl = ax.table(
        cellText=rows,
        colLabels=col_labels,
        cellLoc="center",
        colWidths=[0.5, 0.125, 0.125, 0.125, 0.125],
        bbox=[0, 0, 1, 1],
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.35)

    for r in range(len(rows) + 1):
        tbl[(r, 0)].get_text().set_ha("left")

    for r in range(1, len(rows) + 1):
        if is_base_row[r - 1]:
            for c in range(5):
                tbl[(r, c)].set_facecolor((0.92, 0.92, 0.92))
        for c in range(5):
            tbl[(r, c)].get_text().set_color(colors[r - 1][c])

    fig.savefig(OUT_PNG, dpi=200)


if __name__ == "__main__":
    main()
