"""Build the TIFS main paper and supplement. The supplement's figure, table and equation counters are set to continue
the numbering of the main paper (SPS rule for supplemental material)."""
import re
import subprocess


def latexmk(f, g=False):
    subprocess.run(["latexmk", "-pdf", "-interaction=nonstopmode", "-quiet"] + (["-g"] if g else []) + [f],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def roman(r):
    v = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    return sum(-v[c] if i + 1 < len(r) and v[c] < v[r[i + 1]] else v[c] for i, c in enumerate(r))


def sync_counters():
    aux = open("paper.aux").read()
    nf = max(int(m) for m in re.findall(r"\\newlabel\{fig:[^}]*\}\{\{(\d+)\}", aux))
    nt = max(roman(m) for m in re.findall(r"\\newlabel\{tab:[^}]*\}\{\{([IVXLC]+)\}", aux))
    ne = len(re.findall(r"\\begin\{equation\}", open("paper.tex").read()))
    na = max(int(m) for m in re.findall(r"\\newlabel\{alg:[^}]*\}\{\{(\d+)\}", aux))
    s = open("supplement.tex").read()
    s = re.sub(r"\\setcounter\{figure\}\{\d+\}", f"\\\\setcounter{{figure}}{{{nf}}}", s)
    s = re.sub(r"\\setcounter\{table\}\{\d+\}", f"\\\\setcounter{{table}}{{{nt}}}", s)
    s = re.sub(r"\\setcounter\{algorithm\}\{\d+\}", f"\\\\setcounter{{algorithm}}{{{na}}}", s)
    s = re.sub(r"\\setcounter\{equation\}\{\d+\}", f"\\\\setcounter{{equation}}{{{ne}}}", s)
    open("supplement.tex", "w").write(s)
    return nf, nt, ne, na


latexmk("paper.tex")
print("main counts", sync_counters())
latexmk("supplement.tex")
latexmk("paper.tex")
latexmk("supplement.tex")
latexmk("paper.tex", g=True)
