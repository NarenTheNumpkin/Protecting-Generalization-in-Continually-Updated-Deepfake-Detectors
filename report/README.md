# Project report (Word)

Builds the BCSE497J project report from the university's Word template and the text in `content/`.

```bash
cd report
# 1. figures: plots (300 dpi) and diagrams
python ../paper/make_figures.py ../experiments figs
python ../paper/make_report_figures.py ../experiments figs
for f in workflow gend stages overview; do (cd diagrams && pdflatex $f.tex && pdftoppm -r 300 -png -singlefile $f.pdf ../figs/diag_$f); done
# 2. two passes: render with LibreOffice to measure page numbers for the contents pages, then rebuild
python build_report.py template.docx out/report_pass1.docx
python measure.py out/report_pass1.docx out/pages.json
python build_report.py template.docx out/report.docx out/pages.json
```

`template.docx` is the university's report template (not included). Requires python-docx, pypandoc_binary
(LaTeX equations to Word equations) and LibreOffice for the page measurement. Personal details are left as
highlighted placeholders. Word asks to update fields when the document is opened; answering yes refreshes the
contents and lists.
