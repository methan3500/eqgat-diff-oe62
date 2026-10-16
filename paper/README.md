# Building the manuscript

`paper_lncs.tex` targets Springer's LNCS class. The class file is **not**
included, because Springer distributes it under its own terms. Get
`llncs.cls` from the
[Springer LNCS author kit](https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)
or from CTAN (`texlive-llncs`), put it next to the `.tex`, then:

```bash
pdflatex paper_lncs
bibtex   paper_lncs      # only if you switch the bibliography to BibTeX
pdflatex paper_lncs
pdflatex paper_lncs
```

On Overleaf, start a blank project, upload `paper_lncs.tex`, and add the LNCS
class from Overleaf's own *Springer Lecture Notes in Computer Science*
template rather than uploading a copy.

Figures are drawn with PGFPlots and TikZ inside the document, so there are no
external image files to keep in sync. The packages needed beyond the class are
`graphicx`, `amsmath`, `booktabs`, `siunitx`, `tikz`, `pgfplots` and
`hyperref` — all standard in TeX Live and on Overleaf.

## Before submitting

- The bibliography entries are written out by hand in the `thebibliography`
  environment. Check each one against the published record; page numbers and
  volume numbers in particular have not been verified against the journals.
- Section *Results* reports a single sampling run of 3,000 molecules. If you
  rerun it, the numbers move; the text states them to four significant figures
  and they are not re-derivable from anything else in the repository.
- The multi-iteration biasing trajectory is deliberately absent. Do not add a
  table for it without running the loop.
