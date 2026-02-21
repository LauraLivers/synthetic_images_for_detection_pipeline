# AICH Report Template
Chapters are organised in separate TeX-Files in the folder `include` and can be added to your document with the command `\include{file.tex}`.

Directory tree:

```text
Main directory
	|- appendix				All documents for the appendix (PDFs in this folder will not be ignored by the .gitignore)
	|- img					Images (are dynamically found through \graphicspath in the main document)
	|- include				Chapter files
	|- .gitignore
	|- report-template.tex	Main file
	|_ bibliography.bib			BibTeX file / Bibliography
```

## Command List

Run in order to build the project
* XeLaTeX
* Biber
* makeglossaries
* XeLaTeX
* XeLaTeX
