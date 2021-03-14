@echo off

pandoc README.md -o source/readme.inc --to rst

if "%~1"=="-d" (
	rmdir /Q /S _temp
)

sphinx-multibuild -c . -i source -i tutorials  -s _temp/docs -o docs -a

if "%~2"=="-o" (
	cd docs
	rmdir ".git" /S /Q
	git init
	@echo .doctrees > docs/.gitignore
	git add .
	git commit -m "- no version control in docs -"
	git remote add origin git@github.com:mluerig/phenopype
	git push --force origin master:gh-pages
	cd ..
)
