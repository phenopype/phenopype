@echo off

if "%*"=="-d" (
	rmdir /Q /S _temp
)

sphinx-multibuild -i docs_source -i tutorials  -s _temp/docs -o docs -a

if "%*"=="-o" (
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
