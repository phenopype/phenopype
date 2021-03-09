@echo off

m2r2 readme.md
move /Y readme.rst source/readme.rst

sphinx-multibuild -c . -i source -i tutorials  -s _temp/docs -o docs -a 

if "%~1"=="-o" (
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


