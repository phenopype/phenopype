RMDIR "dist/fix" /S /Q
python setup.py sdist --format=zip
dir /s /b dist\*.zip
7z x E:\git_repos\phenopype\dist\phenopype-1.0.5.zip -odist/fix
	cd dist/fixt 
	git init 
	git add . 
	git commit -m "- no version control in hotfixes -" 
	git remote add origin git@github.com:mluerig/phenopype
	git push --force origin master:fix
	cd ..
) 

