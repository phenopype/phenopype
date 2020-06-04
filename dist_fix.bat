RMDIR "dist/fix" /S /Q
python setup.py sdist
robocopy phenopype dist/fix/phenopype
robocopy phenopype.egg-info dist/fix/phenopype.egg-info
robocopy "E:/git_repos/phenopype" dist/fix MANIFEST.in README.md setup.py
cd dist/fix 
git init 
git add . 
git commit -m "- no version control in hotfix -" 
git remote add origin git@github.com:mluerig/phenopype
git push --force origin master:fix
cd ..
cd ..