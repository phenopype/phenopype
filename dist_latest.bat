RMDIR "dist/latest" /S /Q
python setup.py sdist
robocopy phenopype dist/latest/phenopype /E
robocopy phenopype.egg-info dist/latest/phenopype.egg-info /E
robocopy "E:/git_repos/phenopype" dist/latest MANIFEST.in README.md setup.py
cd dist/latest
git init 
git add . 
git commit -m "- no version control in developmental version -" 
git remote add origin git@github.com:mluerig/phenopype
git push --force origin master:latest
cd ..
cd ..