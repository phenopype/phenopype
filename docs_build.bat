RMDIR "docs" /S /Q
md docs/_temp
sphinx-multibuild -c E:\git_repos\phenopype -i tutorials -i docs_source -s docs_temp -o docs
