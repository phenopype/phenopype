move docs\_static _temp\_static
rmdir E:\git_repos\phenopype\_temp\docs /S /Q
rmdir E:\git_repos\phenopype\docs /S /Q
mkdir E:\git_repos\phenopype\_temp\docs
mkdir E:\git_repos\phenopype\docs
move _temp\_static docs\_static 
sphinx-multibuild -c E:\git_repos\phenopype -i tutorials -i _source -s _temp/docs -o docs
