@echo off

if "%*"=="-d" (
	rmdir /Q /S docs
)

robocopy /S ../phenopype-tutorials docs_source/tutorials /XF README.md /xd .git .ipynb_checkpoints

sphinx-build docs_source docs
