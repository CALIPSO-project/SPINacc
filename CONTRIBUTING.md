HOW TO UPDATE THE CODE ON GITHUB: you need to do multiple steps:

* First, "git add <file_or_dir_you_changed>" to add the files you changed and " git add ." to add all of your changes. It is recommended to only add files you have changed and make sure you updated with any changes updated to github since you downloaded your copies.
* Second, "git commit -m 'Quick commit message that described the changes you are commiting'" to commit them to your local copy (a difference between svn and git and is that git has a local copy that you commit to).
* Third, "git push origin main" to push them to the master repository (here) and "git push origin <branch_name>" to push them to a branch.
It is also recommended to push all changes on a personal branch and then ask for a pull request to the administrator of the tool. Then comment your changes for an easy review.
This might help: https://git-scm.com/docs/gittutorial


### Other USEFUL COMMANDS:
* "git diff" will show you all the changes you have made.
* "git pull" will update all your local files with what exists in the master code (here).
* "git status" will show you what files have been modified.
