## Workflows guideline
We now have 6 workflows on github action. I explain them here quickly so not everyone needs to dig deep into the code.
#### bump-version
It only triggers when there are changes added to main branch (push or merge). When the changes are not negligible, it will update our version of release and create a tag marking our version. The scale of changes are determined by the commit messages ([which is called conventional commits](https://www.conventionalcommits.org/en/v1.0.0/)). For exmple a commit message like:
```
docs: correct spelling of CHANGELOG
```
Here the __docs__ tag is treated as negligible change, therefore the version won't go up and Changelog won't be updated. However a __feat__ tag will be treated as minor changes as an example, version can go from *0.1.1* to *0.2.0*

    feat(ci): automate changelog, docs deploy, and lint ignores
  while a __fix__ tag will only change version from *0.1.1* to *0.1.2*. More information can be found in the link or asking LLM: "I have done blablabla, which tag should I use regarding __conventional commits__."

The non-negligible commits (for example __feat__ or __fix__) formatted in __conventional commits__ will be updated in our CHANGELOG.md.
#### unit-test
This workflow is going to fail since we dont have any unit-test file yet. 
#### Sphinx: Render doc
This is for rendering our static documentation page. It shouldn't fail.
#### pages-build-deployment
This will deploy our static documentation page on github. It shouldn't fail.
#### Install & Import shaqip_student
This will check our dependancy. It shouldn't fail. 
#### code-quality
This workflow will use pre-commit to run  __ruff__ and __mypy__, which check our formatting, linting and type checking. In order to pass this workflow, I recommand to run pre-commit locally before pushing changes to the repository: After install pre-commit locally, run:

    uv run pre-commit run --all-files
__Important__: Changes without passing this workflow shouldn't be merged into the main branch.

## Short Summary (TL;dR)
Since we don't have any unit-test yet, only the unit-test workflow supposes to fail. I think there are mainly two things that might happen at times.
* If __code-quality__ fails, you might check the error messages from workflow and modify the code accordingly. Using pre-commit locally before  push.
* Make sure using conventional commit format to commit changes. AND use the correct tag is crucial! Not using it won't cause any error but it won't update our changelog and release version. (also very bad)

If you are facing other issues about the workflows other than these two listed above,  lemme know!



 
