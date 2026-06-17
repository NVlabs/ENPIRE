---
name: pr
description: "Draft and submit a GitHub Pull Request following the project PR template. Reads recent commits, fills out description/motivation/test sections, checks the checklist interactively, then runs gh pr create."
allowed-tools: Agent Read Glob Grep Bash(git log*) Bash(git diff*) Bash(git status*) Bash(git branch*) Bash(gh pr*)
---

When you start a Pull Request on GitHub, you should write the pull request comment file following this format:

<!--- Provide a general summary of your changes in the Title above -->

### Description

### Motivation and Context

### How has this been tested?

### Additional information (optional, e.g., figures and logs):

### Types of changes
<!--- What types of changes does your code introduce? Put an `x` in all the boxes that apply: -->
- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation update
- [ ] Breaking change (fix or feature that would cause existing functionality to not work as expected)

### Checklist:
<!--- Go over all the following points, and put an `x` in all the boxes that apply. -->
<!--- If you're unsure about any of these, don't hesitate to ask. We're here to help! -->
- [ ] My code follows the code style of this project.
- [ ] My change requires a change to the documentation.
- [ ] I have updated the documentation accordingly.
- [ ] I have added tests to cover my changes.
- [ ] All new and existing tests passed.


Fill out these forms according to recent commits. Make sure you ask in terminal whether these tests are fulfilled, then complete the PR draft and let the human to review before submitting this PR. 