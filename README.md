# TsneBH.jl

The purpose of this module is to implement the T-SNE dimensionality reduction technique developed by [Laurens van der Maaten](https://lvdmaaten.github.io/tsne/). This technique is a stochastic algorithm that allows for the reduction of the dimensions from the original space, while trying to maintain intact the relationships among points in the reduced space, especially the nearest neighbors.

T-SNE itself is an extension of SNE technique, introducing the use of the t-distribution in the embedded space instead of the Gaussian and a new way to compute the gradient.

Essentially using T-SNE is solving an optimization problem, where the objective function is the KL divergence between the distributions of points in the original space (loosely speaking) and the ones in the reduced space. Ideally we'd like to minimize this cost, as to make the two distributions as similar as possible. The optimization is done through a gradient descent algorithm.

### Trees extension

An evolution of T-SNE is to accelerate the computations by means of two tree-based algorithms: Vantage Point trees and the Barnes-Hut. The first one is a clever way to map the space of points and to quickly retrieve which are the nearest neighbors of a given point. The second one, with the use of QuadTrees, is also a way to map the space of points but with the purpose of speeding up the computation of any interaction among them (in our case the gradient).

They are implemented in the `trees.jl` file, but as of now the BarnesHut functions are not stable and might give an Overflow error.


## Main function documentation

```julia
tsne(X::Matrix{Float64}, emb_size::Int64, T::Int64;
                lr::Float64 = 1., perp::Float64 = 30., tol::Float64 = 1e-5,
                max_iter::Int = 50,  momentum::Float64 = 0.01, 
                pca::Bool = true, pca_dim::Int = 50, exag_fact::Float64 = 4.,
                use_trees::Bool = false, ### The BarnesHut algorithm is currently instable, there's a problem with the recursion
                theta_bh::Float64 = 0.2, use_seed::Bool = false, verbose::Bool = true)
```

## Quick RunC
Clone the repo, then `cd` into it. You can run a simple example(after isntantiating the packages):

```julia
julia --project=. ./examples/tsne_run.jl
```
Otherwise you can open `julia --project=.` and do
```julia
using TsneBH
tsne(...) # follow the documentation above
```

## References

- L.J.P. van der Maaten and G.E. Hinton. Visualizing High-Dimensional Data Using t-SNE. Journal of Machine Learning Research 9(Nov):2579-2605, 2008.
- L.J.P. van der Maaten. Accelerating t-SNE using Tree-Based Algorithms. Journal of Machine Learning Research 15(Oct):3221-3245, 2014.
- [lvdmaaten.github.io/tsne](https://lvdmaaten.github.io/tsne/)

<!-- ## Getting started

To make it easy for you to get started with GitLab, here's a list of recommended next steps.

Already a pro? Just edit this README.md and make it your own. Want to make it easy? [Use the template at the bottom](#editing-this-readme)!

## Add your files

- [ ] [Create](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/repository/web_editor.html#create-a-file) or [upload](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/repository/web_editor.html#upload-a-file) files
- [ ] [Add files using the command line](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/gitlab-basics/add-file.html#add-a-file-using-the-command-line) or push an existing Git repository with the following command:

```
cd existing_repo
git remote add origin https://gitlab.com/p1222/tsnebh.jl.git
git branch -M master
git push -uf origin master
```

## Integrate with your tools

- [ ] [Set up project integrations](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://gitlab.com/p1222/tsnebh.jl/-/settings/integrations)

## Collaborate with your team

- [ ] [Invite team members and collaborators](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/members/)
- [ ] [Create a new merge request](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/merge_requests/creating_merge_requests.html)
- [ ] [Automatically close issues from merge requests](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/issues/managing_issues.html#closing-issues-automatically)
- [ ] [Enable merge request approvals](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/merge_requests/approvals/)
- [ ] [Automatically merge when pipeline succeeds](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/project/merge_requests/merge_when_pipeline_succeeds.html)

## Test and Deploy

Use the built-in continuous integration in GitLab.

- [ ] [Get started with GitLab CI/CD](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/ci/quick_start/index.html)
- [ ] [Analyze your code for known vulnerabilities with Static Application Security Testing(SAST)](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/application_security/sast/)
- [ ] [Deploy to Kubernetes, Amazon EC2, or Amazon ECS using Auto Deploy](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/topics/autodevops/requirements.html)
- [ ] [Use pull-based deployments for improved Kubernetes management](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/user/clusters/agent/)
- [ ] [Set up protected environments](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://docs.gitlab.com/ee/ci/environments/protected_environments.html)

***

# Editing this README

When you're ready to make this README your own, just edit this file and use the handy template below (or feel free to structure it however you want - this is just a starting point!).  Thank you to [makeareadme.com](https://gitlab.com/-/experiment/new_project_readme_content:9bb415accb2176f43168a10537f2ef8f?https://www.makeareadme.com/) for this template.

## Suggestions for a good README
Every project is different, so consider which of these sections apply to yours. The sections used in the template are suggestions for most open source projects. Also keep in mind that while a README can be too long and detailed, too long is better than too short. If you think your README is too long, consider utilizing another form of documentation rather than cutting out information.

## Name
Choose a self-explaining name for your project.

## Description
Let people know what your project can do specifically. Provide context and add a link to any reference visitors might be unfamiliar with. A list of Features or a Background subsection can also be added here. If there are alternatives to your project, this is a good place to list differentiating factors.

## Badges
On some READMEs, you may see small images that convey metadata, such as whether or not all the tests are passing for the project. You can use Shields to add some to your README. Many services also have instructions for adding a badge.

## Visuals
Depending on what you are making, it can be a good idea to include screenshots or even a video (you'll frequently see GIFs rather than actual videos). Tools like ttygif can help, but check out Asciinema for a more sophisticated method.

## Installation
Within a particular ecosystem, there may be a common way of installing things, such as using Yarn, NuGet, or Homebrew. However, consider the possibility that whoever is reading your README is a novice and would like more guidance. Listing specific steps helps remove ambiguity and gets people to using your project as quickly as possible. If it only runs in a specific context like a particular programming language version or operating system or has dependencies that have to be installed manually, also add a Requirements subsection.

## Usage
Use examples liberally, and show the expected output if you can. It's helpful to have inline the smallest example of usage that you can demonstrate, while providing links to more sophisticated examples if they are too long to reasonably include in the README.

## Support
Tell people where they can go to for help. It can be any combination of an issue tracker, a chat room, an email address, etc.

## Roadmap
If you have ideas for releases in the future, it is a good idea to list them in the README.

## Contributing
State if you are open to contributions and what your requirements are for accepting them.

For people who want to make changes to your project, it's helpful to have some documentation on how to get started. Perhaps there is a script that they should run or some environment variables that they need to set. Make these steps explicit. These instructions could also be useful to your future self.

You can also document commands to lint the code or run tests. These steps help to ensure high code quality and reduce the likelihood that the changes inadvertently break something. Having instructions for running tests is especially helpful if it requires external setup, such as starting a Selenium server for testing in a browser.

## Authors and acknowledgment
Show your appreciation to those who have contributed to the project.

## License
For open source projects, say how it is licensed.

## Project status
If you have run out of energy or time for your project, put a note at the top of the README saying that development has slowed down or stopped completely. Someone may choose to fork your project or volunteer to step in as a maintainer or owner, allowing your project to keep going. You can also make an explicit request for maintainers.
 -->
