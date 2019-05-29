How to contribute
=================
Found a bug? Have a new feature to suggest? Want to contribute changes to the codebase? Make sure to read this first.

A Note for Contributors
-----------------------
Both Videoflow-Contrib and Videoflow operate under the MIT License. At the discretion of the maintainers of both repositories, code may be moved from Videoflow-Contrib to Videoflow and vice versa.

The maintainers will ensure that the proper chain of commits will flow in both directions, with proper attribution of code. Maintainers will also do their best to notify contributors when their work is moved between repositories.


Bug reporting
-------------
Your code doesn't work, and you have determined that the issue lies with Videoflow? Follow
these steps to report a bug.

1. Your bug may already be fixed.  Make sure to update to the current
Videoflow master branch.

2. Search for similar issues. Make sure to delete `is:open` on the
issue search to find solved tickets as well. It's possible somebody
has encountered this bug already.  Still having a problem? Open an issue on Github
to let us know.

3. Make sure to provide us with useful information about
your configuration: What OS are you using? What Tensorflow version are you using?
Are you running on GPU? If so, what is your version of Cuda, of CuDNN? 
What is your GPU?

4. Provide us with a script to reproduce the issue.  This script should
be runnable as-is and should not require external data download
(use randomly generated data if you need to test the flow in some data).
We recommend that you use Github Gists to post your code.
Any issue that cannot be reproduced is likely to be closed.

5. If possible, take a shot at fixing the bug yourself --if you can!

The more information you provide, the easir it is for us to validate that
there is a bug and the faster we'll be able to take action.
If you want your issue to be resolved quickly, following the steps
above is crucial.

Pull Requests (PRs)
-------------------
**Where should I submit my pull request?** videoflow-contrib
improvements and bug gixes should go to the videoflow-contrib
`master` branch.

Here is a quick guide on how to submit your improvements::

1. Write the 
code.

2. Make sure any new function or class you introduce has
proper docstrings. Make sure any code you touch still
has up-to-date docstrings and documentation.  Use
previously written code as a reference on how to format
them.  In particular, they should be formatted in MarkDown,
and there should be sections for `Arguments`, `Returns` and
`Raises` (if applicable). 

3. Write tests. Your code should have full unit test coverage.
If you want to see your PRs merged promptly, this is crucial.

4. Run our test suite locally. It is easy: from the 
Videoflow folder, simply run ``py.test tests/``


5. Make sure all tests are 
passing.


6. When committing, use appropriate, descriptive 
commit messages.

7. Update the documentation.  If introducing new functionality,
make sure you include code snippets demonstrating the usage
of your new feature.

8. Submit your PR. If your changes have been approved in
a previous discussion, and if you have complete (and passing)
unit tests as well as proper doctrings/documentation, your
PR is likely to be merged promptly.