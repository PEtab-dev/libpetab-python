# petab contribution guide

We are happy about all kinds of contributions to PEtab,
from bug reports and documentation fixes to new features.

## Contributions to this repository

General:

* For code contributions, please adhere to the
  [PEP8 style guide](https://www.python.org/dev/peps/pep-0008/)
* All new functionality should be covered by unit tests
* Please use Python type hints
* Please document all modules, functions, classes, attributes, arguments,
  return values, ... in a style consistent with the rest of the library
* Use descriptive commit messages
* For user-visible changes, add a changelog entry to `changelog.d/`
  (see [`changelog.d/README.md`](changelog.d/README.md))

To contribute to this repository:

* Read our [contribution policy](#contribution-policy) below

* Create a pull request

  *By creating a pull request you agree with your contribution being made
  available under the license terms specified in
  [https://github.com/PEtab-dev/libpetab-python/blob/main/LICENSE](https://github.com/PEtab-dev/libpetab-python/blob/main/LICENSE).*

* Assign a reviewer, or let us know otherwise, that your pull request is ready
  for review.

* Wait for feedback (and feel free to send a gentle reminder if you did not
  get any feedback within a week)

## Contribution policy

We use AI tools ourselves and have no objection to contributors using them.
However, reviewing takes maintainer time, often more than writing the change.
We need a human who understands the change, stands behind it,
and has made it fit the project's architecture, conventions, and style.
The following applies to all contributions:
code, documentation, issues, and comments.

1. **Disclose any use of AI.**
   State in the pull request or issue which AI tools you used and for which
   parts (e.g., "implementation drafted with Claude Code and revised by hand,
   tests written manually").
   Using AI only to understand or debug the code, or for routine
   autocompletion, does not need to be disclosed.
   Pull requests with apparently undisclosed use of AI may be closed
   without review.

2. **You are responsible for your contribution.**
   Review and understand every change, and run the relevant tests before
   submitting.
   Make sure that AI-generated content can be released under this
   project's license.

3. **We talk to you, not to your agent.**
   AI tools may draft the initial issue or pull request description,
   but respond to reviews and discussions yourself.
   If a pull request merely relays between us and an AI tool,
   it is easier for us to use the AI tool ourselves, and we may close it.

4. **Consider discussing first.**
   For anything beyond a small, obvious fix (e.g., a new feature, changes
   to the public API, or more than a few dozen lines), we recommend opening
   an issue or commenting on an existing one first.
   Otherwise, we may decline the pull request on conceptual grounds without
   reviewing the implementation.

5. **We may close contributions that cost more than they are worth.**
   If getting a pull request into shape would take us more time than
   making the change ourselves, we may close it and address the underlying
   issue separately.
   This mostly affects largely AI-generated contributions that do not meet
   the standards above and in the
   [developer's guide](https://petab.readthedocs.io/projects/libpetab-python/en/latest/development.html).

6. **Rough bug reports are welcome.**
   They may be drafted with AI tools if disclosed, but the problem must have
   been reproduced: include a minimal PEtab problem or script we can run.
   AI-generated analyses of the cause are fine if marked as such, but they
   do not replace the reproducible example.
   Reports without one, e.g., problems inferred only from reading the code,
   may be closed.

If we close your contribution with a link to this section, it is not a
judgement of you or your work; we have to prioritize.
