# Changelog fragments

Each pull request with user-visible changes adds its changelog entry as a
separate file to this directory, instead of editing `CHANGELOG.md` directly.
This avoids merge conflicts between pull requests. At release time, the
fragments are collected into `CHANGELOG.md` using
[towncrier](https://towncrier.readthedocs.io/).

## Adding an entry

Create a file `changelog.d/<name>.<type>.md`, where `<type>` is one of

* `breaking`: **Breaking changes**
* `deprecation`: **Deprecations**
* `feature`: **Features**
* `fix`: **Fixes**
* `perf`: **Performance**
* `doc`: **Documentation**
* `misc`: **Other changes**

and `<name>` is `+` followed by a short descriptive name
(e.g., `+unused-placeholders.feature.md`). The name only needs to be unique;
it does not appear in `CHANGELOG.md`.

The file contains the text of the entry as Markdown, without a leading
bullet, hard-wrapped at 79 characters. Describe the change from the user's
perspective, and end with a link to the pull request.
For example, `changelog.d/+unused-placeholders.feature.md`:

```markdown
New validation task `CheckUnusedPlaceholders` warns about observable and
noise placeholders that are not used in the respective formulas
([#599](https://github.com/PEtab-dev/libpetab-python/pull/599))
```

Changes that don't affect users, such as CI-only changes or refactoring,
don't need an entry.

You can also create the file using `towncrier create`, e.g.,
`towncrier create +unused-placeholders.feature.md`. towncrier is included in
the `test` dependency group.

To preview the changelog section for the next release, run
`towncrier build --draft --version <next version>` from the repository root.
