# Contributing

Want to contribute? Great! You can do so through the standard GitHub pull
request model. For large contributions we do encourage you to file a ticket in
the GitHub issues tracking system prior to any code development to coordinate
with the DL-benchmarks development team early in the process. Coordinating up
front helps to avoid frustration later on.  Please follow the pattern used in
the benchmarking files, e.g. defining a config class/struct containing
benchmarking parameters, measuring performance for both training and
inference, making sure every file contains licensing information, etc).

Your contribution should be licensed under the MIT license, the license used
by this project. If you want to also contribute code by others which is only
available under a different open source license, make sure that the license is
compatible with this project's license and that the file
[3rd-party-licenses.txt] (3rd-party-licenses.txt) is updated accordingly.

## Sign your work

This project tracks patch provenance and licensing using the Developer
Certificate of Origin (from [developercertificate.org][DCO]) and Signed-off-by
tags initially developed by the Linux kernel project.  

```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
660 York Street, Suite 102,
San Francisco, CA 94110 USA

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
```

With the sign-off in a commit message you certify that you authored the patch
or otherwise have the right to submit it under an open source license. The
procedure is simple: To certify above Developer's Certificate of Origin 1.1
for your contribution just append a line

    Signed-off-by: Random J Developer <random@developer.example.org>

to every commit message using your real name (sorry, no pseudonyms or
anonymous contributions).  If you have set your `user.name` and `user.email`
git configs you can automatically sign the commit by running the git-commit
command with the -s option.  There may be multiple sign-offs if more than one
developer was involved in authoring the contribution.

For a more detailed description of this procedure, please see
[SubmittingPatches][] which was extracted from the Linux kernel project, and
which is stored in an external repository.

[DCO]: http://developercertificate.org/
[SubmittingPatches]: https://github.com/wking/signed-off-by/blob/7d71be37194df05c349157a2161c7534feaf86a4/Documentation/SubmittingPatches
