# CI provenance and claim rules

CI evidence is identified by the exact Git commit and, for a release claim,
the immutable tag that selected that commit.

| Evidence | Commit | Relationship to `ted-v1.0.0` | Allowed statement |
| --- | --- | --- | --- |
| release-candidate run 29526380376 | `a312b387fc77f59d390ebd14db7fe7bfcddfd31d` | separate branch; not an ancestor of the tag | the named candidate branch passed its 10 jobs |
| archived tag | `5cb7b25458b41437b54623488d37b4872e79f474` | exact tagged commit | immutable source baseline; no clean full-suite pass is asserted |
| main run 29546657448 | `c6db8bff70aaa491be2c6f73ed00b65d2f487231` | direct child after compatibility fixes | post-release repair commit is green |

Do not convert either branch run into an exact-tag claim. For v1.0.1, retain
the workflow run URL, commit, tag, job matrix, JUnit, terminal summary,
installed-wheel smoke output, Docker logs, and release attestation together.

The package metadata declares Python `>=3.9,<3.14`; this is a compatibility
range, not a test result. The v1.0.1 candidate workflow tests installed wheels
on Linux Python 3.11, 3.12, and 3.13. Python 3.11 is the canonical locked
reproduction environment. Do not describe a matrix member as release-tested
until its exact-tag job passes.
